/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Micro-benchmark for the chat-template render on the request path: the
// ChatMessages -> JSON conversion plus the minja render of a real-world
// template, exactly the call LLMRequestFactory::create(messages, ...) makes.
//
// The other hop benchmarks stub the chat template out (llm_request_factory
// benchmark uses a FakeChatTemplate), so this is the only place the render
// itself is measured. The conversation shape mirrors
// api_service_benchmark's make_chat_json (system + alternating user/assistant
// turns of 256 chars) so the two hops can be compared per request.
//
//   * BM_ChatTemplate_JinjaRender          - n messages, no tools
//   * BM_ChatTemplate_JinjaRenderWithTools - a 4-turn tool-calling
//                                            conversation plus n tools, which
//                                            exercises the `tool | tojson`
//                                            branch of the template
//   * BM_ChatTemplate_MinjaApply           - the n-message conversation
//                                            through
//                                            minja::chat_template::apply
//                                            directly (the pre-native path)
//   * BM_ChatTemplate_MinjaBuiltins        - constructing minja's builtin
//                                            globals, which apply does per
//                                            render
//   * BM_ChatTemplate_MessagesToJson       - the ChatMessages -> ordered_json
//                                            conversion alone
//   * BM_ChatTemplate_MinjaRenderCachedBuiltins - JSON -> minja::Value,
//                                            context setup and render with
//                                            cached builtins
//   * BM_ChatTemplate_AllocatorControl     - eight small malloc/free pairs, the
//                                            allocator floor under all of it
//   * BM_ChatTemplate_DeepseekConversation - V3.2/V4 ordinary conversations
//   * BM_ChatTemplate_DeepseekToolResults  - consecutive, ordered or reversed
//                                            tool results after one assistant
//   * BM_ChatTemplate_DeepseekToolSchemas  - fixed input with n schemas
//
// Build in Release mode before comparing latency. JinjaRender and Deepseek
// cases include per-request conversion, preprocessing, rendering,
// classification and result destruction, but exclude template construction and
// request setup. Run (example, using a Release build):
//   python setup.py test --test-name chat_template_benchmark
//   ./chat_template_benchmark --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>
#include <glog/logging.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iterator>
#include <memory>
#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/chat_template/chat_template.h"
#include "core/framework/chat_template/deepseek_v32_cpp_template.h"
#include "core/framework/chat_template/deepseek_v4_cpp_template.h"
#include "core/framework/chat_template/jinja_chat_template.h"
#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {
namespace {

// benchmark 1.8.x exposes only DoNotOptimize(Tp const&) -- deprecated -- and
// DoNotOptimize(Tp&); there is no rvalue overload, so a temporary or a const
// local resolves to the deprecated one. Sinking the value into a non-const
// parameter first selects the supported overload.
template <typename T>
inline BENCHMARK_ALWAYS_INLINE void do_not_optimize(T value) {
  benchmark::DoNotOptimize(value);
}

// Match the per-render callable and time capture in JinjaChatTemplate.
minja::Value make_strftime_now() {
  const auto now = std::chrono::system_clock::now();
  return minja::Value::callable(
      [now](const std::shared_ptr<minja::Context>& /*context*/,
            minja::ArgumentsValue& args) -> minja::Value {
        args.expectArgs("strftime_now", {1, 1}, {0, 0});
        const std::string format = args.args[0].get<std::string>();
        const std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::tm local_time{};
        CHECK(localtime_r(&time, &local_time) != nullptr);
        std::ostringstream out;
        out << std::put_time(&local_time, format.c_str());
        return minja::Value(out.str());
      });
}

// Qwen2.5's published chat template: a system-prompt branch that serialises
// the tool list, per-role message rendering including tool calls and tool
// responses, and the generation prompt. A representative production template
// rather than a toy loop.
constexpr char kQwen25ChatTemplate[] = R"JINJA({%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
)JINJA";

constexpr size_t kTurnChars = 256;

TokenizerArgs make_tokenizer_args() {
  TokenizerArgs args;
  args.chat_template(kQwen25ChatTemplate);
  args.bos_token("");
  args.eos_token("<|im_end|>");
  return args;
}

// System prompt plus alternating user / assistant turns, the same shape as
// api_service_benchmark's make_chat_json.
ChatMessages make_messages(size_t num_messages) {
  ChatMessages messages;
  messages.reserve(num_messages);
  messages.emplace_back("system", "You are a helpful assistant.");
  for (size_t i = 1; i < num_messages; ++i) {
    messages.emplace_back(i % 2 == 1 ? "user" : "assistant",
                          std::string(kTurnChars, 'y'));
  }
  return messages;
}

// A completed tool call round trip: the assistant turn carries tool_calls and
// the tool turn answers it, so both template branches run.
ChatMessages make_tool_messages() {
  ChatMessages messages;
  messages.reserve(4);
  messages.emplace_back("system", "You are a helpful assistant.");
  messages.emplace_back("user", "What is the weather in Paris?");
  messages.emplace_back("assistant", "");
  Message::ToolCall tool_call;
  tool_call.id = "call_1";
  tool_call.type = "function";
  tool_call.function.name = "get_weather_0";
  tool_call.function.arguments = R"({"city":"Paris"})";
  messages.back().tool_calls = Message::ToolCallVec{tool_call};
  messages.emplace_back("tool", "18C, cloudy");
  messages.back().tool_call_id = "call_1";
  return messages;
}

std::vector<JsonTool> make_tools(size_t num_tools) {
  std::vector<JsonTool> tools;
  tools.reserve(num_tools);
  for (size_t i = 0; i < num_tools; ++i) {
    const nlohmann::json parameters = {
        {"type", "object"},
        {"properties",
         {{"city",
           {{"type", "string"},
            {"description", "The city to look up, e.g. Paris"}}},
          {"unit", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}}},
          {"days", {{"type", "integer"}, {"minimum", 1}, {"maximum", 14}}}}},
        {"required", {"city"}}};
    tools.emplace_back(
        "function",
        JsonFunction("get_weather_" + std::to_string(i),
                     "Get the current weather and forecast for a city.",
                     parameters));
  }
  return tools;
}

size_t content_bytes(const ChatMessages& messages) {
  size_t bytes = 0;
  for (const auto& message : messages) {
    if (const auto* text = std::get_if<std::string>(&message.content)) {
      bytes += text->size();
    }
  }
  return bytes;
}

// The range is the exact message count, including the system message when
// present. Every conversation ends with a user requesting a new response.
ChatMessages make_deepseek_messages(size_t num_messages) {
  ChatMessages messages;
  messages.reserve(num_messages);
  if (num_messages == 1) {
    messages.emplace_back("user", std::string(kTurnChars, 'u'));
    return messages;
  }
  messages.emplace_back("system", "You are a helpful assistant.");
  for (size_t i = 1; i < num_messages; ++i) {
    messages.emplace_back(i % 2 == 1 ? "user" : "assistant",
                          std::string(kTurnChars, 'y'));
    if (i % 2 == 0) {
      messages.back().reasoning_content = std::string(kTurnChars, 'r');
    }
  }
  return messages;
}

// All calls use the same function so this case scales tool-call/result
// processing independently of schema serialization. Reverse arrival exercises
// V4's sorting path; ordered arrival exercises its no-reordering path.
ChatMessages make_deepseek_tool_messages(size_t num_results,
                                         bool reverse_results) {
  ChatMessages messages;
  messages.reserve(num_results + 3);
  messages.emplace_back("system", "You are a helpful assistant.");
  messages.emplace_back("user", "Look up the weather for these cities.");
  messages.emplace_back("assistant", "");
  messages.back().reasoning_content = std::string(kTurnChars, 'r');
  Message::ToolCallVec calls;
  calls.reserve(num_results);
  for (size_t i = 0; i < num_results; ++i) {
    Message::ToolCall call;
    call.id = "call_" + std::to_string(i);
    call.type = "function";
    call.function.name = "get_weather_0";
    call.function.arguments = R"({"city":"Paris"})";
    calls.emplace_back(std::move(call));
  }
  messages.back().tool_calls = std::move(calls);
  for (size_t i = 0; i < num_results; ++i) {
    const size_t call_index = reverse_results ? num_results - i - 1 : i;
    messages.emplace_back(
        "tool",
        std::to_string(call_index) + ":" + std::string(kTurnChars, 't'));
    messages.back().tool_call_id = "call_" + std::to_string(call_index);
  }
  return messages;
}

uint32_t prompt_fingerprint(const std::string& prompt) {
  // A stable, exactly representable benchmark counter makes accidental output
  // changes visible when comparing runs. Unit tests remain the semantic oracle.
  uint32_t fingerprint = 2166136261U;
  for (const unsigned char byte : prompt) {
    fingerprint = (fingerprint ^ byte) * 16777619U;
  }
  return fingerprint;
}

template <typename Template>
void run_deepseek_benchmark(benchmark::State& state,
                            const ChatMessages& messages,
                            const std::vector<JsonTool>& tools,
                            bool thinking) {
  const TokenizerArgs tokenizer_args;
  const Template chat_template(tokenizer_args);
  const nlohmann::ordered_json kwargs = {{"thinking", thinking},
                                         {"reasoning_effort", "low"}};
  const std::optional<ChatTemplateRenderResult> expected =
      chat_template.apply_with_generation_mode(messages, tools, kwargs);
  const std::optional<ChatTemplateRenderResult> repeated =
      chat_template.apply_with_generation_mode(messages, tools, kwargs);
  const ChatTemplateGenerationMode expected_mode =
      thinking ? ChatTemplateGenerationMode::REASONING
               : ChatTemplateGenerationMode::CHAT;
  if (!expected.has_value() || expected->prompt.empty() ||
      !repeated.has_value() || repeated->prompt != expected->prompt ||
      expected->generation_mode != expected_mode ||
      repeated->generation_mode != expected_mode) {
    state.SkipWithError(
        "DeepSeek render failed or produced inconsistent output");
    return;
  }

  for (auto _ : state) {
    std::optional<ChatTemplateRenderResult> rendered =
        chat_template.apply_with_generation_mode(messages, tools, kwargs);
    if (!rendered.has_value()) {
      state.SkipWithError("DeepSeek render failed");
      break;
    }
    benchmark::DoNotOptimize(*rendered);
  }
  state.SetItemsProcessed(state.iterations());  // One item is one request.
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(expected->prompt.size()));
  state.counters["messages"] = static_cast<double>(messages.size());
  state.counters["schemas"] = static_cast<double>(tools.size());
  state.counters["prompt_bytes"] = static_cast<double>(expected->prompt.size());
  state.counters["prompt_fingerprint"] =
      static_cast<double>(prompt_fingerprint(expected->prompt));
}

template <typename Template>
void BM_ChatTemplate_DeepseekConversation(benchmark::State& state) {
  const ChatMessages messages =
      make_deepseek_messages(static_cast<size_t>(state.range(0)));
  const std::vector<JsonTool> no_tools;
  run_deepseek_benchmark<Template>(
      state, messages, no_tools, state.range(1) != 0);
}

template <typename Template>
void BM_ChatTemplate_DeepseekToolResults(benchmark::State& state) {
  const ChatMessages messages = make_deepseek_tool_messages(
      static_cast<size_t>(state.range(0)), state.range(2) != 0);
  const std::vector<JsonTool> tools = make_tools(/*num_tools=*/1);
  run_deepseek_benchmark<Template>(state, messages, tools, state.range(1) != 0);
  state.counters["tool_results"] = static_cast<double>(state.range(0));
}

template <typename Template>
void BM_ChatTemplate_DeepseekToolSchemas(benchmark::State& state) {
  const ChatMessages messages = make_tool_messages();
  const std::vector<JsonTool> tools =
      make_tools(static_cast<size_t>(state.range(0)));
  run_deepseek_benchmark<Template>(state, messages, tools, state.range(1) != 0);
}

void BM_ChatTemplate_JinjaRender(benchmark::State& state) {
  const JinjaChatTemplate chat_template(make_tokenizer_args());
  const ChatMessages messages =
      make_messages(static_cast<size_t>(state.range(0)));
  const std::vector<JsonTool> no_tools;
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();

  for (auto _ : state) {
    std::optional<ChatTemplateRenderResult> rendered =
        chat_template.apply_with_generation_mode(messages, no_tools, kwargs);
    if (!rendered.has_value()) {
      state.SkipWithError("chat template render failed");
      break;
    }
    do_not_optimize(rendered->prompt.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(content_bytes(messages)));
}

void BM_ChatTemplate_JinjaRenderWithTools(benchmark::State& state) {
  const JinjaChatTemplate chat_template(make_tokenizer_args());
  const ChatMessages messages = make_tool_messages();
  const std::vector<JsonTool> tools =
      make_tools(static_cast<size_t>(state.range(0)));
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();

  for (auto _ : state) {
    std::optional<ChatTemplateRenderResult> rendered =
        chat_template.apply_with_generation_mode(messages, tools, kwargs);
    if (!rendered.has_value()) {
      state.SkipWithError("chat template render failed");
      break;
    }
    do_not_optimize(rendered->prompt.data());
  }
}

// The same conversation through minja::chat_template::apply directly -- the
// path JinjaChatTemplate used before rendering natively -- for the A/B.
void BM_ChatTemplate_MinjaApply(benchmark::State& state) {
  const minja::chat_template chat_template(
      kQwen25ChatTemplate, "", "<|im_end|>");
  const ChatMessages messages =
      make_messages(static_cast<size_t>(state.range(0)));
  nlohmann::ordered_json messages_json = nlohmann::ordered_json::array();
  messages_json.get_ref<nlohmann::ordered_json::array_t&>().reserve(
      messages.size());
  for (const auto& message : messages) {
    messages_json.emplace_back(nlohmann::ordered_json{
        {"role", message.role},
        {"content", std::get<std::string>(message.content)}});
  }

  for (auto _ : state) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages_json;
    inputs.tools = nlohmann::ordered_json::array();
    inputs.add_generation_prompt = true;
    inputs.extra_context = nlohmann::ordered_json::object();
    std::string prompt =
        chat_template.apply(inputs, minja::chat_template_options());
    do_not_optimize(prompt.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(content_bytes(messages)));
}

// The builtin globals minja::chat_template::apply constructs on every render
// and JinjaChatTemplate now builds once.
void BM_ChatTemplate_MinjaBuiltins(benchmark::State& state) {
  for (auto _ : state) {
    std::shared_ptr<minja::Context> builtins = minja::Context::builtins();
    do_not_optimize(builtins.get());
  }
}

// --------------------------------------------------------------------------
// Attribution of JinjaRender against MinjaApply
//
// JinjaRender = ChatMessages -> ordered_json conversion with polyfill flags
//             + native context setup/render + generation-mode classification.
// The two below measure message conversion and native context setup/rendering
// separately; they do not isolate every term in the end-to-end comparison.
// --------------------------------------------------------------------------

// The text-only ChatMessages -> ordered_json conversion. This isolates JSON
// construction and excludes conversion-time polyfill feature tracking.
void BM_ChatTemplate_MessagesToJson(benchmark::State& state) {
  const ChatMessages messages =
      make_messages(static_cast<size_t>(state.range(0)));

  for (auto _ : state) {
    nlohmann::ordered_json messages_json = nlohmann::ordered_json::array();
    messages_json.get_ref<nlohmann::ordered_json::array_t&>().reserve(
        messages.size());
    for (const auto& message : messages) {
      nlohmann::ordered_json message_json;
      message_json["role"] = message.role;
      message_json["content"] = std::get<std::string>(message.content);
      messages_json.emplace_back(std::move(message_json));
    }
    benchmark::DoNotOptimize(messages_json);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(content_bytes(messages)));
}

// Native rendering from pre-built JSON includes its recursive conversion to
// minja::Value, context setup, rendering and destruction, using a parsed root
// and cached builtins. MinjaApply additionally copies the JSON into its inputs
// and performs minja's input adaptation/capability checks.
void BM_ChatTemplate_MinjaRenderCachedBuiltins(benchmark::State& state) {
  const std::shared_ptr<minja::TemplateNode> root =
      minja::Parser::parse(kQwen25ChatTemplate,
                           {/*trim_blocks=*/true,
                            /*lstrip_blocks=*/true,
                            /*keep_trailing_newline=*/false});
  const std::shared_ptr<minja::Context> builtins = minja::Context::builtins();
  const ChatMessages messages =
      make_messages(static_cast<size_t>(state.range(0)));
  nlohmann::ordered_json messages_json = nlohmann::ordered_json::array();
  messages_json.get_ref<nlohmann::ordered_json::array_t&>().reserve(
      messages.size());
  for (const auto& message : messages) {
    messages_json.emplace_back(nlohmann::ordered_json{
        {"role", message.role},
        {"content", std::get<std::string>(message.content)}});
  }
  const nlohmann::ordered_json tools = nlohmann::ordered_json::array();

  for (auto _ : state) {
    minja::Value values = minja::Value::object();
    values.set("messages", minja::Value(messages_json));
    values.set("add_generation_prompt", minja::Value(true));
    auto context = minja::Context::make(std::move(values), builtins);
    context->set("bos_token", minja::Value(""));
    context->set("eos_token", minja::Value("<|im_end|>"));
    context->set("strftime_now", make_strftime_now());
    context->set("tools", minja::Value(tools));
    std::string prompt = root->render(context);
    do_not_optimize(prompt.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(content_bytes(messages)));
}

// Control: eight representative small allocations, freed in reverse. This
// measures allocator cost on the current platform, without reproducing the
// complete allocation pattern of a message conversion.
void BM_ChatTemplate_AllocatorControl(benchmark::State& state) {
  constexpr size_t kSizes[] = {32, 48, 64, 96, 128, 192, 256, 320};
  for (auto _ : state) {
    void* blocks[std::size(kSizes)];
    for (size_t i = 0; i < std::size(kSizes); ++i) {
      blocks[i] = std::malloc(kSizes[i]);
      do_not_optimize(blocks[i]);
    }
    for (size_t i = std::size(kSizes); i-- > 0;) {
      std::free(blocks[i]);
    }
  }
}

// Conversation length in messages.
BENCHMARK(BM_ChatTemplate_JinjaRender)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
// Number of tools, each with a parameter schema.
BENCHMARK(BM_ChatTemplate_JinjaRenderWithTools)
    ->RangeMultiplier(4)
    ->Range(1, 16)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_MinjaApply)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_MinjaBuiltins)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_MessagesToJson)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_MinjaRenderCachedBuiltins)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ChatTemplate_AllocatorControl)->Unit(benchmark::kNanosecond);

// DeepSeek ranges: exact input messages, consecutive tool results, or schemas;
// thinking is explicit to avoid V3.2/V4's different default modes. All measured
// calls include the full output and generation-mode classification.
BENCHMARK_TEMPLATE(BM_ChatTemplate_DeepseekConversation, DeepseekV32CppTemplate)
    ->ArgsProduct({{1, 4, 16, 64}, {0, 1}})
    ->ArgNames({"messages", "thinking"})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_ChatTemplate_DeepseekConversation, DeepseekV4CppTemplate)
    ->ArgsProduct({{1, 4, 16, 64}, {0, 1}})
    ->ArgNames({"messages", "thinking"})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_ChatTemplate_DeepseekToolResults, DeepseekV32CppTemplate)
    ->ArgsProduct({{1, 4, 16, 64}, {0, 1}, {0, 1}})
    ->ArgNames({"results", "thinking", "reversed"})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_ChatTemplate_DeepseekToolResults, DeepseekV4CppTemplate)
    ->ArgsProduct({{1, 4, 16, 64}, {0, 1}, {0, 1}})
    ->ArgNames({"results", "thinking", "reversed"})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_ChatTemplate_DeepseekToolSchemas, DeepseekV32CppTemplate)
    ->ArgsProduct({{1, 4, 16, 64}, {0, 1}})
    ->ArgNames({"schemas", "thinking"})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_ChatTemplate_DeepseekToolSchemas, DeepseekV4CppTemplate)
    ->ArgsProduct({{1, 4, 16, 64}, {0, 1}})
    ->ArgNames({"schemas", "thinking"})
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace xllm
