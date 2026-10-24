/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/chat_template/jinja_chat_template.h"

#include <gtest/gtest.h>

#include <ctime>
#include <future>
#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace xllm {
namespace {

class TestableJinjaChatTemplate : public JinjaChatTemplate {
 public:
  TestableJinjaChatTemplate(const TokenizerArgs& args)
      : JinjaChatTemplate(args) {}

  using JinjaChatTemplate::apply;
  using JinjaChatTemplate::needs_polyfills;
};

// Qwen2.5's published template: supports system messages, tools, tool calls
// and tool responses natively, and renders tool-call arguments with `tojson`.
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

// What minja::chat_template::apply itself produces for these inputs; the
// native render must match it byte for byte.
std::string reference_render(const std::string& template_str,
                             const std::string& eos_token,
                             const nlohmann::ordered_json& messages,
                             const nlohmann::ordered_json& tools,
                             const nlohmann::ordered_json& kwargs) {
  minja::chat_template reference(template_str, "", eos_token);
  minja::chat_template_inputs inputs;
  inputs.messages = messages;
  inputs.tools = tools;
  inputs.add_generation_prompt = true;
  inputs.extra_context = kwargs;
  return reference.apply(inputs, minja::chat_template_options());
}

nlohmann::ordered_json make_weather_tools() {
  return nlohmann::ordered_json::array(
      {{{"type", "function"},
        {"function",
         {{"name", "get_weather"},
          {"description", "Get the weather for a city."},
          {"parameters",
           {{"type", "object"},
            {"properties", {{"city", {{"type", "string"}}}}},
            {"required", {"city"}}}}}}}});
}

void expect_both_paths_match_minja(const std::string& template_str,
                                   const ChatMessages& messages,
                                   const nlohmann::ordered_json& messages_json,
                                   const std::vector<JsonTool>& tools,
                                   const nlohmann::ordered_json& tools_json,
                                   const nlohmann::ordered_json& kwargs,
                                   bool polyfills) {
  TokenizerArgs args;
  args.chat_template(template_str);
  TestableJinjaChatTemplate chat_template(args);
  EXPECT_EQ(chat_template.needs_polyfills(messages_json, tools_json),
            polyfills);
  const std::string expected =
      reference_render(template_str, "", messages_json, tools_json, kwargs);
  const auto typed_result = chat_template.apply(messages, tools, kwargs);
  ASSERT_TRUE(typed_result.has_value());
  EXPECT_EQ(*typed_result, expected);
  const auto json_result =
      chat_template.apply(messages_json, tools_json, kwargs);
  ASSERT_TRUE(json_result.has_value());
  EXPECT_EQ(*json_result, expected);
}

TEST(JinjaChatTemplate, TypedMessagesMatchMinjaAcrossCapabilities) {
  const ChatMessages messages = {Message("system", "be brief"),
                                 Message("user", "hello")};
  const nlohmann::ordered_json messages_json = {
      {{"role", "system"}, {"content", "be brief"}},
      {{"role", "user"}, {"content", "hello"}}};
  const nlohmann::ordered_json kwargs = {{"mode", "fast"}};
  const std::vector<JsonTool> no_tools;
  const nlohmann::ordered_json no_tools_json = nlohmann::ordered_json::array();
  const std::string native_template =
      "{% for message in messages %}"
      "{{ message.role }}:{{ message.content }}|{% endfor %}{{ mode }}";
  expect_both_paths_match_minja(native_template,
                                messages,
                                messages_json,
                                no_tools,
                                no_tools_json,
                                kwargs,
                                /*polyfills=*/false);
  const std::string no_system_template =
      "{% for message in messages %}{% if message.role != 'system' %}"
      "{{ message.role }}:{{ message.content }}|{% endif %}{% endfor %}"
      "{{ mode }}";
  expect_both_paths_match_minja(no_system_template,
                                messages,
                                messages_json,
                                no_tools,
                                no_tools_json,
                                kwargs,
                                /*polyfills=*/true);

  JsonTool weather_tool;
  weather_tool.function.name = "get_weather";
  weather_tool.function.description = "Get the weather for a city.";
  weather_tool.function.parameters = {
      {"type", "object"},
      {"properties", {{"city", {{"type", "string"}}}}},
      {"required", {"city"}}};
  const std::vector<JsonTool> tools = {weather_tool};
  for (const std::string& template_str :
       {native_template, std::string(kQwen25ChatTemplate)}) {
    expect_both_paths_match_minja(
        template_str,
        messages,
        messages_json,
        tools,
        make_weather_tools(),
        kwargs,
        /*polyfills=*/template_str == native_template);
  }
}

TEST(JinjaChatTemplate, TypedToolCallsAndResponsesMatchMinja) {
  ChatMessages messages = {Message("user", "weather?"),
                           Message("assistant", ""),
                           Message("tool", "18C")};
  messages[1].tool_calls = Message::ToolCallVec{
      {"call_1", "function", {"get_weather", R"({"city":"Paris"})"}}};
  messages[1].reasoning_content = "look up weather";
  messages[2].tool_call_id = "call_1";
  const nlohmann::ordered_json messages_json = {
      {{"role", "user"}, {"content", "weather?"}},
      {{"role", "assistant"},
       {"content", ""},
       {"reasoning_content", "look up weather"},
       {"tool_calls",
        {{{"id", "call_1"},
          {"type", "function"},
          {"function",
           {{"name", "get_weather"}, {"arguments", R"({"city":"Paris"})"}}}}}}},
      {{"role", "tool"}, {"content", "18C"}, {"tool_call_id", "call_1"}}};
  const std::vector<JsonTool> tools;
  const nlohmann::ordered_json tools_json = nlohmann::ordered_json::array();
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  for (const std::string& template_str :
       {std::string(kQwen25ChatTemplate),
        std::string("{% for message in messages %}"
                    "{{ message.role }}:{{ message.content }}|{% endfor %}")}) {
    expect_both_paths_match_minja(template_str,
                                  messages,
                                  messages_json,
                                  tools,
                                  tools_json,
                                  kwargs,
                                  /*polyfills=*/true);
  }
}

TEST(JinjaChatTemplate, EmptyAndNullToolCallsKeepPolyfillSemantics) {
  const std::string template_str =
      "{% for message in messages %}{{ message.content }}|{% endfor %}";
  ChatMessages messages = {Message("assistant", "")};
  messages[0].tool_calls = Message::ToolCallVec();
  const nlohmann::ordered_json messages_json = {
      {{"role", "assistant"},
       {"content", ""},
       {"tool_calls", nlohmann::ordered_json::array()}}};
  const std::vector<JsonTool> tools;
  const nlohmann::ordered_json tools_json = nlohmann::ordered_json::array();
  const nlohmann::ordered_json kwargs;
  expect_both_paths_match_minja(template_str,
                                messages,
                                messages_json,
                                tools,
                                tools_json,
                                kwargs,
                                /*polyfills=*/true);

  TokenizerArgs args;
  args.chat_template(template_str);
  TestableJinjaChatTemplate chat_template(args);
  const nlohmann::ordered_json null_calls = {
      {{"role", "assistant"}, {"content", ""}, {"tool_calls", nullptr}}};
  EXPECT_FALSE(chat_template.needs_polyfills(null_calls, tools_json));
  const auto result = chat_template.apply(null_calls, tools_json, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result,
            reference_render(template_str, "", null_calls, tools_json, kwargs));
  EXPECT_EQ(*result, "|");
}

TEST(JinjaChatTemplate, TypedContentAndMultimodalArraysMatchMinja) {
  const std::string template_str =
      "{% for message in messages %}{% for part in message.content %}"
      "{% if part.type == 'text' %}{{ part.text }}"
      "{% else %}{{ part.image }}:{{ part.image_url }}{% endif %}"
      "{% endfor %}|{% endfor %}";
  const std::vector<JsonTool> tools;
  const nlohmann::ordered_json tools_json = nlohmann::ordered_json::array();
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  const ChatMessages string_messages = {Message("system", "be brief"),
                                        Message("user", "hello")};
  const nlohmann::ordered_json string_messages_json = {
      {{"role", "system"}, {"content", "be brief"}},
      {{"role", "user"}, {"content", "hello"}}};
  expect_both_paths_match_minja(template_str,
                                string_messages,
                                string_messages_json,
                                tools,
                                tools_json,
                                kwargs,
                                /*polyfills=*/true);

  const ChatMessages typed_messages = {
      Message("system", MMContentVec{MMContent("text", std::string("brief"))}),
      Message("user",
              MMContentVec{MMContent("text", std::string("hello")),
                           MMContent("image_url", ImageURL{"unused", {}})}),
      Message("assistant", MMContentVec{})};
  const nlohmann::ordered_json typed_messages_json = {
      {{"role", "system"},
       {"content", {{{"type", "text"}, {"text", "brief"}}}}},
      {{"role", "user"},
       {"content",
        {{{"type", "text"}, {"text", "hello"}},
         {{"type", "image_url"},
          {"image", "mm place holder"},
          {"image_url", "mm place holder"}}}}},
      {{"role", "assistant"}, {"content", nlohmann::ordered_json::array()}}};
  expect_both_paths_match_minja(template_str,
                                typed_messages,
                                typed_messages_json,
                                tools,
                                tools_json,
                                kwargs,
                                /*polyfills=*/false);
}

TEST(JinjaChatTemplate, RenderErrorsReturnNulloptForBothInputPaths) {
  TokenizerArgs args;
  args.chat_template(
      "{% for message in messages %}{{ message.content }}{% endfor %}"
      "{% if fail %}{{ raise_exception('request rejected') }}{% endif %}");
  TestableJinjaChatTemplate chat_template(args);
  const ChatMessages messages = {Message("user", "hello")};
  const nlohmann::ordered_json messages_json = {
      {{"role", "user"}, {"content", "hello"}}};
  const nlohmann::ordered_json kwargs = {{"fail", true}};
  const std::vector<JsonTool> tools;
  const nlohmann::ordered_json tools_json = nlohmann::ordered_json::array();
  EXPECT_FALSE(chat_template.apply(messages, tools, kwargs).has_value());
  EXPECT_FALSE(
      chat_template.apply(messages_json, tools_json, kwargs).has_value());
}

TEST(JinjaChatTemplate, NativeRenderMatchesMinja) {
  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "You are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "Hello! How can I help?"}},
      {{"role", "user"}, {"content", "What is the weather in Paris?"}}};
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();

  TokenizerArgs args;
  args.chat_template(kQwen25ChatTemplate);
  args.bos_token("");
  args.eos_token("<|im_end|>");
  TestableJinjaChatTemplate template_(args);

  // A plain conversation and one carrying tool definitions both render
  // natively on this template.
  for (const nlohmann::ordered_json& tools :
       {nlohmann::ordered_json::array(), make_weather_tools()}) {
    EXPECT_FALSE(template_.needs_polyfills(messages, tools));
    auto result = template_.apply(messages, tools, kwargs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(),
              reference_render(
                  kQwen25ChatTemplate, "<|im_end|>", messages, tools, kwargs));
  }
}

TEST(JinjaChatTemplate, PolyfilledInputsMatchMinja) {
  // Qwen2.5 renders tool-call arguments with `tojson`, so minja rewrites
  // string arguments into objects first; whichever path that takes, the
  // output must equal minja's own.
  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "You are a helpful assistant."}},
      {{"role", "user"}, {"content", "What is the weather in Paris?"}},
      {{"role", "assistant"},
       {"content", ""},
       {"tool_calls",
        {{{"id", "call_1"},
          {"type", "function"},
          {"function",
           {{"name", "get_weather"}, {"arguments", R"({"city":"Paris"})"}}}}}}},
      {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "18C"}}};
  const nlohmann::ordered_json tools = make_weather_tools();
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();

  TokenizerArgs args;
  args.chat_template(kQwen25ChatTemplate);
  args.bos_token("");
  args.eos_token("<|im_end|>");
  TestableJinjaChatTemplate template_(args);

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(),
            reference_render(
                kQwen25ChatTemplate, "<|im_end|>", messages, tools, kwargs));
}

TEST(JinjaChatTemplate, NativeRenderDefinesMinjaGlobals) {
  // bos_token / eos_token, strftime_now and chat_template_kwargs are globals
  // minja::chat_template::apply installs; the native render must too.
  const std::string template_str =
      "{{ bos_token }}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}{{ eos_token }}"
      "{% endfor %}"
      "{{ strftime_now('%Y') }}|{{ mode }}";
  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "be brief"}},
      {{"role", "user"}, {"content", "hello"}}};
  const nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  const nlohmann::ordered_json kwargs = {{"mode", "fast"}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("<s>");
  args.eos_token("</s>");
  TestableJinjaChatTemplate template_(args);
  ASSERT_FALSE(template_.needs_polyfills(messages, tools));

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());

  const std::time_t now = std::time(nullptr);
  char year[8];
  std::strftime(year, sizeof(year), "%Y", std::localtime(&now));
  EXPECT_EQ(
      result.value(),
      std::string("<s>system: be brief</s>user: hello</s>") + year + "|fast");
}

TEST(JinjaChatTemplate, ConcurrentRendersKeepRequestContextsIndependent) {
  TokenizerArgs args;
  args.chat_template(
      "{{ strftime_now('%Y') }}|{{ mode }}|"
      "{% for message in messages %}{{ message['content'] }}{% endfor %}"
      "|{{ strftime_now('%Y') }}");
  const TestableJinjaChatTemplate encoder(args);
  const nlohmann::ordered_json native_messages = {
      {{"role", "system"}, {"content", "shared"}},
      {{"role", "user"}, {"content", "request"}}};
  ASSERT_FALSE(encoder.needs_polyfills(native_messages,
                                       nlohmann::ordered_json::array()));

  constexpr int32_t kWorkerCount = 8;
  constexpr int32_t kRenderCount = 32;
  std::vector<std::future<bool>> renders;
  renders.reserve(kWorkerCount);
  for (int32_t worker = 0; worker < kWorkerCount; ++worker) {
    renders.emplace_back(std::async(std::launch::async, [&encoder, worker] {
      const std::string request = "request-" + std::to_string(worker);
      const nlohmann::ordered_json kwargs = {{"mode", request}};
      ChatMessages messages;
      messages.emplace_back("system", "shared");
      messages.emplace_back("user", request);
      for (int32_t iteration = 0; iteration < kRenderCount; ++iteration) {
        const auto result = encoder.apply(messages, kwargs);
        if (!result.has_value()) {
          return false;
        }
        const size_t separator = result->find('|');
        if (separator == std::string::npos) {
          return false;
        }
        const std::string year = result->substr(0, separator);
        if (year.empty() || *result != year + "|" + request + "|shared" +
                                           request + "|" + year) {
          return false;
        }
      }
      return true;
    }));
  }
  for (auto& render : renders) {
    EXPECT_TRUE(render.get());
  }
}

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected =
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("<|end_of_turn|>");
  TestableJinjaChatTemplate template_(args);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, AppliesChatTemplateKwargs) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% endif %}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}"
      "{% endfor %}"
      "{% if not enable_thinking %}<no_think>{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "user"}, {"content", "describe this image"}}};
  nlohmann::ordered_json chat_template_kwargs = {{"enable_thinking", false}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);
  const nlohmann::ordered_json tools = nlohmann::json::array();
  auto result = template_.apply(messages, tools, chat_template_kwargs);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), "user: describe this image<no_think>");
}

TEST(JinjaChatTemplate, ReportsRenderedGenerationMode) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% else %}</think>{% endif %}";
  ChatMessages messages;
  messages.emplace_back("user", "hello");
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);

  const std::vector<JsonTool> tools;
  auto reasoning = template_.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json{{"enable_thinking", true}});
  ASSERT_TRUE(reasoning.has_value());
  EXPECT_EQ(reasoning->generation_mode, ChatTemplateGenerationMode::REASONING);

  auto chat = template_.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json{{"enable_thinking", false}});
  ASSERT_TRUE(chat.has_value());
  EXPECT_EQ(chat->generation_mode, ChatTemplateGenerationMode::CHAT);

  args.chat_template("assistant:");
  TestableJinjaChatTemplate unknown_template(args);
  auto unknown = unknown_template.apply_with_generation_mode(
      messages, tools, nlohmann::ordered_json::object());
  ASSERT_TRUE(unknown.has_value());
  EXPECT_EQ(unknown->generation_mode, ChatTemplateGenerationMode::UNKNOWN);
}

TEST(ChatTemplate, ClassifiesTerminalGenerationMarkers) {
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix <think>\n"),
            ChatTemplateGenerationMode::REASONING);
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix </think>\t"),
            ChatTemplateGenerationMode::CHAT);
  EXPECT_EQ(ChatTemplate::generation_mode_from_prompt("prefix"),
            ChatTemplateGenerationMode::UNKNOWN);
}

TEST(JinjaChatTemplate, SupportsUndefinedTests) {
  // Qwen3.8 uses both forms for optional chat-template arguments. Quoted text
  // must remain unchanged while executable Jinja expressions are normalized.
  const std::string template_str =
      "plain text: value is undefined|"
      "{{ 'quoted value is undefined' }}|"
      "{% if enable_thinking is undefined %}default"
      "{% else %}configured{% endif %}|"
      "{% if enable_thinking is not undefined %}present"
      "{% else %}missing{% endif %}";

  nlohmann::ordered_json messages = {{{"role", "user"}, {"content", "hello"}}};
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);

  auto default_result = template_.apply(messages);
  ASSERT_TRUE(default_result.has_value());
  EXPECT_EQ(default_result.value(),
            "plain text: value is undefined|quoted value is undefined|default|"
            "missing");

  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {{"enable_thinking", false}};
  auto configured_result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(configured_result.has_value());
  EXPECT_EQ(configured_result.value(),
            "plain text: value is undefined|quoted value is undefined|"
            "configured|present");
}

TEST(JinjaChatTemplate, PreservesGlmToolResponses) {
  const std::string template_str = R"JINJA(
{%- macro is_list_of_outputs(m) -%}
  {%- if m.content and m.content.0.output is defined -%}1{%- endif -%}
{%- endmacro -%}
{%- for m in messages -%}
  {%- if m.role == 'tool' -%}
    {{- '<|observation|><tool_response>' -}}
    {%- if is_list_of_outputs(m) -%}
      {%- for item in m.content -%}{{- item.output -}}{%- endfor -%}
    {%- else -%}{{- m.content -}}{%- endif -%}
    {{- '</tool_response>' -}}
  {%- elif m.tool_calls -%}
    {%- for call in m.tool_calls -%}
      {{- call.function.name -}}{{- call.function.arguments | tojson -}}
    {%- endfor -%}
  {%- else -%}{{- m.content -}}{%- endif -%}
{%- endfor -%}
)JINJA";
  TokenizerArgs args;
  args.chat_template(template_str);
  TestableJinjaChatTemplate template_(args);
  const nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  const nlohmann::ordered_json string_messages = {
      {{"role", "tool"}, {"content", "Paris is sunny."}}};
  EXPECT_FALSE(template_.needs_polyfills(string_messages, tools));
  auto string_result = template_.apply(string_messages);
  ASSERT_TRUE(string_result.has_value());
  EXPECT_EQ(*string_result,
            "<|observation|><tool_response>Paris is sunny.</tool_response>");

  const nlohmann::ordered_json list_messages = {
      {{"role", "tool"},
       {"content", {{{"output", "Paris "}}, {{"output", "is sunny."}}}}}};
  auto list_result = template_.apply(list_messages);
  ASSERT_TRUE(list_result.has_value());
  EXPECT_EQ(*list_result, *string_result);

  args.chat_template(
      "m.content.0.output is defined|"
      "{{ 'm.content.0.output is defined' }}");
  TestableJinjaChatTemplate literal_template(args);
  auto literal_result = literal_template.apply(string_messages);
  ASSERT_TRUE(literal_result.has_value());
  EXPECT_EQ(*literal_result,
            "m.content.0.output is defined|m.content.0.output is defined");
}

TEST(JinjaChatTemplate, PreservesDeferredToolMetadata) {
  TokenizerArgs args;
  args.chat_template(kQwen25ChatTemplate);
  TestableJinjaChatTemplate chat_template(args);
  const ChatMessages messages = {Message("user", "hello")};
  JsonTool tool;
  tool.function.name = "search";
  tool.function.parameters = {{"type", "object"}};
  tool.function.defer_loading = true;
  const std::vector<JsonTool> tools = {tool};
  const auto prompt =
      chat_template.apply(messages, tools, nlohmann::ordered_json::object());
  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("\"defer_loading\": true"), std::string::npos);
}

}  // namespace
}  // namespace xllm
