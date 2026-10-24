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

#include "framework/chat_template/deepseek_v32_cpp_template.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

namespace xllm {

TEST(DeepseekV32CppTemplate, PreservesLastUserThinkingAfterDroppingDeveloper) {
  const TokenizerArgs args;
  const DeepseekV32CppTemplate encoder(args);
  ChatMessages messages;
  messages.emplace_back("developer", "old instruction");
  messages.emplace_back("user", "hello");

  const nlohmann::ordered_json kwargs = {{"thinking", true},
                                         {"reasoning_effort", "low"}};
  const auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜><think>");
}

TEST(DeepseekV32CppTemplate, SnapshotsCustomBosToken) {
  TokenizerArgs args;
  args.bos_token("<custom-bos>");
  const DeepseekV32CppTemplate encoder(args);
  args.bos_token("<changed-bos>");
  ChatMessages messages;
  messages.emplace_back("user", "hello");

  const auto prompt = encoder.apply(
      messages, /*json_tools=*/{}, nlohmann::ordered_json{{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt, "<custom-bos><｜User｜>hello<｜Assistant｜></think>");
}

TEST(DeepseekV32CppTemplate, RendersHistoryWithoutUserMessages) {
  const TokenizerArgs args;
  const DeepseekV32CppTemplate encoder(args);
  ChatMessages messages;
  constexpr int32_t kMessageCount = 64;
  messages.reserve(kMessageCount);
  std::string expected = "<｜begin▁of▁sentence｜>";
  for (int32_t idx = 0; idx < kMessageCount; ++idx) {
    messages.emplace_back("system", "instruction");
    expected += "instruction";
  }

  const auto prompt = encoder.apply(
      messages, /*json_tools=*/{}, nlohmann::ordered_json{{"thinking", false}});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt, expected);
}

TEST(DeepseekV32CppTemplate, PreservesLongPayloadBytesAndLiteralPlaceholders) {
  const DeepseekV32CppTemplate encoder{TokenizerArgs()};
  const std::string literals =
      "中文🙂{content}{thinking_content}{reasoning}{tool_calls}{dsml_token}";
  const std::string system_content =
      std::string(64 * 1024, 's') + '\0' + literals;
  const std::string user_content =
      std::string(64 * 1024, 'u') + '\0' + literals;
  const std::string assistant_content =
      std::string(64 * 1024, 'a') + '\0' + literals;
  const std::string reasoning_content =
      std::string(128 * 1024, 'r') + '\0' + literals;
  ChatMessages messages;
  messages.reserve(3);
  messages.emplace_back("system", system_content);
  messages.emplace_back("user", user_content);
  Message assistant("assistant", assistant_content);
  assistant.reasoning_content = reasoning_content;
  messages.emplace_back(std::move(assistant));

  for (const bool thinking : {false, true}) {
    SCOPED_TRACE(thinking);
    const nlohmann::ordered_json kwargs = {{"thinking", thinking},
                                           {"reasoning_effort", "low"}};
    const auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
    ASSERT_TRUE(prompt.has_value());
    const std::string expected =
        std::string("<｜begin▁of▁sentence｜>") + system_content + "<｜User｜>" +
        user_content + "<｜Assistant｜>" +
        (thinking ? "<think>" + reasoning_content + "</think>" : "</think>") +
        assistant_content + "<｜end▁of▁sentence｜>";
    EXPECT_EQ(*prompt, expected);
  }
}

TEST(DeepseekV32CppTemplate, BasicUserMessage) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<｜begin▁of▁sentence｜>"), std::string::npos);
  EXPECT_NE(prompt->find("<｜User｜>hello<｜Assistant｜>"), std::string::npos);
}

TEST(DeepseekV32CppTemplate, DefaultThinkingModeIsChat) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("</think>"), std::string::npos);
  EXPECT_EQ(prompt->find("<think>"), std::string::npos);
}

TEST(DeepseekV32CppTemplate, ThinkingModeEnabledByKwargs) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "hello");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;

  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("<think>"), std::string::npos);
}

TEST(DeepseekV32CppTemplate, ReportsRenderedGenerationMode) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);
  ChatMessages messages;
  messages.emplace_back("user", "hello");

  auto chat = encoder.apply_with_generation_mode(
      messages, /*json_tools=*/{}, nlohmann::ordered_json::object());
  ASSERT_TRUE(chat.has_value());
  EXPECT_EQ(chat->generation_mode, ChatTemplateGenerationMode::CHAT);

  auto reasoning = encoder.apply_with_generation_mode(
      messages, /*json_tools=*/{}, nlohmann::ordered_json{{"thinking", true}});
  ASSERT_TRUE(reasoning.has_value());
  EXPECT_EQ(reasoning->generation_mode, ChatTemplateGenerationMode::REASONING);
}

TEST(DeepseekV32CppTemplate, ToolsInjectionFormat) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "weather in beijing");

  std::vector<JsonTool> tools;
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.description = "query weather";
  tool.function.parameters = nlohmann::json{
      {"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}};
  tools.push_back(tool);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, tools, kwargs);
  ASSERT_TRUE(prompt.has_value());

  // vLLM DSML format
  EXPECT_NE(prompt->find("## Tools"), std::string::npos);
  EXPECT_NE(prompt->find("get_weather"), std::string::npos);
  EXPECT_NE(prompt->find("<functions>"), std::string::npos);
  EXPECT_NE(prompt->find("</functions>"), std::string::npos);
  // User message after tools
  EXPECT_NE(prompt->find("<｜User｜>weather in beijing"
                         "<｜Assistant｜>"),
            std::string::npos);
}

TEST(DeepseekV32CppTemplate, ToolsInjectedAsNewSystemMessage) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("system", "You are helpful.");
  messages.emplace_back("user", "hi");

  std::vector<JsonTool> tools;
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "search";
  tool.function.description = "search the web";
  tool.function.parameters = nlohmann::json{{"type", "object"}};
  tools.push_back(tool);

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  auto prompt = encoder.apply(messages, tools, kwargs);
  ASSERT_TRUE(prompt.has_value());

  // Tools section before system content
  size_t tools_pos = prompt->find("## Tools");
  size_t content_pos = prompt->find("You are helpful.");
  ASSERT_NE(tools_pos, std::string::npos);
  ASSERT_NE(content_pos, std::string::npos);
  EXPECT_LT(tools_pos, content_pos);
}

TEST(DeepseekV32CppTemplate, DropThinkingOnlyWhenLastMessageIsUser) {
  TokenizerArgs args;
  args.bos_token("<｜begin▁of▁sentence｜>");
  DeepseekV32CppTemplate encoder(args);

  // user -> assistant(with reasoning) -> tool
  ChatMessages messages;
  messages.emplace_back("user", "weather?");

  Message assistant_msg("assistant", "calling tool");
  assistant_msg.reasoning_content = "thinking about it";
  Message::ToolCall tc;
  tc.id = "1";
  tc.type = "function";
  tc.function.name = "get_weather";
  tc.function.arguments = R"({"city":"beijing"})";
  assistant_msg.tool_calls = Message::ToolCallVec{tc};
  messages.push_back(assistant_msg);

  messages.emplace_back("tool", "sunny");

  nlohmann::ordered_json kwargs = nlohmann::json::object();
  kwargs["thinking"] = true;

  auto prompt = encoder.apply(messages, /*json_tools=*/{}, kwargs);
  ASSERT_TRUE(prompt.has_value());

  // Last message is "tool", NOT "user",
  // so thinking should NOT be dropped.
  EXPECT_NE(prompt->find("thinking about it"), std::string::npos);
}

namespace {

Message make_tool_assistant(int32_t call_count) {
  Message assistant("assistant", "");
  assistant.reasoning_content = "plan";
  Message::ToolCallVec calls;
  calls.reserve(call_count);
  for (int32_t idx = 0; idx < call_count; ++idx) {
    Message::ToolCall call;
    call.id = std::to_string(idx);
    call.type = "function";
    call.function.name = "lookup";
    call.function.arguments = R"({"query":"value"})";
    calls.emplace_back(std::move(call));
  }
  assistant.tool_calls = std::move(calls);
  return assistant;
}

TEST(DeepseekV32CppTemplate, PreservesEmptyAndBinaryToolResults) {
  const DeepseekV32CppTemplate encoder{TokenizerArgs()};
  const std::string tool_content =
      std::string("工具结果🙂") + '\0' +
      "{content}{thinking_content}{reasoning}{tool_calls}{dsml_token}";
  ChatMessages messages;
  messages.reserve(4);
  messages.emplace_back("user", "run tools");
  Message assistant = make_tool_assistant(/*call_count=*/2);
  assistant.reasoning_content = "";
  messages.emplace_back(std::move(assistant));
  messages.emplace_back("tool", "");
  messages.emplace_back("tool", tool_content);

  const auto prompt = encoder.apply(
      messages, /*json_tools=*/{}, nlohmann::ordered_json{{"thinking", true}});
  ASSERT_TRUE(prompt.has_value());
  const std::string expected =
      std::string(
          "<｜begin▁of▁sentence｜><｜User｜>run tools"
          "<｜Assistant｜><think></think>"
          "\n\n<｜DSML｜function_calls>"
          "\n<｜DSML｜invoke name=\"lookup\">"
          "\n<｜DSML｜parameter name=\"query\" string=\"true\">value"
          "</｜DSML｜parameter>\n</｜DSML｜invoke>"
          "\n<｜DSML｜invoke name=\"lookup\">"
          "\n<｜DSML｜parameter name=\"query\" string=\"true\">value"
          "</｜DSML｜parameter>\n</｜DSML｜invoke>"
          "\n</｜DSML｜function_calls><｜end▁of▁sentence｜>"
          "\n\n<function_results>\n<result></result>\n<result>") +
      tool_content + "</result>\n</function_results>\n\n<think>";
  EXPECT_EQ(*prompt, expected);
}

TEST(DeepseekV32CppTemplate, ConsecutiveToolResultsKeepRoundBoundaries) {
  const DeepseekV32CppTemplate encoder{TokenizerArgs()};
  ChatMessages messages;
  messages.emplace_back("user", "run tools");
  messages.emplace_back(make_tool_assistant(/*call_count=*/3));
  messages.emplace_back("tool", "first");
  messages.emplace_back("tool", "second");
  messages.emplace_back("tool", "third");
  messages.emplace_back(make_tool_assistant(/*call_count=*/1));
  messages.emplace_back("tool", "last");

  for (const bool thinking : {false, true}) {
    const auto prompt =
        encoder.apply(messages,
                      /*json_tools=*/{},
                      nlohmann::ordered_json{{"thinking", thinking}});
    ASSERT_TRUE(prompt.has_value());
    EXPECT_NE(prompt->find("<function_results>\n<result>first</result>"
                           "\n<result>second</result>\n<result>third</result>"
                           "\n</function_results>"),
              std::string::npos);
    const std::string ending = std::string(
                                   "<function_results>\n<result>last</"
                                   "result>\n</function_results>\n\n") +
                               (thinking ? "<think>" : "</think>");
    EXPECT_EQ(prompt->substr(prompt->size() - ending.size()), ending);
  }
}

TEST(DeepseekV32CppTemplate, PartialToolResultsLeaveResultBlockOpen) {
  const DeepseekV32CppTemplate encoder{TokenizerArgs()};
  ChatMessages messages;
  messages.emplace_back("user", "run tools");
  messages.emplace_back(make_tool_assistant(/*call_count=*/2));
  messages.emplace_back("tool", "first");
  const auto prompt = encoder.apply(messages);
  ASSERT_TRUE(prompt.has_value());
  const std::string ending = "<function_results>\n<result>first</result>";
  EXPECT_EQ(prompt->substr(prompt->size() - ending.size()), ending);
}

TEST(DeepseekV32CppTemplate, RejectsToolResultAfterInterveningUser) {
  const DeepseekV32CppTemplate encoder{TokenizerArgs()};
  ChatMessages messages;
  messages.emplace_back("user", "run tools");
  messages.emplace_back(make_tool_assistant(/*call_count=*/1));
  messages.emplace_back("tool", "first");
  messages.emplace_back("user", "new request");
  messages.emplace_back("tool", "orphan");
  EXPECT_DEATH(encoder.apply(messages),
               "Tool message does not follow assistant");
}

TEST(DeepseekV32CppTemplate, RejectsMoreResultsThanAssistantCalls) {
  const DeepseekV32CppTemplate encoder{TokenizerArgs()};
  ChatMessages messages;
  messages.emplace_back("user", "run tools");
  messages.emplace_back(make_tool_assistant(/*call_count=*/1));
  messages.emplace_back("tool", "first");
  messages.emplace_back("tool", "extra");
  EXPECT_DEATH(encoder.apply(messages), "Invalid tool call order");
}

}  // namespace

}  // namespace xllm
