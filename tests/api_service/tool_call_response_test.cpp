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

#include "api_service/tool_call_response.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "api_service/stream_output_parser.h"

namespace xllm {
namespace {

std::vector<JsonTool> make_test_tools() {
  return {JsonTool("function",
                   JsonFunction("get_weather",
                                "Get the weather",
                                nlohmann::json{{"type", "object"}}))};
}

TEST(ToolCallResultTest, KeepsNormalTextWhenNoToolCallMarkerExists) {
  const std::string text = "The weather is sunny.";

  const auto result =
      api_service::process_tool_calls(text, make_test_tools(), "glm45", "stop");

  EXPECT_FALSE(result.tool_calls.has_value());
  EXPECT_EQ(result.text, text);
  EXPECT_EQ(result.finish_reason, "stop");
}

TEST(ToolCallResultTest, KeepsNormalTextWhenMarkerDoesNotParse) {
  const std::string text =
      "The weather is sunny. <tool_call>unknown_tool\n"
      "<arg_key>location</arg_key>\n"
      "<arg_value>Beijing</arg_value>\n"
      "</tool_call>";

  const auto result =
      api_service::process_tool_calls(text, make_test_tools(), "glm45", "stop");

  EXPECT_FALSE(result.tool_calls.has_value());
  EXPECT_EQ(result.text, text);
  EXPECT_EQ(result.finish_reason, "stop");
}

TEST(ToolCallResultTest, PreservesParsedToolCall) {
  const auto result = api_service::process_tool_calls(
      "I will check. <tool_call>get_weather\n"
      "<arg_key>location</arg_key>\n"
      "<arg_value>Beijing</arg_value>\n"
      "</tool_call>",
      make_test_tools(),
      "glm45",
      "stop");

  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  EXPECT_EQ(result.tool_calls->at(0).function().name(), "get_weather");
  EXPECT_EQ(result.finish_reason, "tool_calls");
}

TEST(ToolCallResultTest, KeepsToolCallsAfterSourceArenaIsDestroyed) {
  api_service::ToolCallResult result;
  {
    google::protobuf::Arena arena;
    result = api_service::process_tool_calls(
        "<tool_call>get_weather\n"
        "<arg_key>location</arg_key>\n"
        "<arg_value>Beijing</arg_value>\n"
        "</tool_call>",
        make_test_tools(),
        "glm45",
        "length",
        &arena);
  }

  ASSERT_TRUE(result.tool_calls.has_value());
  ASSERT_EQ(result.tool_calls->size(), 1);
  const auto& call = result.tool_calls->at(0);
  EXPECT_EQ(call.function().name(), "get_weather");
  EXPECT_EQ(nlohmann::json::parse(call.function().arguments()),
            (nlohmann::json{{"location", "Beijing"}}));
  EXPECT_EQ(result.finish_reason, "length");
}

TEST(ToolCallStreamResponseTest, SendsBufferedArgumentsBeforeFinishing) {
  auto stream_parser =
      std::make_shared<StreamOutputParser>(make_test_tools(), "qwen25", "");
  auto* parser = stream_parser->get_tool_call_parser(0);
  const auto parsed = parser->parse_streaming_increment(
      "<tool_call>\n"
      R"({"name":"get_weather","arguments":{"location":"Beijing"}})");
  ASSERT_EQ(parsed.calls.size(), 1);
  EXPECT_EQ(parsed.calls[0].name, "get_weather");
  EXPECT_TRUE(parsed.calls[0].parameters.empty());

  std::string arguments;
  int32_t sent_tool_index = -1;
  ASSERT_TRUE(api_service::check_for_unstreamed_tool_args(
      stream_parser,
      /*index=*/0,
      [&](const std::string& remaining, int32_t tool_index) {
        arguments += remaining;
        sent_tool_index = tool_index;
        return true;
      }));

  EXPECT_EQ(sent_tool_index, 0);
  EXPECT_EQ(nlohmann::json::parse(arguments),
            (nlohmann::json{{"location", "Beijing"}}));
}

TEST(ToolCallStreamResponseTest, PropagatesArgumentSendFailure) {
  auto stream_parser =
      std::make_shared<StreamOutputParser>(make_test_tools(), "qwen25", "");
  stream_parser->get_tool_call_parser(0)->parse_streaming_increment(
      "<tool_call>\n"
      R"({"name":"get_weather","arguments":{"location":"Beijing"}})");

  bool send_attempted = false;
  EXPECT_FALSE(api_service::check_for_unstreamed_tool_args(
      stream_parser,
      /*index=*/0,
      [&](const std::string& /*remaining*/, int32_t /*tool_index*/) {
        send_attempted = true;
        return false;
      }));
  EXPECT_TRUE(send_attempted);
}

TEST(ToolCallStreamResponseTest, DoesNotResendStreamedArguments) {
  auto stream_parser =
      std::make_shared<StreamOutputParser>(make_test_tools(), "glm45", "");
  const auto parsed =
      stream_parser->get_tool_call_parser(0)->parse_streaming_increment(
          "<tool_call>get_weather\n"
          "<arg_key>location</arg_key>\n"
          "<arg_value>Beijing</arg_value>\n"
          "</tool_call>");
  ASSERT_EQ(parsed.calls.size(), 1);
  EXPECT_FALSE(parsed.calls[0].parameters.empty());

  EXPECT_TRUE(api_service::check_for_unstreamed_tool_args(
      stream_parser,
      /*index=*/0,
      [](const std::string& /*remaining*/, int32_t /*tool_index*/) {
        ADD_FAILURE() << "Arguments were already streamed";
        return false;
      }));
}

}  // namespace
}  // namespace xllm
