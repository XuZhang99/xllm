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

#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>

#include <cstdint>
#include <exception>
#include <sstream>
#include <utility>
#include <variant>

namespace xllm {
namespace {

constexpr const char* kBosToken = "<｜begin▁of▁sentence｜>";
constexpr const char* kThinkingStartToken = "<think>";
constexpr const char* kThinkingEndToken = "</think>";
constexpr const char* kDsmlToken = "｜DSML｜";
constexpr const char* kToolsSystemTemplate =
    "## Tools\n"
    "You have access to a set of tools you can "
    "use to answer the user's question.\n"
    "You can invoke functions by writing a "
    "\"<{dsml_token}function_calls>\" block like "
    "the following as part of your reply to the "
    "user:\n"
    "<{dsml_token}function_calls>\n"
    "<{dsml_token}invoke name=\"$FUNCTION_NAME\">\n"
    "<{dsml_token}parameter "
    "name=\"$PARAMETER_NAME\" "
    "string=\"true|false\">"
    "$PARAMETER_VALUE"
    "</{dsml_token}parameter>\n"
    "...\n"
    "</{dsml_token}invoke>\n"
    "<{dsml_token}invoke "
    "name=\"$FUNCTION_NAME2\">\n"
    "...\n"
    "</{dsml_token}invoke>\n"
    "</{dsml_token}function_calls>\n"
    "String and scalar parameters should be "
    "specified as is without any escaping or "
    "quotes, while lists and objects should use "
    "JSON format. The \"string\" attribute should "
    "be set to \"true\" for string type parameters "
    "and \"false\" for other types (numbers, "
    "booleans, arrays, objects).\n"
    "If the thinking_mode is enabled, then after "
    "function results you should strongly consider "
    "outputting a thinking block. "
    "Here is an example:\n"
    "<{dsml_token}function_calls>\n"
    "...\n"
    "</{dsml_token}function_calls>\n"
    "<function_results>\n"
    "...\n"
    "</function_results>\n"
    "{thinking_start_token}...thinking about "
    "results{thinking_end_token}\n"
    "Here are the functions available in "
    "JSONSchema format:\n"
    "<functions>\n"
    "{tool_schemas}\n"
    "</functions>\n";

constexpr const char* kSystemMessageTemplate = "{content}";
constexpr const char* kUserMessageTemplate =
    "<｜User｜>{content}<｜Assistant｜>";
constexpr const char* kAssistantMessageTemplate =
    "{reasoning}{content}{tool_calls}<｜end▁of▁sentence｜>";
constexpr const char* kThinkingTemplate = "{reasoning}";
constexpr const char* kResponseFormatTemplate =
    "## Response Format:\n\n"
    "You MUST strictly adhere to the following schema to reply:\n"
    "{schema}";
constexpr const char* kToolCallTemplate =
    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>";
constexpr const char* kToolCallsTemplate =
    "<{dsml_token}function_calls>\n{tool_calls}\n</{dsml_token}function_calls>";
constexpr const char* kToolOutputTemplate = "\n<result>{content}</result>";

constexpr const char* kRoleSystem = "system";
constexpr const char* kRoleDeveloper = "developer";
constexpr const char* kRoleUser = "user";
constexpr const char* kRoleTool = "tool";
constexpr const char* kRoleAssistant = "assistant";

constexpr const char* kThinkingModeThinking = "thinking";
constexpr const char* kThinkingModeChat = "chat";

std::string to_json(const nlohmann::ordered_json& value) {
  try {
    return value.dump(/*indent=*/-1,
                      /*indent_char=*/' ',
                      /*ensure_ascii=*/false,
                      nlohmann::json::error_handler_t::replace);
  } catch (const std::exception&) {
    return value.dump(/*indent=*/-1,
                      /*indent_char=*/' ',
                      /*ensure_ascii=*/true,
                      nlohmann::json::error_handler_t::replace);
  }
}

bool get_thinking_enabled(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("thinking") && kwargs["thinking"].is_boolean()) {
    return kwargs["thinking"].get<bool>();
  }
  if (kwargs.contains("enable_thinking") &&
      kwargs["enable_thinking"].is_boolean()) {
    return kwargs["enable_thinking"].get<bool>();
  }
  return false;
}

std::string get_thinking_mode(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("thinking_mode") && kwargs["thinking_mode"].is_string()) {
    return kwargs["thinking_mode"].get<std::string>();
  }
  return get_thinking_enabled(kwargs) ? kThinkingModeThinking
                                      : kThinkingModeChat;
}

std::vector<nlohmann::ordered_json> tool_calls_from_openai_format(
    const Message::ToolCallVec& tool_calls) {
  std::vector<nlohmann::ordered_json> out;
  out.reserve(tool_calls.size());
  for (const Message::ToolCall& tool_call : tool_calls) {
    nlohmann::ordered_json args_json = nlohmann::json::object();
    if (!tool_call.function.arguments.empty()) {
      try {
        args_json = nlohmann::json::parse(tool_call.function.arguments);
      } catch (const std::exception&) {
        args_json = nlohmann::json::object();
      }
    }
    nlohmann::ordered_json item;
    item["name"] = tool_call.function.name;
    item["arguments"] = std::move(args_json);
    out.emplace_back(std::move(item));
  }
  return out;
}

std::string encode_arguments_to_dsml(const nlohmann::ordered_json& tool_call) {
  std::ostringstream oss;
  nlohmann::ordered_json arguments = nlohmann::json::object();
  if (tool_call.contains("arguments")) {
    arguments = tool_call["arguments"];
    if (arguments.is_string()) {
      try {
        arguments = nlohmann::json::parse(arguments.get<std::string>());
      } catch (const std::exception&) {
        arguments = nlohmann::json::object();
      }
    }
  }
  bool first = true;
  for (auto it = arguments.begin(); it != arguments.end(); ++it) {
    if (!first) {
      oss << "\n";
    }
    first = false;
    const nlohmann::ordered_json& value = it.value();
    bool is_string = value.is_string();
    oss << "<" << kDsmlToken << "parameter name=\"" << it.key()
        << "\" string=\"" << (is_string ? "true" : "false") << "\">";
    if (is_string) {
      oss << value.get<std::string>();
    } else {
      oss << to_json(value);
    }
    oss << "</" << kDsmlToken << "parameter>";
  }
  return oss.str();
}

std::string render_tools(const nlohmann::ordered_json& tools) {
  std::vector<std::string> schemas;
  schemas.reserve(tools.size());
  for (const nlohmann::ordered_json& tool : tools) {
    schemas.emplace_back(to_json(tool));
  }
  return absl::StrReplaceAll(std::string(kToolsSystemTemplate),
                             {{"{tool_schemas}", absl::StrJoin(schemas, "\n")},
                              {"{dsml_token}", kDsmlToken},
                              {"{thinking_start_token}", kThinkingStartToken},
                              {"{thinking_end_token}", kThinkingEndToken}});
}

int32_t find_last_user_index(const nlohmann::ordered_json& messages) {
  for (int32_t idx = static_cast<int32_t>(messages.size()) - 1; idx >= 0;
       --idx) {
    const nlohmann::ordered_json& message = messages[idx];
    std::string role = message.value("role", "");
    if (role == kRoleUser || role == kRoleDeveloper) {
      return idx;
    }
  }
  return -1;
}

std::string render_system_message(const nlohmann::ordered_json& message) {
  return absl::StrReplaceAll(std::string(kSystemMessageTemplate),
                             {{"{content}", message.value("content", "")}});
}

std::string render_user_message(const std::string& content) {
  return absl::StrReplaceAll(std::string(kUserMessageTemplate),
                             {{"{content}", content}});
}

std::string render_assistant_block(const std::string& reasoning,
                                   const std::string& content,
                                   const std::string& tool_calls) {
  return absl::StrReplaceAll(std::string(kAssistantMessageTemplate),
                             {{"{reasoning}", reasoning},
                              {"{content}", content},
                              {"{tool_calls}", tool_calls}});
}

std::string render_assistant_tool_calls(
    const std::vector<nlohmann::ordered_json>& tool_calls) {
  if (tool_calls.empty()) {
    return "";
  }
  std::vector<std::string> chunks;
  chunks.reserve(tool_calls.size());
  for (const nlohmann::ordered_json& tool_call : tool_calls) {
    std::string invoke = absl::StrReplaceAll(
        std::string(kToolCallTemplate),
        {{"{dsml_token}", kDsmlToken},
         {"{name}", tool_call.value("name", "")},
         {"{arguments}", encode_arguments_to_dsml(tool_call)}});
    chunks.emplace_back(std::move(invoke));
  }
  std::string rendered =
      absl::StrReplaceAll(std::string(kToolCallsTemplate),
                          {{"{dsml_token}", kDsmlToken},
                           {"{tool_calls}", absl::StrJoin(chunks, "\n")}});
  return "\n\n" + rendered;
}

std::string render_response_format(
    const nlohmann::ordered_json& response_format) {
  return absl::StrReplaceAll(std::string(kResponseFormatTemplate),
                             {{"{schema}", to_json(response_format)}});
}

std::string get_text_content(const Message::Content& content) {
  if (std::holds_alternative<std::string>(content)) {
    return std::get<std::string>(content);
  }
  return "";
}

// Always insert a new system message at index 0 with tools,
// matching the Python behavior where existing system messages are kept
// separately (tools section comes before system content).
nlohmann::ordered_json normalize_messages(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& tools) {
  nlohmann::ordered_json out = nlohmann::json::array();
  for (const Message& message : messages) {
    nlohmann::ordered_json item;
    item["role"] = message.role;
    item["content"] = get_text_content(message.content);
    if (message.reasoning_content.has_value()) {
      item["reasoning"] = message.reasoning_content.value();
    }
    if (message.tool_calls.has_value()) {
      item["tool_calls"] =
          tool_calls_from_openai_format(message.tool_calls.value());
    }
    out.emplace_back(std::move(item));
  }
  if (!tools.empty()) {
    nlohmann::ordered_json tools_json = nlohmann::json::array();
    for (const xllm::JsonTool& tool : tools) {
      nlohmann::ordered_json openai_tool;
      openai_tool["type"] = "function";
      nlohmann::ordered_json function;
      function["name"] = tool.function.name;
      function["description"] = tool.function.description;
      function["parameters"] = tool.function.parameters;
      if (tool.function.defer_loading.has_value()) {
        function["defer_loading"] = *tool.function.defer_loading;
      }
      openai_tool["function"] = std::move(function);
      tools_json.emplace_back(std::move(openai_tool));
    }

    nlohmann::ordered_json system_message;
    system_message["role"] = kRoleSystem;
    system_message["tools"] = std::move(tools_json);
    out.insert(out.begin(), std::move(system_message));
  }
  return out;
}

nlohmann::ordered_json drop_thinking_messages(
    const nlohmann::ordered_json& messages,
    int32_t last_user_index) {
  nlohmann::ordered_json output = nlohmann::json::array();
  for (int32_t idx = 0; idx < static_cast<int32_t>(messages.size()); ++idx) {
    const nlohmann::ordered_json& message = messages[idx];
    std::string role = message.value("role", "");
    if (role == kRoleUser || role == kRoleSystem || role == kRoleTool ||
        idx >= last_user_index) {
      output.emplace_back(message);
      continue;
    }
    if (role == kRoleAssistant) {
      nlohmann::ordered_json copied = message;
      copied.erase("reasoning");
      output.emplace_back(std::move(copied));
    }
  }
  return output;
}

std::string render_message(const nlohmann::ordered_json& messages,
                           int32_t index,
                           int32_t last_user_idx,
                           const std::string& thinking_mode) {
  if (index < 0 || index >= static_cast<int32_t>(messages.size())) {
    LOG(FATAL) << "Message index out of range";
  }
  if (thinking_mode != kThinkingModeThinking &&
      thinking_mode != kThinkingModeChat) {
    LOG(FATAL) << "Invalid thinking mode";
  }

  std::string prompt;
  const nlohmann::ordered_json& msg = messages[index];
  std::string role = msg.value("role", "");
  std::string content = msg.value("content", "");

  nlohmann::ordered_json tools = nlohmann::json::array();
  if (msg.contains("tools")) {
    tools = msg["tools"];
  }
  nlohmann::ordered_json response_format;
  if (msg.contains("response_format")) {
    response_format = msg["response_format"];
  }

  if (role == kRoleSystem) {
    prompt += render_system_message(msg);
    if (tools.is_array() && !tools.empty()) {
      nlohmann::ordered_json fn_tools = nlohmann::json::array();
      for (const auto& t : tools) {
        if (t.contains("function")) {
          fn_tools.emplace_back(t["function"]);
        }
      }
      prompt += "\n\n" + render_tools(fn_tools);
    }
    if (!response_format.is_null()) {
      prompt += "\n\n" + render_response_format(response_format);
    }
    return prompt;
  }

  if (role == kRoleDeveloper) {
    if (content.empty()) {
      LOG(FATAL) << "Developer message content is empty";
    }
    std::string developer_content;
    if (tools.is_array() && !tools.empty()) {
      nlohmann::ordered_json fn_tools = nlohmann::json::array();
      for (const auto& t : tools) {
        if (t.contains("function")) {
          fn_tools.emplace_back(t["function"]);
        }
      }
      developer_content += "\n\n" + render_tools(fn_tools);
    }
    if (!response_format.is_null()) {
      developer_content += "\n\n" + render_response_format(response_format);
    }
    developer_content += "\n\n# The user's message is: " + content;
    prompt += render_user_message(developer_content);
    if (index == last_user_idx && thinking_mode == kThinkingModeThinking) {
      prompt += kThinkingStartToken;
    } else {
      prompt += kThinkingEndToken;
    }
    return prompt;
  }

  if (role == kRoleUser) {
    prompt += render_user_message(content);
    if (index == last_user_idx && thinking_mode == kThinkingModeThinking) {
      prompt += kThinkingStartToken;
    } else {
      prompt += kThinkingEndToken;
    }
    return prompt;
  }

  if (role == kRoleTool) {
    int32_t prev_assistant_idx = index - 1;
    while (prev_assistant_idx >= 0 &&
           messages[prev_assistant_idx].value("role", "") == kRoleTool) {
      --prev_assistant_idx;
    }
    if (prev_assistant_idx < 0 ||
        messages[prev_assistant_idx].value("role", "") != kRoleAssistant) {
      LOG(FATAL) << "Tool message does not follow assistant";
    }
    const nlohmann::ordered_json& assistant_message =
        messages[prev_assistant_idx];
    if (!assistant_message.contains("tool_calls") ||
        !assistant_message["tool_calls"].is_array()) {
      LOG(FATAL) << "Missing assistant tool calls for tool output";
    }
    int32_t tool_call_order = index - prev_assistant_idx;
    int32_t total_tool_calls =
        static_cast<int32_t>(assistant_message["tool_calls"].size());
    if (tool_call_order <= 0 || tool_call_order > total_tool_calls) {
      LOG(FATAL) << "Invalid tool call order";
    }
    if (tool_call_order == 1) {
      prompt += "\n\n<function_results>";
    }
    std::string tool_result = absl::StrReplaceAll(
        std::string(kToolOutputTemplate), {{"{content}", content}});
    prompt += tool_result;
    if (tool_call_order == total_tool_calls) {
      prompt += "\n</function_results>";
      if (index >= last_user_idx && thinking_mode == kThinkingModeThinking) {
        prompt += "\n\n";
        prompt += kThinkingStartToken;
      } else {
        prompt += "\n\n";
        prompt += kThinkingEndToken;
      }
    }
    return prompt;
  }

  if (role == kRoleAssistant) {
    std::string thinking_part;
    std::string tool_calls_content;
    std::vector<nlohmann::ordered_json> tool_calls;
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
      tool_calls.reserve(msg["tool_calls"].size());
      for (const nlohmann::ordered_json& tool_call : msg["tool_calls"]) {
        tool_calls.emplace_back(tool_call);
      }
    }
    if (!tool_calls.empty()) {
      tool_calls_content = render_assistant_tool_calls(tool_calls);
    }
    if (thinking_mode == kThinkingModeThinking && index > last_user_idx) {
      std::string reasoning = msg.value("reasoning", "");
      if (reasoning.empty() && tool_calls.empty()) {
        LOG(FATAL) << "Thinking mode assistant message must contain "
                      "reasoning/tool_calls";
      }
      std::string thinking_tmpl = absl::StrReplaceAll(
          std::string(kThinkingTemplate), {{"{reasoning}", reasoning}});
      thinking_part = thinking_tmpl + kThinkingEndToken;
    }
    bool is_prefix = msg.contains("prefix") && msg["prefix"].is_boolean() &&
                     msg["prefix"].get<bool>();
    if (tool_calls.empty() && is_prefix) {
      prompt += content;
    } else {
      prompt +=
          render_assistant_block(thinking_part, content, tool_calls_content);
    }
    return prompt;
  }

  LOG(FATAL) << "Unknown role: " + role;
}

}  // namespace

DeepseekV32CppTemplate::DeepseekV32CppTemplate(const TokenizerArgs& args)
    : bos_token_(args.bos_token().empty() ? kBosToken : args.bos_token()) {}

std::optional<std::string> DeepseekV32CppTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, kwargs);
}

std::optional<std::string> DeepseekV32CppTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  try {
    nlohmann::ordered_json normalized =
        normalize_messages(messages, json_tools);
    std::string thinking_mode = get_thinking_mode(chat_template_kwargs);

    // Only drop thinking when the last message is
    // from user, matching vLLM:
    // drop_thinking = messages[-1]["role"] == "user"
    bool drop_thinking = false;
    if (!normalized.empty()) {
      std::string last_role = normalized.back().value("role", "");
      drop_thinking = (last_role == kRoleUser);
    }
    if (thinking_mode == kThinkingModeThinking && drop_thinking) {
      normalized =
          drop_thinking_messages(normalized, find_last_user_index(normalized));
    }

    const int32_t last_user_idx = find_last_user_index(normalized);
    std::string prompt = bos_token_;
    for (int32_t idx = 0; idx < static_cast<int32_t>(normalized.size());
         ++idx) {
      prompt += render_message(normalized, idx, last_user_idx, thinking_mode);
    }
    return prompt;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to apply DeepSeek V3.2 native template: " << e.what();
    return std::nullopt;
  }
}

}  // namespace xllm
