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

#include <utility>

#include "framework/chat_template/chat_template.h"

namespace xllm {

std::optional<ChatTemplateRenderResult>
ChatTemplate::apply_with_generation_mode(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  std::optional<std::string> prompt =
      apply(messages, json_tools, chat_template_kwargs);
  if (!prompt.has_value()) {
    return std::nullopt;
  }
  const ChatTemplateGenerationMode generation_mode =
      generation_mode_from_prompt(prompt.value());
  return ChatTemplateRenderResult{std::move(prompt.value()), generation_mode};
}

ChatTemplateGenerationMode ChatTemplate::generation_mode_from_prompt(
    std::string_view prompt) {
  constexpr std::string_view kJsonWhitespace = " \t\n\r";
  while (!prompt.empty() &&
         kJsonWhitespace.find(prompt.back()) != std::string_view::npos) {
    prompt.remove_suffix(1);
  }
  constexpr std::string_view kThinkingMarker = "<think>";
  constexpr std::string_view kChatMarker = "</think>";
  if (prompt.ends_with(kThinkingMarker)) {
    return ChatTemplateGenerationMode::REASONING;
  }
  if (prompt.ends_with(kChatMarker)) {
    return ChatTemplateGenerationMode::CHAT;
  }
  return ChatTemplateGenerationMode::UNKNOWN;
}

}  // namespace xllm
