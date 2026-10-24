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

#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"

namespace xllm {

enum class ChatTemplateGenerationMode : uint8_t {
  CHAT = 0,
  REASONING = 1,
  UNKNOWN = 2,
};

struct ChatTemplateRenderResult final {
  std::string prompt;
  ChatTemplateGenerationMode generation_mode =
      ChatTemplateGenerationMode::UNKNOWN;
};

class ChatTemplate {
 public:
  virtual ~ChatTemplate() = default;

  virtual std::optional<std::string> apply(
      const ChatMessages& messages) const = 0;

  virtual std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<xllm::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const = 0;

  std::optional<ChatTemplateRenderResult> apply_with_generation_mode(
      const ChatMessages& messages,
      const std::vector<xllm::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const;

  static ChatTemplateGenerationMode generation_mode_from_prompt(
      std::string_view prompt);
};

}  // namespace xllm
