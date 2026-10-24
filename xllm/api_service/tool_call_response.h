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

#pragma once

#include <google/protobuf/arena.h>
#include <google/protobuf/repeated_ptr_field.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "chat.pb.h"
#include "core/common/types.h"

namespace xllm {

class StreamOutputParser;

namespace api_service {

using SendFunc = std::function<bool(const std::string&, int32_t)>;

// Send any remaining tool arguments before the final stream response.
// Shared by the Chat and Anthropic response adapters.
bool check_for_unstreamed_tool_args(
    std::shared_ptr<StreamOutputParser> stream_parser,
    size_t index,
    SendFunc send_func);

struct ToolCallResult {
  std::optional<google::protobuf::RepeatedPtrField<proto::ToolCall>> tool_calls;
  std::string text;
  std::string finish_reason;
};

ToolCallResult process_tool_calls(std::string text,
                                  const std::vector<xllm::JsonTool>& tools,
                                  const std::string& parser_format,
                                  std::string finish_reason,
                                  google::protobuf::Arena* arena = nullptr);

}  // namespace api_service
}  // namespace xllm
