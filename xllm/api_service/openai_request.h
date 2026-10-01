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

#include <string>
#include <utility>

#include "core/common/types.h"

namespace xllm::api_service {

enum class OpenAIEndpoint { CHAT, COMPLETION, EMBEDDING, MM_EMBEDDING };

// Normalize the public HTTP schema before decoding the internal protobuf.
// When provided, error_param is cleared and set for attributed validation
// errors. schema_error distinguishes request-schema errors from engine errors.
std::pair<Status, std::string> normalize_openai_request(
    std::string body,
    OpenAIEndpoint endpoint,
    const std::string& default_model,
    std::string* error_param = nullptr,
    bool* schema_error = nullptr);

}  // namespace xllm::api_service
