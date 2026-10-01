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

#include <nlohmann/json.hpp>
#include <string>

#include "chat.pb.h"
#include "completion.pb.h"
#include "core/common/types.h"
#include "embedding.pb.h"
#include "models.pb.h"

namespace xllm::api_service {

int32_t openai_http_status(StatusCode code);
std::string openai_error_json(StatusCode code,
                              const std::string& message,
                              const std::string& param = "",
                              bool schema_error = false);
nlohmann::json openai_response_json(const proto::ChatResponse& response,
                                    bool named_tool_choice = false,
                                    bool required_tool_choice = false);
nlohmann::json openai_response_json(const proto::CompletionResponse& response,
                                    bool stream = false);
nlohmann::json openai_embedding_json(const proto::EmbeddingResponse& response,
                                     const std::string& encoding_format);
nlohmann::json openai_usage_json(const proto::Usage& usage, bool stream = true);
nlohmann::json openai_models_json(const proto::ModelListResponse& response);

}  // namespace xllm::api_service
