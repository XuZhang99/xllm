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

#include <brpc/controller.h>

#include <charconv>

#include "api_service/openai_json.h"
#include "core/common/constants.h"

namespace xllm::api_service {

inline std::pair<Status, std::string> openai_request_body(
    const brpc::Controller& controller,
    bool binary_input = true) {
  const auto* length =
      binary_input ? controller.http_request().GetHeader(kInferContentLength)
                   : nullptr;
  const size_t size = controller.request_attachment().size();
  size_t json_size = size;
  if (length != nullptr) {
    const auto [end, error] = std::from_chars(
        length->data(), length->data() + length->size(), json_size);
    if (error != std::errc() || end != length->data() + length->size() ||
        json_size > size) {
      return {Status(StatusCode::INVALID_ARGUMENT,
                     "Invalid inference JSON content length."),
              ""};
    }
  }
  std::string body;
  controller.request_attachment().copy_to(&body, json_size, 0);
  return {Status(), std::move(body)};
}

inline void write_openai_error(brpc::Controller* controller,
                               StatusCode code,
                               const std::string& message,
                               const std::string& param = "",
                               bool schema_error = false) {
  controller->http_response().set_status_code(openai_http_status(code));
  controller->http_response().set_content_type("application/json");
  controller->response_attachment().clear();
  controller->response_attachment().append(
      openai_error_json(code, message, param, schema_error));
}

}  // namespace xllm::api_service
