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

#include "call.h"

#include <charconv>

#include "api_service/request_id.h"
#include "core/common/constants.h"
#include "core/util/verbose_trace_logger.h"

namespace xllm {

Call::Call(brpc::Controller* controller,
           std::string body_x_request_id,
           bool is_http_request)
    : controller_(controller) {
  init(std::move(body_x_request_id), is_http_request);
}

void Call::init(std::string body_x_request_id, bool is_http_request) {
  if (controller_->http_request().GetHeader("x-request-time")) {
    x_request_time_ = *controller_->http_request().GetHeader("x-request-time");
  } else if (controller_->http_request().GetHeader("x-request-timems")) {
    x_request_time_ =
        *controller_->http_request().GetHeader("x-request-timems");
  }

  x_request_id_ =
      api_service::resolve_x_request_id(controller_, body_x_request_id);
  if (is_http_request) {
    controller_->http_response().SetHeader("x-request-id", x_request_id_);
  }

  XLLM_VERBOSE_TRACE() << "event=request_received x-request-id="
                       << x_request_id_
                       << " path=" << controller_->http_request().uri().path();

  init_request_payload();
}

std::string Call::take_request_payload() {
  std::string payload;
  request_payload_.copy_to(&payload);
  request_payload_.clear();
  return payload;
}

void Call::init_request_payload() {
  const auto infer_content_len =
      controller_->http_request().GetHeader(kInferContentLength);
  const auto content_len =
      controller_->http_request().GetHeader(kContentLength);

  if (infer_content_len == nullptr || content_len == nullptr) {
    return;
  }

  size_t infer_len = 0;
  size_t len = 0;
  const auto [infer_end, infer_error] =
      std::from_chars(infer_content_len->data(),
                      infer_content_len->data() + infer_content_len->size(),
                      infer_len);
  const auto [content_end, content_error] = std::from_chars(
      content_len->data(), content_len->data() + content_len->size(), len);
  if (infer_error != std::errc() || content_error != std::errc() ||
      infer_end != infer_content_len->data() + infer_content_len->size() ||
      content_end != content_len->data() + content_len->size() ||
      infer_len > len || len > controller_->request_attachment().size()) {
    LOG(ERROR) << "Invalid binary request payload length.";
    return;
  }

  const size_t payload_size = len - infer_len;
  const size_t appended = controller_->request_attachment().append_to(
      &request_payload_, payload_size, infer_len);
  if (appended != payload_size) {
    request_payload_.clear();
    LOG(ERROR) << "failed to retain binary request payload: expected "
               << payload_size << " bytes, got " << appended;
  }
}

}  // namespace xllm
