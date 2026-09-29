/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <memory>
#include <vector>

#include "core/framework/request/dit_request.h"
#include "core/runtime/dit_forward_params.h"

namespace xllm {

class DiTBatch final {
 public:
  DiTBatch() = default;
  void add(std::shared_ptr<DiTRequest> request);
  void reserve(size_t request_count) { request_vec_.reserve(request_count); }
  size_t size() const { return request_vec_.size(); }
  bool empty() const { return request_vec_.empty(); }

  // prepare forward input
  DiTForwardInput prepare_forward_input();

  void process_forward_output(const DiTForwardOutput& output);

 private:
  std::vector<std::shared_ptr<DiTRequest>> request_vec_;
};

}  // namespace xllm
