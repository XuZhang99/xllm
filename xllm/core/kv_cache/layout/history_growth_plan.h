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

#include <glog/logging.h>

#include <cstddef>

namespace xllm {

// Logical append demand for a token history. This plan neither reserves blocks
// nor decides whether an allocator can satisfy the demand.
class HistoryGrowthPlan final {
 public:
  HistoryGrowthPlan(size_t token_stride,
                    size_t held_blocks,
                    size_t target_tokens)
      : held_blocks_(held_blocks) {
    CHECK_GT(token_stride, 0u);
    required_blocks_ = target_tokens / token_stride +
                       static_cast<size_t>(target_tokens % token_stride != 0);
  }

  size_t required_blocks() const { return required_blocks_; }
  size_t additional_blocks() const {
    return required_blocks_ > held_blocks_ ? required_blocks_ - held_blocks_
                                           : 0;
  }
  bool is_satisfied_by(size_t appended_blocks) const {
    return appended_blocks >= additional_blocks();
  }

 private:
  size_t held_blocks_;
  size_t required_blocks_;
};

}  // namespace xllm
