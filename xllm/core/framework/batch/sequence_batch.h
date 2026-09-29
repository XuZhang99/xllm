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

#include "core/framework/batch/batch_state.h"

namespace xllm {

// LLM/VLM batch: scheduled sequence rows and budgets are authoritative.
class SequenceBatch final {
 public:
  BatchInputType input_type() const { return BatchInputType::SEQUENCE; }
  BatchState& state() { return state_; }
  const BatchState& state() const { return state_; }
  size_t size() const { return state_.sequence_plan().size(); }
  Sequence* sequence(size_t index) const {
    return state_.sequence_plan()[index].sequence;
  }
  std::vector<Sequence*> get_sequences() const;
  void refresh_sequences_from_groups() {
    state_.refresh_sequences_from_groups();
  }
  ForwardInput prepare_forward_input(uint32_t num_decoding_tokens,
                                     uint32_t min_decoding_batch_size,
                                     const ModelArgs& args,
                                     int32_t cp_size);
  ForwardInput prepare_forward_input(const ModelArgs& args,
                                     ThreadPool* thread_pool,
                                     int32_t cp_size);

 private:
  BatchState state_;
};

}  // namespace xllm
