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

#include "core/framework/batch/sequence_batch.h"

namespace xllm {

std::vector<Sequence*> SequenceBatch::get_sequences() const {
  if (!state_.sequence_plan().empty()) {
    return state_.sequence_plan().sequences();
  }
  return state_.group_sequences();
}

ForwardInput SequenceBatch::prepare_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    int32_t cp_size) {
  return state_.prepare_sequence_input(
      num_decoding_tokens, min_decoding_batch_size, args, cp_size);
}

ForwardInput SequenceBatch::prepare_forward_input(const ModelArgs& args,
                                                  ThreadPool* thread_pool,
                                                  int32_t cp_size) {
  return state_.prepare_distributed_input(args, thread_pool, cp_size);
}

}  // namespace xllm
