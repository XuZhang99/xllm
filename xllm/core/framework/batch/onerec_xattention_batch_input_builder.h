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

#include "core/framework/batch/onerec_batch_input_builder.h"

namespace xllm {

// Isolated builder type for the future OneRec xattention pipeline.
// It currently reuses the legacy OneRec builder behavior, but keeps a separate
// type boundary so new step-meta / multi-round input organization can be added
// without polluting OneRecBatchInputBuilder.
class OneRecXAttentionBatchInputBuilder final : public OneRecBatchInputBuilder {
 public:
  OneRecXAttentionBatchInputBuilder(const BatchInputData& data,
                                    const ModelArgs* args,
                                    MPMCThreadPool* thread_pool = nullptr)
      : OneRecBatchInputBuilder(data, args, thread_pool),
        sequence_groups_(data.sequence_groups),
        allowed_max_tokens_(data.allowed_max_tokens),
        args_(args) {}

  ForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) override;

 private:
  const std::vector<SequencesGroup*>& sequence_groups_;
  const std::vector<uint32_t>& allowed_max_tokens_;
  const ModelArgs* args_ = nullptr;
};

}  // namespace xllm
