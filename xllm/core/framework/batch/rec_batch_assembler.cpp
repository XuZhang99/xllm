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

#include "core/framework/batch/rec_batch_assembler.h"

#include <glog/logging.h>

#include "core/framework/batch/batch_assembly_plan.h"

namespace xllm {

RecBatchAssembler::RecBatchAssembler(int32_t dp_size, BatchInputType input_type)
    : dp_size_(dp_size),
      input_type_(input_type),
      uses_group_input_(RecBatch(input_type).uses_group_input()) {
  CHECK_GT(dp_size_, 0);
}

std::vector<Batch> RecBatchAssembler::assemble(
    const std::vector<std::shared_ptr<Request>>& requests,
    const std::vector<Sequence*>& sequences,
    const std::vector<size_t>& budgets,
    std::vector<std::vector<BlockTransferInfo>>* swap_infos) const {
  BatchAssemblyPlan plan(
      dp_size_, requests, sequences, budgets, uses_group_input_, swap_infos);
  std::vector<Batch> batches;
  batches.reserve(dp_size_);
  for (int32_t rank = 0; rank < dp_size_; ++rank) {
    batches.emplace_back(RecBatch(input_type_));
  }
  plan.populate(batches);
  return batches;
}

}  // namespace xllm
