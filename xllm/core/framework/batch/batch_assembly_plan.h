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

#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/request/request.h"

namespace xllm {

// Per-call DP assembly plan. Borrows scheduler inputs until populate() returns;
// validates all ranks and budgets before consuming pending block transfers.
// Domain assemblers choose the concrete batch and the sequence/group contract.
class BatchAssemblyPlan final {
 public:
  BatchAssemblyPlan(int32_t dp_size,
                    const std::vector<std::shared_ptr<Request>>& requests,
                    const std::vector<Sequence*>& sequences,
                    const std::vector<size_t>& budgets,
                    bool group_input,
                    std::vector<std::vector<BlockTransferInfo>>* swap_infos);
  void populate(std::vector<Batch>& batches) const;

 private:
  int32_t dp_size_;
  const std::vector<std::shared_ptr<Request>>& requests_;
  const std::vector<Sequence*>& sequences_;
  const std::vector<size_t>& budgets_;
  bool group_input_;
  bool retain_request_groups_;
  std::vector<std::vector<BlockTransferInfo>>* swap_infos_;
  std::vector<size_t> sequence_counts_;
  std::vector<size_t> group_counts_;
  size_t num_prompt_tokens_ = 0;
  size_t num_generated_tokens_ = 0;
};

}  // namespace xllm
