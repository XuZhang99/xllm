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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/request/request.h"

namespace xllm {

// Composed by sequence and Rec factories to assemble their selected input
// contract. Only configuration is retained; each invocation owns its scratch
// state and the scheduler continues to own the requests and sequences.
class BatchAssembler final {
 public:
  BatchAssembler(int32_t dp_size, BatchInputType input_type);

  std::vector<Batch> assemble(
      const std::vector<std::shared_ptr<Request>>& requests,
      const std::vector<Sequence*>& sequences,
      const std::vector<size_t>& budgets,
      std::vector<std::vector<BlockTransferInfo>>* swap_infos) const;

 private:
  int32_t dp_size_;
  BatchInputType input_type_;
};

}  // namespace xllm
