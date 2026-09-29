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

#include <memory>
#include <vector>

#include "core/framework/batch/dit_batch.h"

namespace xllm {

// DiT requests have no autoregressive sequence or KV-budget contract.
class DiTBatchFactory final {
 public:
  std::vector<DiTBatch> create_batches(
      const std::vector<std::shared_ptr<DiTRequest>>& requests) const;
};

}  // namespace xllm
