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

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "core/framework/block/block_manager_pool.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/storage/kv_cache_utils.h"

namespace xllm {

class KVCacheTransferCoordinatorBase;
class ModelArgs;
class WorkerClient;
namespace runtime {
struct Options;
}

struct KVCacheManagerFactoryResult final {
  std::unique_ptr<KVCacheManager> manager;
  KVCacheShape shape;
};

// Tensor widths and multiplicities belonging to one scheduler mapping group.
// Target and draft tensors share block IDs, but may have different widths.
struct VmmKVCacheLayout final {
  std::vector<size_t> block_bytes;
  std::vector<size_t> tensor_counts;
};

class KVCacheManagerFactory final {
 public:
  static VmmKVCacheLayout describe_vmm_layout(const KVCacheCapacity& capacity,
                                              const ModelArgs& model_args,
                                              int64_t world_size,
                                              torch::ScalarType dtype,
                                              bool is_draft_engine = false);

  // Reserve persistent state tensors before the fixed physical pool consumes
  // free device memory. A speculative pair passes its combined state budget.
  static void initialize_vmm_pool(const runtime::Options& options,
                                  int64_t world_size,
                                  size_t worker_count,
                                  int64_t reserved_state_bytes,
                                  bool is_multimodal = false);

  static KVCacheCapacity estimate_capacity(
      const ModelArgs& model_args,
      const runtime::Options& options,
      torch::ScalarType dtype,
      int64_t world_size,
      const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
      bool is_multimodal = false,
      int64_t reserved_memory_bytes = 0,
      bool initialize_virtual_memory = true);

  static KVCacheManagerFactoryResult create(
      const KVCacheCapacity& kv_cache_capacity,
      const ModelArgs& model_args,
      int64_t world_size,
      torch::ScalarType dtype,
      BlockManagerPool::Options options,
      std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator,
      int32_t dp_size = 1,
      std::optional<HostCacheValidationOptions> host_validation_options =
          std::nullopt,
      const VmmKVCacheLayout* shared_vmm_layout = nullptr);
};

}  // namespace xllm
