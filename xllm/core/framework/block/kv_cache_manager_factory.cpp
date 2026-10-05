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

#include "core/framework/block/kv_cache_manager_factory.h"

#include <glog/logging.h>

#include <limits>
#include <utility>

#include "core/distributed_runtime/engine.h"
#include "core/framework/block/hierarchy_block_manager_pool.h"
#include "core/framework/kv_cache/kv_cache_utils.h"

namespace xllm {

KVCacheManagerFactoryResult KVCacheManagerFactory::create(
    const KVCacheCapacity& kv_cache_capacity,
    const ModelArgs& model_args,
    int64_t world_size,
    BlockManagerPool::Options options,
    Engine* engine,
    int32_t dp_size,
    std::optional<HostCacheValidationOptions> host_validation_options) {
  CHECK_GT(world_size, 0) << "world_size must be greater than 0";
  CHECK_GT(dp_size, 0) << "dp_size must be greater than 0";

  KVCacheShape shape(kv_cache_capacity, model_args, world_size);
  CHECK(shape.has_key_cache_shape())
      << "KV cache shape must contain a key cache shape";
  const std::vector<int64_t>& key_cache_shape = shape.key_cache_shape();
  CHECK(!key_cache_shape.empty()) << "KV cache shape cannot be empty";
  CHECK_GE(key_cache_shape.front(), 0)
      << "KV cache embedding block count cannot be negative";
  CHECK_LE(key_cache_shape.front(),
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "KV cache embedding block count exceeds uint32_t range";
  options.num_embedding_blocks(static_cast<uint32_t>(key_cache_shape.front()));

  if (options.enable_linear_state()) {
    CHECK_LE(kv_cache_capacity.num_linear_state_blocks(),
             static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        << "Linear state slot count exceeds int32_t range";
    options.linear_state_num_slots(
        static_cast<int32_t>(kv_cache_capacity.num_linear_state_blocks()));
  }

  if (host_validation_options.has_value()) {
    HostCacheValidationOptions& validation = *host_validation_options;
    validation.device_block_count = kv_cache_capacity.n_blocks();
    validation.has_key_cache_shape = shape.has_key_cache_shape();
    validation.has_grouped_cache_layout = shape.has_grouped_cache_layout();
    validation.has_conv_cache_shape = shape.has_conv_cache_shape();
    validation.has_ssm_cache_shape = shape.has_ssm_cache_shape();
    const std::optional<std::string> validation_error =
        validate_host_cache_options(validation);
    if (validation_error.has_value()) {
      LOG(FATAL) << *validation_error;
    }
  }

  std::unique_ptr<KVCacheManager> manager;
  if (options.enable_host_offload()) {
    CHECK(engine != nullptr)
        << "Engine is required for host-offload KV cache manager";
    manager =
        std::make_unique<HierarchyBlockManagerPool>(options, engine, dp_size);
  } else {
    manager = std::make_unique<BlockManagerPool>(options, dp_size);
  }

  return KVCacheManagerFactoryResult{std::move(manager), std::move(shape)};
}

}  // namespace xllm
