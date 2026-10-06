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

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

#include "core/common/device_monitor.h"
#include "core/common/metrics.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/block/hierarchy_block_manager_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/kv_cache/kv_cache_utils.h"
#include "core/framework/xtensor/phy_page_pool.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_client.h"
#include "models/model_registry.h"

namespace xllm {

int64_t KVCacheEstimator::estimate_memory_budget(
    const runtime::Options& options,
    const KVCacheEstimateContext& context) const {
  if (context.xtensor_cache_size.has_value()) {
    CHECK_GT(*context.xtensor_cache_size, 0)
        << "XTensor KV cache budget must be positive";
    LOG(INFO) << "XTensor mode: available memory from PhyPagePool: "
              << readable_size(*context.xtensor_cache_size);
    return *context.xtensor_cache_size;
  }

  CHECK(!context.worker_memory.empty())
      << "KV cache estimation requires worker memory snapshots";
  const int64_t encoder_cache_reserved_bytes =
      context.is_multimodal ? options.max_encoder_cache_size() * 1024 * 1024
                            : 0;
  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < context.worker_memory.size(); ++i) {
    const KVCacheMemorySnapshot& memory = context.worker_memory[i];
    int64_t available_memory = memory.available_memory;
    const int64_t total_memory = memory.total_memory;
    LOG(INFO) << "worker #" << i
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ". Using max_memory_utilization: "
              << options.max_memory_utilization()
              << ", max_cache_size: " << readable_size(options.max_cache_size())
              << ", encoder_cache_reserved: "
              << readable_size(encoder_cache_reserved_bytes);
    GAUGE_SET(weight_size_in_kilobytes,
              (total_memory - available_memory) / 1024);
    GAUGE_SET(total_memory_size_in_kilobytes, total_memory / 1024);
    if (options.max_memory_utilization() < 1.0) {
      const int64_t buffer_memory = static_cast<int64_t>(
          total_memory * (1.0 - options.max_memory_utilization()));
      available_memory -= buffer_memory;
    }
    if (options.max_cache_size() > 0) {
      available_memory = std::min(available_memory, options.max_cache_size());
    }
    available_memory -= encoder_cache_reserved_bytes;
    cache_size_in_bytes = std::min(cache_size_in_bytes, available_memory);
  }
  return cache_size_in_bytes;
}

KVCacheCapacity KVCacheEstimator::estimate(
    const runtime::Options& options,
    const KVCacheEstimateContext& context) const {
  KVCacheEstimateOptions estimate_options;
  estimate_options.dtype = context.dtype;
  estimate_options.kv_cache_dtype = options.kv_cache_dtype();
  estimate_options.indexer_cache_dtype =
      ::xllm::KVCacheConfig::get_instance().indexer_cache_dtype();
  estimate_options.cache_size_in_bytes =
      estimate_memory_budget(options, context);
  estimate_options.block_size = options.block_size();
  estimate_options.world_size = context.world_size;
  estimate_options.max_seqs_per_batch =
      static_cast<int64_t>(options.max_seqs_per_batch());
  estimate_options.max_concurrent_requests = static_cast<int64_t>(
      ::xllm::ServiceConfig::get_instance().max_concurrent_requests());
  estimate_options.max_tokens_per_batch =
      static_cast<int64_t>(options.max_tokens_per_batch());
  estimate_options.max_tokens_per_chunk_for_prefill =
      static_cast<int64_t>(options.max_tokens_per_chunk_for_prefill());
  estimate_options.max_linear_state_cache_slots =
      options.max_linear_state_cache_slots();
  estimate_options.linear_state_cache_block_limit =
      context.linear_state_cache_block_limit;
  estimate_options.is_draft_engine = options.is_draft_engine();
  estimate_options.enable_chunked_prefill = options.enable_chunked_prefill();
  estimate_options.enable_schedule_overlap = options.enable_schedule_overlap();
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  estimate_options.enable_prefix_cache =
      kv_cache_config.enable_prefix_cache() &&
      !kv_cache_config.enable_xtensor();
  estimate_options.enable_disagg_pd = options.enable_disagg_pd();
  estimate_options.instance_role = options.instance_role();
  if (!context.is_multimodal) {
    estimate_options.num_speculative_tokens =
        static_cast<int64_t>(options.num_speculative_tokens());
    estimate_options.dp_size = options.dp_size();
    estimate_options.enable_dp_fair_token_budget =
        ::xllm::SchedulerConfig::get_instance().enable_dp_fair_token_budget();
    if (options.enable_mtp_draft_body_tp1() && options.is_draft_engine()) {
      estimate_options.world_size = 1;
    }
    estimate_options.layerwise_split_size =
        options.is_draft_engine()
            ? 1
            : ParallelConfig::get_instance().layerwise_split_size();

    if (options.enable_task_pipeline() &&
        options.num_speculative_tokens() > 0 && !options.is_draft_engine()) {
      estimate_options.embedding_context_bytes_per_block =
          sizeof(int64_t) + 2 * sizeof(int32_t);
      if (!SpeculativeConfig::is_block_diffusion_algorithm(
              options.speculative_algorithm())) {
        CHECK_GT(model_args_.hidden_size(), 0);
        const int64_t element_bytes = static_cast<int64_t>(
            torch::scalarTypeToTypeMeta(context.dtype).itemsize());
        estimate_options.embedding_context_bytes_per_block +=
            sizeof(int64_t) + 2 * model_args_.hidden_size() * element_bytes +
            sizeof(bool);
      }
    }
  }

  return estimate(std::move(estimate_options));
}

KVCacheCapacity KVCacheManagerFactory::estimate_capacity(
    const ModelArgs& model_args,
    const runtime::Options& options,
    torch::ScalarType dtype,
    int64_t world_size,
    const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
    bool is_multimodal) {
  KVCacheEstimateContext context;
  context.dtype = dtype;
  context.world_size = world_size;
  context.is_multimodal = is_multimodal;
  context.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args.model_type());
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  if (config.enable_xtensor() && !is_multimodal) {
    const auto& phy_pool = PhyPagePool::get_instance();
    CHECK(phy_pool.is_initialized()) << "PhyPagePool not initialized";
    context.xtensor_cache_size = static_cast<int64_t>(phy_pool.num_total()) *
                                 config.phy_page_granularity_size();
  } else {
    CHECK(!worker_clients.empty()) << "KV cache estimation requires workers";
    std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
    futures.reserve(worker_clients.size());
    for (const auto& worker : worker_clients) {
      futures.emplace_back(worker->estimate_kv_cache_capacity_async());
    }
    auto results = folly::collectAll(futures).get();
    context.worker_memory.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      if (!results[i].hasValue()) {
        LOG(ERROR) << "Failed to estimate kv cache capacity for worker: " << i;
        continue;
      }
      const auto& [available_memory, total_memory] = results[i].value();
      context.worker_memory.emplace_back(
          KVCacheMemorySnapshot{available_memory, total_memory});
    }
    CHECK(!context.worker_memory.empty())
        << "Failed to estimate KV cache capacity for all workers";
  }

  KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(options, context);
  GAUGE_SET(total_kv_cache_size_in_kilobytes,
            capacity.cache_size_in_bytes() / 1024);
  for (const auto& device : options.devices()) {
    DeviceMonitor::get_instance().set_total_kv_cache_memory(
        device.index(), capacity.cache_size_in_bytes());
    DeviceMonitor::get_instance().set_total_activation_memory(device.index());
  }
  return capacity;
}

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
