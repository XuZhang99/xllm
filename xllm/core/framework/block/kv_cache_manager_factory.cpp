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
#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/vmm_arena_capacity.h"
#include "core/framework/block/hierarchy_block_manager_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/kv_cache/storage/vmm_kv_page_allocator_impl.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_client.h"
#include "models/model_registry.h"

namespace xllm {

VmmKVCacheLayout KVCacheManagerFactory::describe_vmm_layout(
    const KVCacheCapacity& capacity,
    const ModelArgs& model_args,
    int64_t world_size,
    torch::ScalarType dtype,
    bool is_draft_engine) {
  const KVCacheShape shape(capacity, model_args, world_size);
  CHECK(shape.has_value_cache_shape());
  CHECK_EQ(capacity.scale_slot_size(), 0)
      << "Virtual memory does not support quantized KV cache";
  CHECK(!capacity.enable_mla_kv_cache_quant());
  CHECK(!shape.has_grouped_cache_layout());
  CHECK(!capacity.enable_indexer_cache_quant() &&
        !shape.has_index_cache_scale_shape())
      << "Virtual memory does not support quantized indexer cache";
  CHECK(!model_args.index_kpool_compress())
      << "Virtual memory does not support KPool cache";
#if defined(USE_NPU)
  CHECK(!use_npu_nz_kv_cache_layout(model_args.model_type()))
      << "Virtual memory does not support the NPU NZ KV cache layout.";
#endif
  const KVCacheCreateOptions create_options =
      get_kv_cache_create_options(model_args, shape, is_draft_engine);
  size_t full_attention_layers = 0;
  for (int64_t layer_id = 0; layer_id < capacity.n_layers(); ++layer_id) {
    if (!create_options.enable_linear_attention() ||
        !is_linear_attention_layer(layer_id,
                                   create_options.full_attention_interval(),
                                   create_options.layer_types())) {
      ++full_attention_layers;
    }
  }
  CHECK_GT(full_attention_layers, 0)
      << "Virtual memory requires at least one full-attention layer";
  const size_t element_size = torch::elementSize(dtype);
  VmmKVCacheLayout layout;
  layout.block_bytes = {VirtualKVCachePageLayout::tensor_block_bytes(
                            shape.key_cache_shape(), element_size),
                        VirtualKVCachePageLayout::tensor_block_bytes(
                            shape.value_cache_shape(), element_size)};
  layout.tensor_counts = {full_attention_layers, full_attention_layers};
  if (shape.has_index_cache_shape()) {
    CHECK_EQ(dtype, torch::kBFloat16)
        << "Virtual memory indexer cache requires BF16";
    CHECK_EQ(shape.index_cache_shape().front(),
             shape.key_cache_shape().front());
    const std::vector<bool> indexer_layers =
        resolve_indexer_cache_enabled_layers(model_args, capacity.n_layers());
    size_t indexer_count = 0;
    for (int64_t layer_id = 0; layer_id < capacity.n_layers(); ++layer_id) {
      const size_t layer_index = static_cast<size_t>(layer_id);
      if ((!create_options.enable_linear_attention() ||
           !is_linear_attention_layer(layer_id,
                                      create_options.full_attention_interval(),
                                      create_options.layer_types())) &&
          (indexer_layers.empty() || indexer_layers[layer_index])) {
        ++indexer_count;
      }
    }
    if (indexer_count > 0) {
      layout.block_bytes.emplace_back(
          VirtualKVCachePageLayout::tensor_block_bytes(
              shape.index_cache_shape(), element_size));
      layout.tensor_counts.emplace_back(indexer_count);
    }
  }
  return layout;
}

void KVCacheManagerFactory::initialize_vmm_pool(const runtime::Options& options,
                                                int64_t world_size,
                                                size_t worker_count,
                                                int64_t reserved_state_bytes,
                                                bool is_multimodal) {
  CHECK(reserved_state_bytes == 0 ||
        !LoadConfig::get_instance().enable_vmm_weights())
      << "Virtual weight memory does not support hybrid linear-state caches.";
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  auto& coordinator = DistributedVirtualMemoryCoordinator::get_instance();
  const auto& phy_pool = PhysicalPagePool::get_instance();
  if (!phy_pool.is_initialized()) {
    const int64_t encoder_cache_reserved_bytes =
        is_multimodal ? options.max_encoder_cache_size() * 1024 * 1024 : 0;
    CHECK_GT(coordinator.init_physical_page_pools(
                 config.max_memory_utilization(),
                 config.max_cache_size(),
                 config.max_memory_utilization_configured(),
                 encoder_cache_reserved_bytes,
                 reserved_state_bytes),
             0)
        << "Failed to initialize KV cache physical page pools";
  }
  if (!coordinator.is_initialized()) {
    coordinator.init_page_budget(phy_pool.num_total(),
                                 options.dp_size(),
                                 static_cast<int32_t>(worker_count),
                                 config.phy_page_granularity_size(),
                                 /*enable_prealloc=*/true);
  }
  coordinator.set_model_parallel_strategy(
      options.model_id(), options.dp_size(), static_cast<int32_t>(world_size));
}

KVCacheCapacity KVCacheManagerFactory::estimate_capacity(
    const ModelArgs& model_args,
    const runtime::Options& options,
    torch::ScalarType dtype,
    int64_t world_size,
    const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
    bool is_multimodal,
    int64_t reserved_memory_bytes,
    bool initialize_virtual_memory) {
  KVCacheEstimateContext context;
  context.dtype = dtype;
  context.world_size = world_size;
  context.is_multimodal = is_multimodal;
  context.reserved_memory_bytes = reserved_memory_bytes;
  context.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args.model_type());
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  const auto& phy_pool = PhysicalPagePool::get_instance();
  if (config.enable_vmm_kvcache() && phy_pool.is_initialized()) {
    context.virtual_memory_cache_size =
        static_cast<int64_t>(phy_pool.num_total()) *
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
      CHECK(results[i].hasValue())
          << "Failed to estimate KV cache capacity for worker: " << i;
      const auto& [available_memory, total_memory] = results[i].value();
      context.worker_memory.emplace_back(
          KVCacheMemorySnapshot{available_memory, total_memory});
    }
    if (config.enable_vmm_kvcache()) {
      const int64_t encoder_bytes =
          is_multimodal ? options.max_encoder_cache_size() * 1024 * 1024 : 0;
      int64_t initial_bytes = std::numeric_limits<int64_t>::max();
      for (const KVCacheMemorySnapshot& memory : context.worker_memory) {
        const VmmArenaCapacity arena = calculate_vmm_arena_capacity(
            memory.available_memory,
            memory.total_memory,
            config.max_memory_utilization(),
            config.max_memory_utilization_configured(),
            config.vmm_elastic(),
            config.max_cache_size(),
            reserved_memory_bytes + encoder_bytes);
        initial_bytes = std::min(initial_bytes, arena.initial_bytes);
      }
      // Persistent states must fit startup memory, rather than the larger
      // virtual arena that an elastic pool can grow into later.
      context.virtual_memory_cache_size = initial_bytes;
    }
  }

  KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(options, context);
  if (config.enable_vmm_kvcache() && initialize_virtual_memory) {
    initialize_vmm_pool(options,
                        world_size,
                        worker_clients.size(),
                        capacity.linear_cache_size_in_bytes(),
                        is_multimodal);
    const VmmKVCacheLayout tensors = describe_vmm_layout(
        capacity, model_args, world_size, dtype, options.is_draft_engine());
    const VirtualKVCachePageLayout layout(tensors.block_bytes,
                                          config.phy_page_granularity_size());
    const size_t n_blocks = layout.block_capacity(
        PhysicalPagePool::get_instance().num_total(), tensors.tensor_counts);
    const int64_t pool_bytes =
        static_cast<int64_t>(PhysicalPagePool::get_instance().num_total()) *
        config.phy_page_granularity_size();
    capacity.n_blocks(static_cast<int64_t>(n_blocks))
        .cache_size_in_bytes(pool_bytes +
                             capacity.linear_cache_size_in_bytes());
  }
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
    torch::ScalarType dtype,
    BlockManagerPool::Options options,
    std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator,
    int32_t dp_size,
    std::optional<HostCacheValidationOptions> host_validation_options,
    const VmmKVCacheLayout* shared_vmm_layout) {
  CHECK_GT(world_size, 0) << "world_size must be greater than 0";
  CHECK_GT(dp_size, 0) << "dp_size must be greater than 0";
  CHECK(!options.enable_vmm_kvcache() ||
        (!options.enable_host_offload() &&
         (!host_validation_options.has_value() ||
          host_validation_options->host_blocks_factor <= 1.0)))
      << "Virtual memory does not support host KV cache offload; set "
         "--host_blocks_factor=0.";
  if (options.enable_vmm_kvcache() && options.enable_prefix_cache()) {
    LOG(WARNING) << "Virtual memory does not support prefix caching; "
                    "disabling prefix caching for this KV cache manager.";
    options.enable_prefix_cache(false);
  }
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

  VmmKVPageAllocator* page_allocator = nullptr;
  if (options.enable_vmm_kvcache()) {
    VmmKVCacheLayout component_layout =
        describe_vmm_layout(kv_cache_capacity, model_args, world_size, dtype);
    const VmmKVCacheLayout& tensors =
        shared_vmm_layout == nullptr ? component_layout : *shared_vmm_layout;
    auto& allocator = DistributedVirtualMemoryCoordinator::get_instance()
                          .vmm_kv_page_allocator();
    if (options.enable_disagg_pd()) {
      for (size_t bytes : tensors.block_bytes) {
        CHECK_LE(bytes, allocator.page_size())
            << "Virtual memory PD requires each tensor block to fit one page";
        CHECK_EQ(allocator.page_size() % bytes, 0)
            << "Virtual memory PD tensor blocks must divide the page size";
      }
    }
    const VirtualKVCachePageLayout layout(tensors.block_bytes,
                                          allocator.page_size());
    const size_t physical_pages = layout.physical_pages(tensors.tensor_counts);
    // Leave the tensor shape unchanged, but expose only complete page groups
    // to the scheduler. This prevents mapping past a short final tensor page.
    const size_t virtual_pages =
        static_cast<size_t>(kv_cache_capacity.n_blocks()) /
        layout.blocks_per_page();
    CHECK_GT(virtual_pages, 0)
        << "KV cache must fit one complete VM page group";
    CHECK(allocator.register_model(options.model_id(),
                                   options.num_layers(),
                                   physical_pages,
                                   virtual_pages));
    const size_t usable_pages =
        allocator.get_num_total_virt_pages(options.model_id());
    CHECK_GT(usable_pages, 0) << "Physical pool cannot fit one VM page group";
    const size_t usable_blocks = usable_pages * layout.blocks_per_page();
    LOG(INFO) << "Virtual KV cache: blocks=" << usable_blocks
              << ", blocks_per_page=" << layout.blocks_per_page()
              << ", physical_pages_per_group=" << physical_pages;
    options.virtual_block_bytes(layout.block_bytes())
        .num_blocks(static_cast<uint32_t>(usable_blocks));
    page_allocator = &allocator;
  }
  std::unique_ptr<KVCacheManager> manager;
  if (options.enable_host_offload()) {
    CHECK(transfer_coordinator != nullptr)
        << "KV cache transfer coordinator is required for host-offload KV "
           "cache manager";
    manager = std::make_unique<HierarchyBlockManagerPool>(
        options, std::move(transfer_coordinator), dp_size, page_allocator);
  } else {
    manager =
        std::make_unique<BlockManagerPool>(options, dp_size, page_allocator);
  }

  return KVCacheManagerFactoryResult{std::move(manager), std::move(shape)};
}

}  // namespace xllm
