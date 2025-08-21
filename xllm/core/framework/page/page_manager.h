#pragma once

#include <folly/Unit.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "common/threadpool.h"
#include "common/type_traits.h"
#include "memory/options.h"
#include "memory/page_allocator.h"

namespace xllm {

class PageManager {
 public:
  explicit PageManager(const memory::Options& options,
                       const torch::Device& device);

  ~PageManager() override = default;

  bool allocate(int32_t& seq_id, size_t num_tokens);
  void deallocate(int32_t seq_id);
  void cache(int32_t seq_id);

  folly::SemiFuture<bool> allocate_async(int32_t& seq_id, size_t num_tokens);
  folly::SemiFuture<folly::Unit> deallocate_async(int32_t seq_id);
  folly::SemiFuture<folly::Unit> cache_async(int32_t seq_id);

  size_t num_free_pages_per_layer() const;
  size_t num_used_pages_per_layer() const;
  double kv_cache_utilization() const;

 private:
  void add_multi_layer_kv_xtensors();
  // allocate seq id for sequence
  int32_t allocate_seq_id();
  // release seq id for sequence
  void deallocate_seq_id(int32_t seq_id);
  bool has_enough_pages(size_t num_pages_per_layer);

 private:
  torch::Device device_;
  std::unique_ptr<PageAllocator> page_allocator_;
  MultiLayerXTensorPair multi_layer_kv_xtensor_;
  size_t num_used_pages_per_layer_ = 0;
  ThreadPool threadpool_;
};

}  // namespace xllm