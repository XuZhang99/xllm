#pragma once

#include <folly/futures/Future.h>

#include <memory>

#include "common/threadpool.h"
#include "page_manager.h"

namespace xllm {
class PageManagerClient {
 public:
  PageManagerClient() = default;
  explicit PageManagerClient(PageManager* p) : page_manager_(p) {}
  virtual ~PageManagerClient() = default;

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
  PageManager* page_manager_ = nullptr;
};
}  // namespace xllm