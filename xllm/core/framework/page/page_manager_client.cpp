#include "page_manager_client.h"

namespace xllm {

bool PageManagerClient::allocate(int32_t& seq_id, size_t num_tokens) {
  return page_manager_->allocate(seq_id, num_tokens);
}

void PageManagerClient::deallocate(int32_t seq_id) {
  page_manager_->deallocate(seq_id);
}

void PageManagerClient::cache(int32_t seq_id) { page_manager_->cache(seq_id); }

folly::SemiFuture<bool> PageManagerClient::allocate_async(int32_t& seq_id,
                                                          size_t num_tokens) {
  return page_manager_->allocate_async(seq_id, num_tokens);
}

folly::SemiFuture<folly::Unit> PageManagerClient::deallocate_async(
    int32_t seq_id) {
  return page_manager_->deallocate_async(seq_id);
}

folly::SemiFuture<folly::Unit> PageManagerClient::cache_async(int32_t seq_id) {
  return page_manager_->cache_async(seq_id);
}

size_t PageManagerClient::num_free_pages_per_layer() const {
  return page_manager_->num_free_pages_per_layer();
}

size_t PageManagerClient::num_used_pages_per_layer() const {
  return page_manager_->num_used_pages_per_layer();
}

double PageManagerClient::kv_cache_utilization() const {
  return page_manager_->kv_cache_utilization();
}

}  // namespace xllm