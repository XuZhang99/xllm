#pragma once

#include "common/macros.h"
#include "page_manager_client.h"
#include "page_manager_server.h"

namespace xllm {
class PageManagerPool {
 public:
  explicit PageManagerPool(const memory::Options& options, int32_t dp_size);
  ~PageManagerPool() = default;

  void setup_single_node_page_managers();
  void setup_multi_node_page_managers(const std::string& master_node_addr);

  int32_t get_manager_with_max_free_pages() const;
  int32_t get_dp_rank(Sequence* sequence) const;

  bool allocate(Sequence* sequence);
  bool allocate(std::vector<Sequence*>& sequences);
  bool allocate(Sequence* sequence, size_t num_tokens);

  void deallocate(Request* request);
  void deallocate(std::vector<Sequence*>& sequences);
  void deallocate(Sequence* sequence);

  void cache(Sequence* sequence);

  std::vector<size_t> num_free_pages_per_layer() const;
  std::vector<size_t> num_used_pages_per_layer() const;
  double kv_cache_utilization() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(PageManagerPool);

 private:
  memory::Options options_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  std::vector<std::shared_ptr<PageManagerClient>> page_manager_clients_;
  std::vector<std::shared_ptr<PageManager>> page_managers_;
  std::vector<std::unique_ptr<PageManagerServer>> page_manager_servers_;
}
}  // namespace xllm