#pragma once

#include <unordered_set>

#include "option.h"
#include "phy_page.h"

namespace xllm {

// PageAllocator is used to track memory pages of key and value. It is not
// thread safe. This class manages the allocation and deallocation of page.
class PageAllocator final {
 public:
  explicit PageAllocator(const memory::Options& options);

  ~PageAllocator();

  // disable copy, move and assign
  PageAllocator(const PageAllocator&) = delete;
  PageAllocator(PageAllocator&&) = delete;
  PageAllocator& operator=(const PageAllocator&) = delete;
  PageAllocator& operator=(PageAllocator&&) = delete;

  // allocate a list of page_ids for key or value for all layers
  std::vector<uint32_t> allocate(int64_t n_pages_per_layer);

  // allocate a page id for key or value for all layers
  uint32_t allocate();

  // get num of total physical pages for key and value for all layers
  size_t get_num_total_phy_pages_per_layer() const {
    return free_phy_page_ids_.size();
  }

  // get num of free physical pages for key and value for one layer
  size_t get_num_free_phy_pages_per_layer() const {
    return num_free_phy_pages_per_layer_;
  }

  // get num of used physical pages for key and value for one layer
  size_t get_num_used_phy_pages_per_layer() const {
    return free_phy_page_ids_.size() - num_free_phy_pages_per_layer_;
  }

  // get back one page to allocator
  void deallocate(uint32_t page_id);

  // get back a list of pages to allocator
  void deallocate(std::vector<uint32_t>& page_ids);

  void map(void* vir_ptr, aclrtDrvMemHandle phy_handle) const;
  void map(void* vir_ptr, uint32_t page_id, int64_t layer_idx) const;
  void batch_map(void* vir_ptr,
                 std::vector<uint32_t>& page_ids,
                 uint32_t num_new_pages,
                 int64_t layer_idx) const;

 private:
  memory::Options options_;

  // free physical pages
  std::vector<std::vector<std::shared_ptr<PhyPage>>>
      free_phy_pages_;  // [num_layers, num_total_pages_per_layer]

  int64_t num_free_phy_pages_per_layer_;

  std::vector<uint32_t> free_phy_page_ids_;
};
}  // namespace xllm