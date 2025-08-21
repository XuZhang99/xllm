#include "page_allocator.h"

namespace xllm {

PageAllocator::PageAllocator(const memory::Options& options)
    : options_(options) {
  CHECK_GT(options_.total_pages(), 0) << "No pages to allocate";
  CHECK_GT(options_.granularity_size(), 0)
      << "Granularity size must be positive";
  CHECK_EQ(options_.total_pages() % options_.num_layers(), 0)
      << "Total physical pages must be divisible by number of layers";

  num_free_phy_pages_per_layer_ =
      options_.total_pages() / options_.num_layers();

  free_phy_pages_.resize(options_.num_layers());
  for (auto& free_phy_pages_per_layer : free_phy_pages_) {
    free_phy_pages_per_layer.resize(num_free_phy_pages_per_layer_);
  }

  for (int64_t i = 0; i < options_.num_layers(); ++i) {
    for (int64_t j = num_free_phy_pages_per_layer_ - 1; j >= 0; --j)
      free_phy_pages_[i][j] = std::make_shared<PhyPage>(
          options_.device(), options_.granularity_size());
  }

  free_phy_page_ids_.reserve(num_free_phy_pages_per_layer_);
  for (int64_t i = num_free_phy_pages_per_layer_ - 1; i >= 0; --i) {
    free_phy_page_ids_.push_back(i);
  }
}

PageAllocator::~PageAllocator() { free_phy_pages_.clear(); }

std::vector<uint32_t> PageAllocator::allocate(int64_t n_pages_per_layer) {
  CHECK_LT(n_pages_per_layer, num_free_phy_pages_per_layer_)
      << "Not enough physical pages available";
  std::vector<uint32_t> phy_page_ids;
  phy_page_ids.resize(n_pages_per_layer);

  for (int64_t i = 0; i < n_pages_per_layer; ++i) {
    phy_page_ids[i] = allocate();
  }

  return phy_page_ids;
}

uint32_t PageAllocator::allocate() {
  CHECK_GT(num_free_phy_pages_per_layer_, 0)
      << "No more physical pages available";

  uint32_t phy_page_id;
  phy_page_id = free_phy_page_ids_[--num_free_phy_pages_per_layer_];

  return phy_page_id;
}

void PageAllocator::deallocate(std::vector<uint32_t>& page_ids) {
  for (auto& page_id : page_ids) {
    deallocate(page_id);
  }
}

// caller should make sure the page_id is valid
void PageAllocator::deallocate(uint32_t page_id) {
  CHECK_LT(num_free_phy_pages_per_layer_, free_phy_page_ids_.size());
  free_phy_page_ids_[num_free_phy_pages_per_layer_++] = page_id;
}

// map one virtual pointer to one physical page
void PageAllocator::map(void* vir_ptr, aclrtDrvMemHandle phy_handle) const {
  aclError status;
  status = aclrtMapMem(vir_ptr, options_.granularity_size(), 0, phy_handle, 0);
  CHECK_EQ(status, ACL_SUCCESS)
      << "Failed to map virtual address to physical address";
}

void PageAllocator::map(void* vir_ptr,
                        uint32_t page_id,
                        int64_t layer_idx) const {
  aclrtDrvMemHandle phy_handle =
      free_phy_pages_[layer_idx][page_id]->get_phy_handle();
  map(vir_ptr, phy_handle);
}

void PageAllocator::batch_map(void* vir_ptr,
                              std::vector<uint32_t>& page_ids,
                              uint32_t num_new_pages,
                              int64_t layer_idx) const {
  size_t num_pages = page_ids.size();

  size_t ptr_offset = (num_pages - num_new_pages) * options_.granularity_size();

  void* temp_vir_ptr = reinterpret_cast<void*>((char*)vir_ptr + ptr_offset);

  for (size_t j = num_new_pages; j > 0; --j) {
    uint32_t page_id = page_ids[num_pages - j];
    map(temp_vir_ptr, page_id, layer_idx);
    temp_vir_ptr = reinterpret_cast<void*>((char*)temp_vir_ptr +
                                           options_.granularity_size());
  }
}
}  // namespace xllm
