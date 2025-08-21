#include "multi_layer_xtensor.h"

namespace xllm {

MultiLayerXTensor::MultiLayerXTensor(
    std::vector<std::shared_ptr<XTensor>>& xtensors)
    : xtensors_(xtensors) {
  num_layers_ = xtensors_.size();
  int32_t max_seqs_per_batch = xtensors_[0]->options().max_seqs_per_batch();

  phy_page_ids_vec_.resize(max_seqs_per_batch);
  num_free_seq_ids_ = max_seqs_per_batch;
  free_seq_ids_.reserve(max_seqs_per_batch);
  for (int32_t i = max_seqs_per_batch - 1; i >= 0; i--) {
    free_seq_ids_.push_back(i);
  }
}

void MultiLayerXTensor::append_phy_pages(
    int32_t seq_id,
    const std::vector<uint32_t>& new_phy_pages) {
  phy_page_ids_vec_[seq_id].insert(phy_page_ids_vec_[seq_id].end(),
                                   new_phy_pages.begin(),
                                   new_phy_pages.end());
}

void MultiLayerXTensor::free(int32_t seq_id) {
  for (size_t layer_idx = 0; layer_idx < num_layers_; layer_idx++) {
    void* vir_ptr = get_vir_ptr(seq_id, layer_idx);
    aclError status = aclrtUnmapMem(vir_ptr);
    CHECK_EQ(status, ACL_SUCCESS) << "Failed to unmap virtual memory for layer "
                                  << layer_idx << " of sequence " << seq_id;
  }
  deallocate_seq_id(seq_id);
}

int32_t MultiLayerXTensor::allocate_seq_id() {
  CHECK_GT(num_free_seq_ids_, 0) << "No more available seq_id!";
  return free_seq_ids_[--num_free_seq_ids_];
}

void MultiLayerXTensor::deallocate_seq_id(int32_t seq_id) {
  CHECK_LT(num_free_seq_ids_, free_seq_ids_.size());
  free_seq_ids_[num_free_seq_ids_++] = seq_id;
}

}  // namespace xllm