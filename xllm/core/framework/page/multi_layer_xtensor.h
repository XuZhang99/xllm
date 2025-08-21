#pragma once

#include <memory>

#include "xtensor.h"
namespace xllm {
// for all sequences, for all layers
class MultiLayerXTensor final {
 public:
  explicit MultiLayerXTensor(std::vector<std::shared_ptr<XTensor>>& xtensors);

  ~MultiLayerXTensor() = default;

  void append_phy_pages(int32_t seq_id,
                        const std::vector<uint32_t>& new_phy_pages);

  void free(int32_t seq_id);

  int32_t allocate_seq_id();

  void deallocate_seq_id(int32_t seq_id);

  std::vector<uint32_t> get_phy_page_ids(int32_t seq_id) const {
    return phy_page_ids_vec_[seq_id];
  }

  void* get_vir_ptr(int32_t seq_id, int64_t layer_idx) const {
    return xtensors_[layer_idx]->get_vir_ptr(seq_id);
  }

  size_t get_num_pages_per_layer(int32_t seq_id) const {
    return phy_page_ids_vec_[seq_id].size();
  }

 private:
  int64_t num_layers_;
  std::vector<std::shared_ptr<XTensor>> xtensors_;  // [num_layers]

  // page_id for all sequence
  // for one sequence, page_id is same for all layers
  std::vector<std::vector<uint32_t>>
      phy_page_ids_vec_;  // [max_seqs_per_batch, num_pages_for_seq]

  int32_t num_free_seq_ids_ = 0;
  std::vector<int32_t> free_seq_ids_;
};

}  // namespace xllm
