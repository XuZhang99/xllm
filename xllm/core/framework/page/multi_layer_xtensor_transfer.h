#pragma once
#include <map>

#include "common/type_traits.h"
#include "multi_layer_xtensor.h"

namespace xllm {

class MultiLayerXTensorTransfer {
 public:
  static MultiLayerXTensorTransfer& get_instance() {
    static MultiLayerXTensorTransfer instance;
    return instance;
  }

  void initialize(const std::vector<torch::Device>& devices);

  void set_multi_layer_xtensor(
      std::vector<std::shared_ptr<XTensor>>& k_xtensors,
      std::vector<std::shared_ptr<XTensor>>& v_xtensors,
      torch::Device device);

  MultiLayerXTensorPair move_multi_layer_xtensor(int32_t device_id);

 private:
  MultiLayerXTensorTransfer() = default;
  ~MultiLayerXTensorTransfer() = default;
  DISALLOW_COPY_AND_ASSIGN(MultiLayerXTensorTransfer);

 private:
  std::map<int32_t, std::unique_ptr<MultiLayerXTensor>>
      multi_layer_k_xtensor_map_;
  std::map<int32_t, std::unique_ptr<MultiLayerXTensor>>
      multi_layer_v_xtensor_map_;
};

}  // namespace xllm