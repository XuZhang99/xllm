#include "common/global_flags.h"
#include "multi_layer_xtensor_instance.h"

namespace xllm {

void MultiLayerXTensorTransfer::initialize(
    const std::vector<torch::Device>& devices) {
  for (const auto& device : devices) {
    multi_layer_k_xtensor_map_[device.index()] = nullptr;
    multi_layer_v_xtensor_map_[device.index()] = nullptr;
  }
}

void MultiLayerXTensorTransfer::set_multi_layer_xtensor(
    std::vector<std::shared_ptr<XTensor>>& k_xtensors,
    std::vector<std::shared_ptr<XTensor>>& v_xtensors,
    torch::Device device) {
  multi_layer_k_xtensor_map_[device.index()] =
      std::make_unique<MultiLayerXTensor>(k_xtensors);
  if (!FLAGS_enable_mla) {
    multi_layer_v_xtensor_map_[device.index()] =
        std::make_unique<MultiLayerXTensor>(v_xtensors);
  }
}

MultiLayerXTensorPair MultiLayerXTensorTransfer::move_multi_layer_xtensor(
    int32_t device_id) {
  auto k_it = multi_layer_k_xtensor_map_.find(device_id);
  auto v_it = multi_layer_v_xtensor_map_.find(device_id);
  CHECK(k_it != multi_layer_k_xtensor_map_.end())
      << "MultiLayerXTensor not set for device " << device_id;
  CHECK(v_it != multi_layer_v_xtensor_map_.end())
      << "MultiLayerXTensor not set for device " << device_id;

  multi_layer_k_xtensor_map_.erase(k_it);
  multi_layer_v_xtensor_map_.erase(v_it);
  if (FLAGS_enable_mla) {
    return std::make_pair(std::move(k_it->second), nullptr);
  }
  return std::make_pair(std::move(k_it->second), std::move(v_it->second));
}
}  // namespace xllm