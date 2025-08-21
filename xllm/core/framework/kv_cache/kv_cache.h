#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm {
class KVCache final {
 public:
  KVCache() = default;
  KVCache(torch::Tensor key_cache, torch::Tensor value_cache);
  KVCache(std::shared_ptr<XTensor> key_xtensor,
          std::shared_ptr<XTensor> value_xtensor);
  ~KVCache() = default;

  // TODO: pass in kv_shape and options instead
  torch::Tensor get_k_cache() const;
  torch::Tensor get_v_cache() const;

  std::shared_ptr<XTensor> get_k_xtensor() const;
  std::shared_ptr<XTensor> get_v_xtensor() const;

  bool empty() const {
    return FLAGS_enable_continuous_kvcache
               ? (key_xtensor_ == nullptr || value_xtensor_ == nullptr)
               : (!key_cache_.defined() || !value_cache_.defined());
  }

 private:
  torch::Tensor key_cache_;
  torch::Tensor value_cache_;

  // for continuous kvcache
  std::shared_ptr<XTensor> key_xtensor_;
  std::shared_ptr<XTensor> value_xtensor_;
};

}  // namespace xllm
