#include "kv_cache.h"

namespace xllm {

KVCache::KVCache(torch::Tensor key_cache, torch::Tensor value_cache)
    : key_cache_(std::move(key_cache)), value_cache_(std::move(value_cache)) {}

KVCache::KVCache(std::shared_ptr<XTensor> key_xtensor,
                 std::shared_ptr<XTensor> value_xtensor)
    : key_xtensor_(key_xtensor), value_xtensor_(value_xtensor) {}

torch::Tensor KVCache::get_k_cache() const { return key_cache_; }
torch::Tensor KVCache::get_v_cache() const { return value_cache_; }

std::shared_ptr<XTensor> KVCache::get_k_xtensor() const { return key_xtensor_; }
std::shared_ptr<XTensor> KVCache::get_v_xtensor() const {
  return value_xtensor_;
}
}  // namespace xllm
