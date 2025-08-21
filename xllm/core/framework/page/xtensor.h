#pragma once
#include <torch/torch.h>

#include <vector>

#include "common/macros.h"

namespace xllm {
// for all sequences
// for one layer
// k or v cache
class XTensor final {
 public:
  struct Options {
    PROPERTY(int64_t, num_kv_heads) = 0;
    PROPERTY(int64_t, head_size) = 0;
    PROPERTY(int32_t, max_context_len) = 0;
    PROPERTY(int32_t, max_seqs_per_batch) =
        0;  // TODO: rename to max_seqs_per_batch to num_vir_ptr
    PROPERTY(int64_t, granularity_size) = 2 * 1024 * 1024;  // 2MB
  };

  XTensor(const Options& options, torch::ScalarType dtype);
  XTensor(int64_t buffer_size);

  XTensor(XTensor&&) = default;
  XTensor& operator=(XTensor&&) = default;
  XTensor(const XTensor&) = delete;
  XTensor& operator=(const XTensor&) = delete;

  ~XTensor();

  void* get_base_ptr() const { return base_ptr_; }

  void* get_vir_ptr(int32_t seq_id) const {
    return reinterpret_cast<void*>((char*)base_ptr_ +
                                   seq_id * buffer_size_per_seq_);
  }

  const Options& options() const { return options_; }

 private:
  void reserve_base_ptr();

 private:
  Options options_;
  int64_t buffer_size_;
  torch::ScalarType dtype_;
  int64_t buffer_size_per_seq_;
  int64_t cache_size_per_token_;

  // the start virtual pointer of the xtensor
  void* base_ptr_ = nullptr;
};
}  // namespace xllm