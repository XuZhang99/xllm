#include "xtensor.h"

#include "acl/acl.h"
#include "common/global_flags.h"

namespace xllm {
XTensor::XTensor(const Options& options, torch::ScalarType dtype)
    : options_(options), dtype_(dtype) {
  cache_size_per_token_ = options_.num_kv_heads() * options_.head_size() *
                          torch::scalarTypeToTypeMeta(dtype_).itemsize();

  buffer_size_per_seq_ = cache_size_per_token_ * options_.max_context_len();

  // align up to granularity size
  int64_t granularity_size = options_.granularity_size();
  buffer_size_per_seq_ = (buffer_size_per_seq_ + granularity_size - 1) /
                         granularity_size * granularity_size;
  FLAGS_buffer_size_per_seq = buffer_size_per_seq_;

  // buffer size for all sequences
  buffer_size_ = buffer_size_per_seq_ * options_.max_seqs_per_batch();

  reserve_base_ptr();
}

XTensor::XTensor(int64_t buffer_size) : buffer_size_(buffer_size) {
  options_.max_seqs_per_batch() = 1;
  reserve_base_ptr();
}

XTensor::~XTensor() {
  aclError status = aclrtReleaseMemAddress(base_ptr_);
  CHECK_EQ(status, ACL_SUCCESS) << "Failed to free virtual memory for xtensor";
}

void XTensor::reserve_base_ptr() {
  aclError status =
      aclrtReserveMemAddress(&base_ptr_, buffer_size_, 0, nullptr, 0);
  CHECK_EQ(status, ACL_SUCCESS)
      << "Failed to reserve virtual memory for xtensor";
}
}  // namespace xllm
