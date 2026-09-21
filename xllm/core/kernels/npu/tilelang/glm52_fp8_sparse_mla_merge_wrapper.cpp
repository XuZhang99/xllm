/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_GLM52_FP8_SPARSE_MLA_MERGE_REGISTRY_INC
#error "XLLM_TL_GLM52_FP8_SPARSE_MLA_MERGE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_GLM52_FP8_SPARSE_MLA_MERGE_REGISTRY_INC
}  // namespace

void glm52_fp8_sparse_mla_merge(const torch::Tensor& partial,
                                const torch::Tensor& stats,
                                torch::Tensor& output,
                                int64_t num_splits) {
  CHECK(partial.defined() && stats.defined() && output.defined());
  CHECK(output.device().type() == torch::kPrivateUse1);
  CHECK(partial.device() == output.device() &&
        stats.device() == output.device());
  CHECK(partial.is_contiguous() && stats.is_contiguous() &&
        output.is_contiguous());
  CHECK_EQ(partial.scalar_type(), torch::kFloat32);
  CHECK_EQ(stats.scalar_type(), torch::kFloat32);
  CHECK_EQ(output.scalar_type(), torch::kBFloat16);
  CHECK_EQ(partial.dim(), 3);
  CHECK_EQ(stats.dim(), 3);
  CHECK_EQ(output.dim(), 3);
  CHECK_EQ(partial.size(1), output.size(1));
  CHECK_EQ(partial.size(2), 512);
  CHECK_EQ(output.size(2), 512);
  CHECK_EQ(stats.size(1), 2);
  CHECK_EQ(stats.size(2), 16);
  CHECK(output.size(1) == 4 || output.size(1) == 8 || output.size(1) == 16);
  CHECK(num_splits == 2 || num_splits == 4 || num_splits == 8 ||
        num_splits == 16);
  CHECK_LE(output.size(0) * num_splits, 24);
  CHECK_GE(partial.size(0), output.size(0) * num_splits);
  CHECK_GE(stats.size(0), output.size(0) * num_splits);
  if (output.size(0) == 0) {
    return;
  }
  const auto specialization = make_glm52_fp8_sparse_mla_merge_specialization(
      Glm52Fp8SparseMlaMergeNumHeads{static_cast<int32_t>(output.size(1))},
      Glm52Fp8SparseMlaMergeNumSplits{static_cast<int32_t>(num_splits)});
  const auto* entry =
      find_glm52_fp8_sparse_mla_merge_kernel_entry(specialization);
  CHECK(entry != nullptr) << "No compiled FP8 MLA split merge specialization";
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(output.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(partial.data_ptr()),
            static_cast<uint8_t*>(stats.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()),
            static_cast<int32_t>(output.size(0)),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
