/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch_npu/torch_npu.h>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_HISPARSE_STORE_REGISTRY_INC
#error "XLLM_TL_HISPARSE_STORE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_HISPARSE_STORE_REGISTRY_INC

constexpr int64_t kMaxRows = 1048576;

}  // namespace

void hisparse_store(const torch::Tensor& values,
                    const torch::Tensor& slots,
                    torch::Tensor& host) {
  CHECK(host.device().type() == torch::kPrivateUse1);
  for (const auto& tensor : {values, slots, host}) {
    CHECK(tensor.device() == host.device());
    CHECK(tensor.is_contiguous());
  }
  CHECK_EQ(values.dim(), 2);
  CHECK_EQ(host.dim(), 2);
  CHECK_EQ(slots.dim(), 1);
  CHECK(values.scalar_type() == torch::kBFloat16);
  CHECK(host.scalar_type() == values.scalar_type());
  CHECK(slots.scalar_type() == torch::kInt32);
  CHECK_EQ(values.size(0), slots.numel());
  CHECK_EQ(values.size(1), host.size(1));
  CHECK_GT(host.size(0), 0);
  CHECK_LE(host.size(0), kMaxRows);
  CHECK_LE(slots.numel(), kMaxRows);
  if (slots.numel() == 0) {
    return;
  }
  const auto specialization = make_hisparse_store_specialization(
      HisparseStoreHeadDim{static_cast<int32_t>(host.size(1))});
  const auto* entry = find_hisparse_store_kernel_entry(specialization);
  CHECK(entry != nullptr) << "HiSparse supports BF16 head dimensions 64/512";
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(host.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(values.data_ptr()),
            static_cast<uint8_t*>(slots.data_ptr()),
            static_cast<uint8_t*>(host.data_ptr()),
            static_cast<int32_t>(slots.numel()),
            static_cast<int32_t>(host.size(0)),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
