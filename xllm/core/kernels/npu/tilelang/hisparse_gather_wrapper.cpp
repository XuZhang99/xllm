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

#ifndef XLLM_TL_HISPARSE_GATHER_REGISTRY_INC
#error "XLLM_TL_HISPARSE_GATHER_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_HISPARSE_GATHER_REGISTRY_INC

constexpr int64_t kMaxRows = 1048576;

}  // namespace

void hisparse_gather_out(const torch::Tensor& host,
                         const torch::Tensor& hot,
                         const torch::Tensor& slots,
                         const torch::Tensor& slot_map,
                         const torch::Tensor& tags,
                         torch::Tensor& out) {
  CHECK(host.device().type() == torch::kPrivateUse1);
  for (const auto& tensor : {host, hot, slots, slot_map, tags, out}) {
    CHECK(tensor.device() == host.device());
    CHECK(tensor.is_contiguous());
    CHECK_EQ(tensor.storage_offset(), 0);
  }
  CHECK_EQ(host.dim(), 2);
  CHECK_EQ(hot.dim(), 2);
  CHECK_EQ(out.dim(), 2);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(slot_map.dim(), 1);
  CHECK_EQ(tags.dim(), 1);
  CHECK(host.scalar_type() == torch::kBFloat16);
  CHECK(hot.scalar_type() == host.scalar_type());
  CHECK(out.scalar_type() == host.scalar_type());
  CHECK(slots.scalar_type() == torch::kInt32);
  CHECK(slot_map.scalar_type() == torch::kInt32);
  CHECK(tags.scalar_type() == torch::kInt32);
  CHECK_EQ(host.size(1), hot.size(1));
  CHECK_EQ(host.size(1), out.size(1));
  CHECK_EQ(out.size(0), slots.numel());
  CHECK_GE(slot_map.numel(), host.size(0));
  CHECK_EQ(tags.numel(), hot.size(0));
  CHECK_GT(host.size(0), 0);
  CHECK_GT(hot.size(0), 0);
  CHECK_LE(host.size(0), kMaxRows);
  CHECK_LE(hot.size(0), kMaxRows);
  CHECK_LE(slots.numel(), kMaxRows);
  if (slots.numel() == 0) {
    return;
  }
  const auto specialization = make_hisparse_gather_specialization(
      HisparseGatherHeadDim{static_cast<int32_t>(host.size(1))});
  const auto* entry = find_hisparse_gather_kernel_entry(specialization);
  CHECK(entry != nullptr) << "HiSparse supports BF16 head dimensions 64/512";
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(host.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(host.data_ptr()),
            static_cast<uint8_t*>(hot.data_ptr()),
            static_cast<uint8_t*>(slots.data_ptr()),
            static_cast<uint8_t*>(slot_map.data_ptr()),
            static_cast<uint8_t*>(tags.data_ptr()),
            static_cast<uint8_t*>(out.data_ptr()),
            static_cast<int32_t>(slots.numel()),
            static_cast<int32_t>(host.size(0)),
            static_cast<int32_t>(hot.size(0)),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
