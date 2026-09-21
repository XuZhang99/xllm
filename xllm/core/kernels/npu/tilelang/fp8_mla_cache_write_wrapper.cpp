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
#include <limits>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_FP8_MLA_CACHE_WRITE_REGISTRY_INC
#error "XLLM_TL_FP8_MLA_CACHE_WRITE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_FP8_MLA_CACHE_WRITE_REGISTRY_INC
}  // namespace

void fp8_mla_cache_write(const torch::Tensor& slots,
                         const torch::Tensor& latent,
                         const torch::Tensor& rope,
                         torch::Tensor& latent_cache,
                         torch::Tensor& rope_cache) {
  CHECK(slots.defined() && latent.defined() && rope.defined() &&
        latent_cache.defined() && rope_cache.defined());
  CHECK(latent_cache.device().type() == torch::kPrivateUse1);
  CHECK(slots.device() == latent_cache.device() &&
        latent.device() == latent_cache.device() &&
        rope.device() == latent_cache.device() &&
        rope_cache.device() == latent_cache.device());
  CHECK(slots.is_contiguous() && latent_cache.is_contiguous() &&
        rope_cache.is_contiguous());
  CHECK(slots.scalar_type() == torch::kInt32 ||
        slots.scalar_type() == torch::kInt64);
  CHECK(latent.scalar_type() == torch::kBFloat16 ||
        latent.scalar_type() == torch::kFloat16 ||
        latent.scalar_type() == torch::kFloat32);
  CHECK_EQ(rope.scalar_type(), latent.scalar_type());
  CHECK_EQ(latent_cache.scalar_type(), torch::kUInt8);
  CHECK_EQ(rope_cache.scalar_type(), torch::kUInt8);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(latent.dim(), 2);
  CHECK_EQ(rope.dim(), 2);
  CHECK_EQ(latent_cache.dim(), 2);
  CHECK_EQ(rope_cache.dim(), 2);
  CHECK_EQ(latent.size(0), slots.numel());
  CHECK_EQ(rope.size(0), slots.numel());
  CHECK_EQ(latent.size(1), 512);
  CHECK_EQ(rope.size(1), 64);
  CHECK_EQ(latent_cache.size(1), 512);
  CHECK_EQ(rope_cache.size(1), 64);
  CHECK_EQ(latent_cache.size(0), rope_cache.size(0));
  CHECK_EQ(latent.stride(1), 1);
  CHECK_EQ(rope.stride(1), 1);
  CHECK_GE(latent.stride(0), 512);
  CHECK_GE(rope.stride(0), 64);
  CHECK_LE(slots.numel(), std::numeric_limits<int32_t>::max());
  CHECK_LE(latent_cache.size(0), std::numeric_limits<int32_t>::max());
  if (slots.numel() == 0) {
    return;
  }
  CHECK_GT(latent_cache.size(0), 0);
  const auto specialization = make_fp8_mla_cache_write_specialization(
      Fp8MlaCacheWriteDType{to_tilelang_dtype(latent.scalar_type())},
      Fp8MlaCacheWriteSlotBytes{static_cast<int32_t>(slots.element_size())});
  const auto* entry = find_fp8_mla_cache_write_kernel_entry(specialization);
  CHECK(entry != nullptr) << "No compiled FP8 MLA cache writer for dtype="
                          << latent.scalar_type()
                          << ", slot_bytes=" << slots.element_size();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(latent_cache.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(slots.data_ptr()),
            static_cast<uint8_t*>(latent.data_ptr()),
            static_cast<uint8_t*>(rope.data_ptr()),
            static_cast<uint8_t*>(latent_cache.data_ptr()),
            static_cast<uint8_t*>(rope_cache.data_ptr()),
            slots.numel(),
            latent.stride(0),
            rope.stride(0),
            latent_cache.size(0),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
