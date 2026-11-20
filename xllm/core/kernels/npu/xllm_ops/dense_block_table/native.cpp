/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/core/DeviceGuard.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <vector>

namespace {

torch::Tensor translate(const torch::Tensor& logical_table,
                        const torch::Tensor& mapping,
                        int64_t layers,
                        const std::vector<int64_t>& output_shape,
                        const pybind11::function& launch,
                        const pybind11::object& restore_padding) {
  const c10::DeviceGuard device_guard(logical_table.device());
  const c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
  c10_npu::NPUCachingAllocator::recordStream(logical_table.storage().data_ptr(),
                                             stream);
  c10_npu::NPUCachingAllocator::recordStream(mapping.storage().data_ptr(),
                                             stream);
  const uintptr_t stream_handle =
      reinterpret_cast<uintptr_t>(stream.stream(/*need_empty=*/false));

  if (restore_padding.is_none()) {
    torch::Tensor output = torch::empty(output_shape, logical_table.options());
    launch(logical_table,
           mapping,
           output,
           pybind11::arg("stream") = stream_handle);
    return output;
  }

  const torch::Tensor indices = torch::empty(
      {logical_table.numel()}, logical_table.options().dtype(torch::kInt64));
  launch(logical_table, indices, pybind11::arg("stream") = stream_handle);
  torch::Tensor output =
      torch::index_select(mapping.view({layers, -1}), /*dim=*/1, indices);
  restore_padding(
      logical_table, output, pybind11::arg("stream") = stream_handle);
  return output.view(output_shape);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("translate", &translate);
}
