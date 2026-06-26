/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>

#include "framework/parallel_state/process_group.h"

namespace xllm {

// Single-process multi-device process group backed by a raw ncclComm_t
// (created via ncclCommInitAll). Owns the comm and destroys it in the
// destructor. This deliberately bypasses c10d::ProcessGroupNCCL because two
// ProcessGroupNCCL instances coexisting inside one process can dispatch
// collectives in mismatched orders and deadlock under load. All collectives
// are queued on the device's current CUDA stream so they chain naturally with
// model-forward kernels.
class NcclProcessGroup final : public ProcessGroup {
 public:
  explicit NcclProcessGroup(int32_t rank,
                            int32_t world_size,
                            const torch::Device& device,
                            ncclComm_t comm);

  ~NcclProcessGroup() override;

  void allreduce(torch::Tensor& input) override;
  c10::intrusive_ptr<c10d::Work> allreduce_async(torch::Tensor& input) override;

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) override;
  c10::intrusive_ptr<c10d::Work> allgather_async(
      const torch::Tensor& input,
      std::vector<torch::Tensor>& outputs) override;

  c10::intrusive_ptr<c10d::Work> allgather_base_async(
      const torch::Tensor& input,
      torch::Tensor& output) override;
  torch::Tensor allgather_base_sync(const torch::Tensor& input) override;

  void reduce_scatter(const torch::Tensor& input,
                      torch::Tensor& output) override;

  void all_to_all_single(
      torch::Tensor output,
      torch::Tensor input,
      std::vector<int64_t> output_split_sizes = {},
      std::vector<int64_t> input_split_sizes = {},
      bool async_op = false,
      c10::intrusive_ptr<c10d::Work>* async_work = nullptr) override;

 private:
  // Returns the CUDA stream that NCCL ops should run on. We piggy-back on the
  // current stream of the device so that ops chain naturally with model
  // forward kernels (matches c10d::ProcessGroupNCCL behaviour from the
  // caller's perspective).
  at::cuda::CUDAStream nccl_stream();

  ncclComm_t comm_ = nullptr;
};

}  // namespace xllm
