/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "process_group.h"

namespace xllm {

class ProcessGroupImpl : public ProcessGroup {
 public:
  // Multi-process constructor: bootstraps NCCL via TCPStore. One device per
  // process; this is the original path used by setup_multi_node_workers.
  ProcessGroupImpl(int32_t global_rank,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device);

  // Single-process multi-device constructor: receives a pre-initialized
  // ncclComm_t (created via ncclCommInitAll). Owns the comm and destroys it
  // in the destructor; bypasses c10d::ProcessGroupNCCL since two
  // ProcessGroupNCCL instances inside one process can deadlock under load.
  ProcessGroupImpl(int32_t rank,
                   int32_t world_size,
                   const torch::Device& device,
                   ncclComm_t comm);

  ~ProcessGroupImpl() override;

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
  bool uses_raw_nccl() const { return comm_ != nullptr; }
  // Returns the CUDA stream that NCCL ops should run on. We piggy-back on the
  // current stream of the device so that ops chain naturally with model
  // forward kernels (matches c10d::ProcessGroupNCCL behaviour from the
  // caller's perspective).
  at::cuda::CUDAStream nccl_stream();

  ncclComm_t comm_ = nullptr;
};

}  // namespace xllm
