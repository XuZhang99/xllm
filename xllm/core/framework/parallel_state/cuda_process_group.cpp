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

#include "cuda_process_group.h"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <glog/logging.h>
#include <nccl.h>

#include <chrono>
#include <utility>
#include <vector>

namespace xllm {

namespace {

#define XLLM_NCCLCHECK(cmd)                                          \
  do {                                                               \
    ncclResult_t r = (cmd);                                          \
    if (r != ncclSuccess) {                                          \
      LOG(FATAL) << "NCCL error " << ncclGetErrorString(r) << " at " \
                 << __FILE__ << ":" << __LINE__;                     \
    }                                                                \
  } while (0)

ncclDataType_t to_nccl_data_type(const torch::Tensor& input) {
  switch (input.scalar_type()) {
    case torch::kFloat:
      return ncclFloat32;
    case torch::kHalf:
      return ncclFloat16;
    case torch::kDouble:
      return ncclFloat64;
    case torch::kLong:
      return ncclInt64;
    case torch::kInt:
      return ncclInt32;
    case torch::kChar:
      return ncclInt8;
    case torch::kByte:
      return ncclUint8;
    case torch::kBool:
      return ncclUint8;
    case torch::kBFloat16:
      return ncclBfloat16;
    default:
      LOG(FATAL) << "Unsupported tensor dtype for NCCL: "
                 << input.scalar_type();
  }
}

// Lightweight c10d::Work that waits on a single CUDA event recorded after
// the NCCL launch. The compute path always queues NCCL on a CUDA stream and
// then issues blocking waits via `wait()`, so this is enough to surface
// completion to the caller without taking on the full ProcessGroupNCCL
// state machine.
class CudaNcclWork : public c10d::Work {
 public:
  CudaNcclWork(c10d::OpType op_type,
               at::cuda::CUDAStream stream,
               const torch::Device& device)
      : c10d::Work(/*rank=*/-1, op_type),
        event_(/*flags=*/cudaEventDisableTiming),
        device_(device) {
    event_.record(stream);
  }

  bool isCompleted() override {
    c10::cuda::CUDAGuard guard(device_);
    return event_.query();
  }

  bool wait(std::chrono::milliseconds /*timeout*/ = kNoTimeout) override {
    c10::cuda::CUDAGuard guard(device_);
    // Make the current stream wait for completion of the NCCL work that was
    // recorded on the issuing stream. This matches the semantics callers get
    // from c10d::ProcessGroupNCCL where wait() is a stream-side join, not a
    // host-side synchronize.
    event_.block(c10::cuda::getCurrentCUDAStream(device_.index()));
    return true;
  }

 private:
  at::cuda::CUDAEvent event_;
  torch::Device device_;
};

c10::intrusive_ptr<c10d::Work> make_work(c10d::OpType op_type,
                                         at::cuda::CUDAStream stream,
                                         const torch::Device& device) {
  return c10::make_intrusive<CudaNcclWork>(op_type, stream, device);
}

}  // namespace

ProcessGroupImpl::ProcessGroupImpl(int32_t global_rank,
                                   int32_t world_size,
                                   int32_t rank_size,
                                   int32_t port,
                                   bool trans,
                                   const std::string& host,
                                   const std::string& group_name,
                                   const torch::Device& device)
    : ProcessGroup(global_rank, world_size, device) {
  c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> pg_options =
      c10d::ProcessGroupNCCL::Options::create();
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
  pg_options->group_name = group_name;
#endif
  int32_t rank = global_rank;
  if (world_size != rank_size) {
    auto [local_rank, group_ranks] =
        get_group_rank(world_size, global_rank, rank_size, trans);
    pg_options->global_ranks_in_group = group_ranks;
    rank = local_rank;
  }

  auto store = create_tcp_store(host, port, rank);
  pg_ = std::make_unique<c10d::ProcessGroupNCCL>(
      store, rank, rank_size, pg_options);
}

ProcessGroupImpl::ProcessGroupImpl(int32_t rank,
                                   int32_t world_size,
                                   const torch::Device& device,
                                   ncclComm_t comm)
    : ProcessGroup(rank, world_size, device), comm_(comm) {
  CHECK(comm != nullptr) << "ncclComm_t must be non-null";
}

ProcessGroupImpl::~ProcessGroupImpl() {
  if (uses_raw_nccl()) {
    // We own the comm in the raw-NCCL path; release it before the base class
    // destructor runs so any device state can be torn down cleanly.
    c10::cuda::CUDAGuard guard(device());
    ncclCommDestroy(comm_);
    comm_ = nullptr;
  }
}

at::cuda::CUDAStream ProcessGroupImpl::nccl_stream() {
  return c10::cuda::getCurrentCUDAStream(device().index());
}

void ProcessGroupImpl::allreduce(torch::Tensor& input) {
  if (!uses_raw_nccl()) {
    ProcessGroup::allreduce(input);
    return;
  }
  allreduce_async(input)->wait();
}

c10::intrusive_ptr<c10d::Work> ProcessGroupImpl::allreduce_async(
    torch::Tensor& input) {
  if (!uses_raw_nccl()) {
    return ProcessGroup::allreduce_async(input);
  }
  CHECK_EQ(input.device(), device())
      << "allreduce input must live on the process group's device";
  CHECK(input.is_contiguous()) << "allreduce input must be contiguous";

  c10::cuda::CUDAGuard guard(device());
  at::cuda::CUDAStream stream = nccl_stream();
  XLLM_NCCLCHECK(ncclAllReduce(input.data_ptr(),
                               input.data_ptr(),
                               static_cast<size_t>(input.numel()),
                               to_nccl_data_type(input),
                               ncclSum,
                               comm_,
                               stream.stream()));
  return make_work(c10d::OpType::ALLREDUCE, stream, device());
}

void ProcessGroupImpl::allgather(const torch::Tensor& input,
                                 std::vector<torch::Tensor>& outputs) {
  if (!uses_raw_nccl()) {
    ProcessGroup::allgather(input, outputs);
    return;
  }
  allgather_async(input, outputs)->wait();
}

c10::intrusive_ptr<c10d::Work> ProcessGroupImpl::allgather_async(
    const torch::Tensor& input,
    std::vector<torch::Tensor>& outputs) {
  if (!uses_raw_nccl()) {
    return ProcessGroup::allgather_async(input, outputs);
  }
  CHECK_EQ(static_cast<int32_t>(outputs.size()), world_size())
      << "allgather output count must equal world_size";
  CHECK_EQ(input.device(), device())
      << "allgather input must live on the process group's device";
  CHECK(input.is_contiguous()) << "allgather input must be contiguous";

  c10::cuda::CUDAGuard guard(device());
  at::cuda::CUDAStream stream = nccl_stream();
  // ncclAllGather requires a single contiguous receive buffer; build one
  // sized [world_size, *input.shape] then copy into the per-rank outputs
  // afterwards. This mirrors the ProcessGroup::allgather + cat semantics
  // callers expect.
  std::vector<int64_t> stacked_shape;
  stacked_shape.reserve(input.dim() + 1);
  stacked_shape.push_back(world_size());
  for (int64_t s : input.sizes()) {
    stacked_shape.push_back(s);
  }
  torch::Tensor stacked = torch::empty(stacked_shape, input.options());
  XLLM_NCCLCHECK(ncclAllGather(input.data_ptr(),
                               stacked.data_ptr(),
                               static_cast<size_t>(input.numel()),
                               to_nccl_data_type(input),
                               comm_,
                               stream.stream()));
  // Slice stacked into the supplied outputs. Each slice shares storage with
  // stacked, so the gather kernel only writes once.
  for (int32_t i = 0; i < world_size(); ++i) {
    if (!outputs[i].defined()) {
      outputs[i] = stacked.select(0, i);
    } else {
      outputs[i].copy_(stacked.select(0, i), /*non_blocking=*/true);
    }
  }
  return make_work(c10d::OpType::ALLGATHER, stream, device());
}

c10::intrusive_ptr<c10d::Work> ProcessGroupImpl::allgather_base_async(
    const torch::Tensor& input,
    torch::Tensor& output) {
  if (!uses_raw_nccl()) {
    return ProcessGroup::allgather_base_async(input, output);
  }
  CHECK_EQ(input.device(), device())
      << "allgather_base input must live on the process group's device";
  CHECK(output.defined()) << "allgather_base output must be preallocated";
  CHECK_EQ(output.device(), device())
      << "allgather_base output must live on the process group's device";
  CHECK(output.is_contiguous()) << "allgather_base output must be contiguous";

  torch::Tensor input_buf = input.contiguous();
  CHECK_EQ(output.numel(), input_buf.numel() * world_size())
      << "allgather_base output size must equal world_size * input size";

  c10::cuda::CUDAGuard guard(device());
  at::cuda::CUDAStream stream = nccl_stream();
  XLLM_NCCLCHECK(ncclAllGather(input_buf.data_ptr(),
                               output.data_ptr(),
                               static_cast<size_t>(input_buf.numel()),
                               to_nccl_data_type(input_buf),
                               comm_,
                               stream.stream()));
  return make_work(c10d::OpType::_ALLGATHER_BASE, stream, device());
}

torch::Tensor ProcessGroupImpl::allgather_base_sync(
    const torch::Tensor& input) {
  if (!uses_raw_nccl()) {
    return ProcessGroup::allgather_base_sync(input);
  }
  CHECK_EQ(input.device(), device())
      << "allgather_base input must live on the process group's device";
  std::vector<int64_t> out_shape;
  out_shape.reserve(input.dim() + 1);
  out_shape.push_back(world_size());
  for (int64_t s : input.sizes()) {
    out_shape.push_back(s);
  }
  torch::Tensor output = torch::empty(out_shape, input.options());
  allgather_base_async(input, output)->wait();
  return output;
}

void ProcessGroupImpl::reduce_scatter(const torch::Tensor& input,
                                      torch::Tensor& output) {
  if (!uses_raw_nccl()) {
    ProcessGroup::reduce_scatter(input, output);
    return;
  }
  CHECK(input.is_contiguous()) << "reduce_scatter input must be contiguous";
  CHECK_EQ(input.device(), device())
      << "reduce_scatter input must live on the process group's device";
  CHECK(output.defined()) << "reduce_scatter output must be defined";
  CHECK_EQ(output.device(), device())
      << "reduce_scatter output must live on the process group's device";
  CHECK_EQ(input.numel(), output.numel() * world_size())
      << "reduce_scatter input size must equal world_size * output size";

  c10::cuda::CUDAGuard guard(device());
  at::cuda::CUDAStream stream = nccl_stream();
  XLLM_NCCLCHECK(ncclReduceScatter(input.data_ptr(),
                                   output.data_ptr(),
                                   static_cast<size_t>(output.numel()),
                                   to_nccl_data_type(input),
                                   ncclSum,
                                   comm_,
                                   stream.stream()));
  // Block the caller until the scatter completes so the semantics match the
  // base class (which calls ->wait()).
  make_work(c10d::OpType::REDUCE_SCATTER, stream, device())->wait();
}

void ProcessGroupImpl::all_to_all_single(
    torch::Tensor output,
    torch::Tensor input,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    bool async_op,
    c10::intrusive_ptr<c10d::Work>* async_work) {
  if (!uses_raw_nccl()) {
    ProcessGroup::all_to_all_single(output,
                                    input,
                                    std::move(output_split_sizes),
                                    std::move(input_split_sizes),
                                    async_op,
                                    async_work);
    return;
  }
  CHECK(output.defined()) << "all_to_all_single output must be defined";
  CHECK(input.defined()) << "all_to_all_single input must be defined";
  CHECK_EQ(input.device(), device())
      << "all_to_all_single input must live on the process group's device";
  CHECK_EQ(output.device(), device())
      << "all_to_all_single output must live on the process group's device";

  // Treat complex tensors the same way the base class does: split each
  // complex element into real+imag along the last dim before sending.
  if (input.is_complex()) {
    input = torch::view_as_real(input);
  }
  if (output.is_complex()) {
    output = torch::view_as_real(output);
  }
  CHECK(input.is_contiguous()) << "all_to_all_single input must be contiguous";
  CHECK(output.is_contiguous())
      << "all_to_all_single output must be contiguous";

  const int32_t ws = world_size();
  std::vector<int64_t> in_splits = input_split_sizes;
  std::vector<int64_t> out_splits = output_split_sizes;
  if (in_splits.empty()) {
    CHECK_EQ(input.size(0) % ws, 0)
        << "input dim 0 must be divisible by world_size for equal-split a2a";
    in_splits.assign(ws, input.size(0) / ws);
  }
  if (out_splits.empty()) {
    CHECK_EQ(output.size(0) % ws, 0)
        << "output dim 0 must be divisible by world_size for equal-split a2a";
    out_splits.assign(ws, output.size(0) / ws);
  }
  CHECK_EQ(static_cast<int32_t>(in_splits.size()), ws);
  CHECK_EQ(static_cast<int32_t>(out_splits.size()), ws);

  // NCCL has no direct alltoall-with-splits primitive; emulate via grouped
  // pairs of ncclSend / ncclRecv inside ncclGroupStart/End.
  const int64_t in_inner = input.numel() / std::max<int64_t>(1, input.size(0));
  const int64_t out_inner =
      output.numel() / std::max<int64_t>(1, output.size(0));
  const auto dtype = to_nccl_data_type(input);
  const size_t elem_size = static_cast<size_t>(input.element_size());
  char* input_ptr = static_cast<char*>(input.data_ptr());
  char* output_ptr = static_cast<char*>(output.data_ptr());

  c10::cuda::CUDAGuard guard(device());
  at::cuda::CUDAStream stream = nccl_stream();
  XLLM_NCCLCHECK(ncclGroupStart());
  int64_t in_offset = 0;
  int64_t out_offset = 0;
  for (int32_t r = 0; r < ws; ++r) {
    const size_t send_count = static_cast<size_t>(in_splits[r] * in_inner);
    const size_t recv_count = static_cast<size_t>(out_splits[r] * out_inner);
    if (send_count > 0) {
      XLLM_NCCLCHECK(ncclSend(input_ptr + in_offset * elem_size,
                              send_count,
                              dtype,
                              r,
                              comm_,
                              stream.stream()));
    }
    if (recv_count > 0) {
      XLLM_NCCLCHECK(ncclRecv(output_ptr + out_offset * elem_size,
                              recv_count,
                              dtype,
                              r,
                              comm_,
                              stream.stream()));
    }
    in_offset += in_splits[r] * in_inner;
    out_offset += out_splits[r] * out_inner;
  }
  XLLM_NCCLCHECK(ncclGroupEnd());

  auto work = make_work(c10d::OpType::ALLTOALL_BASE, stream, device());
  if (async_op) {
    CHECK(async_work != nullptr) << "async_work must be provided for async_op";
    *async_work = work;
  } else {
    work->wait();
  }
}

}  // namespace xllm
