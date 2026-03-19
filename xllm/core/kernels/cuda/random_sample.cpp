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

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cstdint>
#include <mutex>
#include <tuple>

#include "cuda_ops_api.h"

namespace {
std::tuple<int64_t, int64_t> get_seed_and_offset(int64_t increment,
                                                 const torch::Device& device) {
  auto gen = at::cuda::detail::getDefaultCUDAGenerator(device.index());
  std::lock_guard<std::mutex> lock(gen.mutex());
  auto* cuda_gen = at::check_generator<at::CUDAGeneratorImpl>(gen);

  auto seed = static_cast<int64_t>(cuda_gen->current_seed());
  auto offset = static_cast<int64_t>(cuda_gen->get_offset());
  cuda_gen->set_offset(static_cast<uint64_t>(offset + (increment + 3) / 4 * 4));

  return std::make_tuple(seed, offset);
}
}  // namespace

namespace xllm::kernel::cuda {

torch::Tensor random_sample(const torch::Tensor& probs) {
  const auto device = probs.device();
  torch::Tensor sampled;
  int64_t batch_size = probs.size(0);
  auto [seed, offset] = get_seed_and_offset(batch_size, device);

  if (probs.dim() == 3) {
    int64_t seq_len = probs.size(1);
    int64_t vocab_size = probs.size(2);
    auto flat_probs = probs.reshape({-1, vocab_size});

    get_function(/*uri=*/"sampling",
                 /*func_name=*/"sampling_from_probs")(
        to_ffi_tensor(flat_probs),
        to_ffi_tensor(sampled),
        /*maybe_indices=*/ffi::Optional<ffi::Tensor>(),
        /*deterministic=*/true,
        /*philox_seed=*/seed,
        /*philox_offset=*/offset);
    return sampled.reshape({batch_size, seq_len});
  }

  get_function(/*uri=*/"sampling",
               /*func_name=*/"sampling_from_probs")(
      to_ffi_tensor(probs),
      to_ffi_tensor(sampled),
      /*maybe_indices=*/ffi::Optional<ffi::Tensor>(),
      /*deterministic=*/true,
      /*philox_seed=*/seed,
      /*philox_offset=*/offset);
  return sampled.flatten();
}

}  // namespace xllm::kernel::cuda