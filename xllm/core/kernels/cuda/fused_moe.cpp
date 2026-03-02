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

#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {

torch::Tensor cutlass_fused_moe(
    const torch::Tensor& input,                   // [num_tokens, hidden]
    const torch::Tensor& token_selected_experts,  // [num_tokens, top_k]
    const torch::Tensor& token_final_scales,      // [num_tokens, top_k]
    const torch::Tensor&
        fc1_expert_weights,  // [num_experts, inter_dim, hidden]
    const torch::Tensor&
        fc2_expert_weights,  // [num_experts, hidden, inter_dim]
    torch::ScalarType output_dtype,
    const std::vector<torch::Tensor>& quant_scales,
    const std::optional<torch::Tensor>& fc1_expert_biases,
    const std::optional<torch::Tensor>& fc2_expert_biases,
    const std::optional<torch::Tensor>& input_sf,
    const std::optional<torch::Tensor>& swiglu_alpha,
    const std::optional<torch::Tensor>& swiglu_beta,
    const std::optional<torch::Tensor>& swiglu_limit,
    int32_t tp_size,
    int32_t tp_rank,
    int32_t ep_size,
    int32_t ep_rank,
    int32_t cluster_size,
    int32_t cluster_rank,
    const std::optional<torch::Tensor>& output,
    bool enable_alltoall,
    bool use_deepseek_fp8_block_scale,
    bool use_w4_group_scaling,
    bool use_mxfp8_act_scaling,
    bool min_latency_mode,
    bool use_packed_weights,
    int32_t tune_max_num_tokens,
    ActivationType activation_type) {
  size_t num_rows = input.size(0);
  size_t hidden_size = fc2_expert_weights.size(1);

  if (min_latency_mode) {
    num_rows *= fc2_expert_weights.size(0);
  }

  std::vector<size_t> output_shape = {num_rows, hidden_size};
  if (!output.has_value()) {
    output.value() = torch::empty(output_shape, output_dtype, input.device());
  } else {
    check_shape_dtype_device(
        output.value(), output_shape, output_dtype, input.device, "output");
  }

  std::string fused_moe_uri = "fused_moe";
  if (Device::is_support_sm90a) {
    fused_moe_uri += "_90";
  } else if (Device::is_support_sm100a || Device::is_support_sm100f) {
    fused_moe_uri += "_100";
  } else if (Device::is_support_sm120a) {
    fused_moe_uri += "_120";
  } else {
    LOG(FATAL) << "FusedMoE is only supported on sm90, sm100, sm120.";
  }

  ffi::Module fused_moe_runner =
      get_function(fused_moe_uri, "init")(input.scalar_type(),
                                          fc1_expert_weights.scalar_type(),
                                          output_dtype,
                                          use_deepseek_fp8_block_scale,
                                          use_w4_group_scaling,
                                          use_mxfp8_act_scaling,
                                          use_packed_weights);

  fused_moe_runner->GetFunction("run_moe").value()(
      to_ffi_tensor(output.value()),
      to_ffi_tensor(input),
      to_ffi_tensor(token_selected_experts),
      to_ffi_optional_tensor(token_final_scales),
      to_ffi_tensor(fc1_expert_weights),
      to_ffi_optional_tensor(fc1_expert_biases),
      to_ffi_tensor(fc2_expert_weights),
      to_ffi_optional_tensor(fc2_expert_biases),
      to_ffi_optional_array_tensors(quant_scales),
      to_ffi_optional_tensor(input_sf),
      to_ffi_optional_tensor(swiglu_alpha),
      to_ffi_optional_tensor(swiglu_beta),
      to_ffi_optional_tensor(swiglu_limit),
      tp_size,
      tp_rank,
      ep_size,
      ep_rank,
      cluster_size,
      cluster_rank,
      enable_alltoall,
      min_latency_mode,
      /*profile_ids=*/ffi::Optional<ffi::Array<int64_t>>(),  // TODO: support
                                                             // auto tuning
                                                             // profile ids
      support_pdl(),
      activation_type);
}
}  // namespace xllm::kernel::cuda