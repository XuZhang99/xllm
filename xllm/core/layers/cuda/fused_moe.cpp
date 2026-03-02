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

#include "fused_moe.h"

#include <glog/logging.h>

#include <iomanip>

#include "common/global_flags.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "layers/common/dp_utils.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(const ModelArgs& model_args,
                           const FusedMoEArgs& moe_args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_total_experts_(model_args.n_routed_experts()),
      topk_(model_args.num_experts_per_tok()),
      hidden_size_(model_args.hidden_size()),
      is_gated_(moe_args.is_gated),
      hidden_act_(model_args.hidden_act()),
      quant_args_(quant_args),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()) {
  const int64_t num_experts = num_total_experts_;
  const int64_t intermediate_size =
      static_cast<int64_t>(model_args.moe_intermediate_size());
  int64_t ep_size = parallel_args.ep_size();
  int64_t ep_rank = 0;
  tp_pg_ = parallel_args.tp_group_;
  if (ep_size > 1) {
    ep_rank = parallel_args.moe_ep_group_->rank();
    tp_pg_ = parallel_args.moe_tp_group_;
  }

  // calculate the number of experts per rank
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  gate_ = register_module("gate", MoEGate(model_args, quant_args, options));

  // create weight buffer
  const int64_t world_size = tp_pg_->world_size();
  int64_t local_intermediate_size = intermediate_size / world_size;

  w13_ = register_parameter(
      "w13",
      torch::empty(
          {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
          options_),
      false);
  w2_ = register_parameter(
      "w2",
      torch::empty(
          {num_experts_per_rank_, hidden_size_, local_intermediate_size},
          options_),
      false);
}

torch::Tensor FusedMoEImpl::create_group_gemm_output(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& group_list,
    torch::ScalarType dtype,
    torch::Tensor& workspace) {
  // unify shape logic: define the target shape once.
  bool is_3d_weight = (b.dim() != 2);
  int64_t num_tokens = a.size(0);
  int64_t out_dim = is_3d_weight ? b.size(1) : b.size(0);

  std::vector<int64_t> output_shape;
  int64_t required_elements = num_tokens * out_dim;

  if (is_3d_weight) {
    output_shape = {num_tokens, out_dim};
  } else {
    output_shape = {group_list.size(0), num_tokens, out_dim};
    required_elements *= group_list.size(0);
  }

  auto options = a.options().dtype(dtype);

  // smoothquant: managed workspace logic
  if (!workspace.defined()) {
    // Lazy initialization: allocate max buffer for the lifecycle
    // Note: accessing class members w13_ and w2_ directly for context
    int64_t max_width = std::max(w13_.size(1), w2_.size(1));
    workspace = torch::empty({num_tokens * max_width}, options);
  }

  // view construction
  CHECK(workspace.numel() >= required_elements)
      << "FusedMoE Workspace too small! Alloc: " << workspace.numel()
      << ", Req: " << required_elements;

  // utilize the pre-calculated output_shape
  return workspace.slice(0, 0, required_elements).view(output_shape);
}

torch::Tensor FusedMoEImpl::forward_experts(const torch::Tensor& hidden_states,
                                            bool enable_all2all_communication) {
  // Dispatcher: route to the appropriate path based on communication mode
  if (enable_all2all_communication) {
    return forward_experts_all2all(hidden_states);
  } else {
    return forward_experts_base(hidden_states);
  }
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& input_params) {
  auto input = hidden_states;
  auto output = forward_experts(input);

  return output;
}

void FusedMoEImpl::load_experts(const StateDict& state_dict) {
  LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
  LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
}

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
