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

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

void batch_prefill(torch::Tensor& float_workspace_buffer,
                   torch::Tensor& int_workspace_buffer,
                   torch::Tensor& page_locked_int_workspace_buffer,
                   const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   const torch::Tensor& q_cu_seq_lens,
                   const torch::Tensor& kv_cu_seq_lens,
                   int64_t window_size_left,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph) {
  std::string uri = get_batch_prefill_uri(/*backend=*/"fa2",
                                          query.scalar_type(),
                                          key.scalar_type(),
                                          output.scalar_type(),
                                          q_cu_seq_lens.scalar_type(),
                                          query.size(-1),
                                          value.size(-1),
                                          /*pos_encoding_mode=*/0,
                                          /*use_sliding_window=*/false,
                                          /*use_logits_soft_cap=*/false,
                                          /*use_fp16_qk_reduction=*/false);

  torch::Tensor kv_indptr_host = kv_cu_seq_lens.to(torch::kCPU);
  torch::Tensor qo_indptr_host = q_cu_seq_lens.to(torch::kCPU);
  torch::Tensor kv_len_arr_host =
      kv_indptr_host.slice(0, 1) - kv_indptr_host.slice(0, 0, -1);
  const int64_t total_num_rows = qo_indptr_host.size(0);
  const int64_t batch_size = q_cu_seq_lens.size(0) - 1;
  const double sm_scale = compute_sm_scale(query.size(-1));

  // plan prefill
  //   auto plan_info = get_module(uri)->GetFunction("plan").value()(
  //       to_ffi_tensor(float_workspace_buffer),
  //       to_ffi_tensor(int_workspace_buffer),
  //       to_ffi_tensor(page_locked_int_workspace_buffer),
  //       to_ffi_tensor(qo_indptr_host),
  //       to_ffi_tensor(kv_indptr_host),
  //       to_ffi_tensor(kv_len_arr_host),
  //       total_num_rows,
  //       batch_size,
  //       query.size(1),  // num_qo_heads
  //       key.size(1),    // num_kv_heads
  //       /*page_size=*/1,
  //       enable_cuda_graph,
  //       query.size(-1),   // head_dim_qk
  //       value.size(-1),   // head_dim_vo
  //       /*causal=*/true,  // causal
  //       /*window_size_left=*/-1,
  //       /*fixed_split_size=*/-1,
  //       /*disable_split_kv=*/false);

  //   auto prefill_plan_info = plan_info.cast<ffi::Array<int64_t>>();

  //   std::cout << "prefill_plan_info: " << prefill_plan_info.size() <<
  //   std::endl; for (int i = 0; i < prefill_plan_info.size(); i++) {
  //     std::cout << "prefill_plan_info[" << i << "]: " << prefill_plan_info[i]
  //               << std::endl;
  //   }

  std::vector<int64_t> plan_info(15);
  plan_info[0] = 1;
  plan_info[1] = 2;
  plan_info[2] = 0;
  plan_info[3] = 128;
  plan_info[4] = 0;
  plan_info[5] = 16;
  plan_info[6] = 32;
  plan_info[7] = 0;
  plan_info[8] = 48;
  plan_info[9] = 56;
  plan_info[10] = 0;
  plan_info[11] = 0;
  plan_info[12] = 0;
  plan_info[13] = 0;
  plan_info[14] = 0;

  ffi::Array<int64_t> prefill_plan_info = ffi::Array<int64_t>(plan_info);

  // batch prefill
  get_module(uri)
      ->GetFunction("ragged_run")
      .value()(to_ffi_tensor(float_workspace_buffer),
               to_ffi_tensor(int_workspace_buffer),
               prefill_plan_info,
               to_ffi_tensor(query),
               to_ffi_tensor(key),
               to_ffi_tensor(value),
               to_ffi_tensor(q_cu_seq_lens),
               to_ffi_tensor(kv_cu_seq_lens),
               to_ffi_tensor(output),
               output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                                      : ffi::Optional<ffi::Tensor>(),
               /*mask_mode_code=CAUSAL*/ 1,
               /*kv_layout_code=*/0,  // NHD layout
               /*window_size_left=*/-1,
               support_pdl(),
               /*maybe_custom_mask=*/ffi::Optional<ffi::Tensor>(),
               /*maybe_mask_indptr=*/ffi::Optional<ffi::Tensor>(),
               /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
               /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
               /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
               /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
               /*logits_soft_cap=*/0.0,
               /*sm_scale=*/sm_scale,
               /*rope_rcp_scale=*/1.0,
               /*rope_rcp_theta=*/1.0 / 10000.0,
               /*token_pos_in_items_len=*/0);
  std::cout << "batch_prefill cuda run end" << std::endl;
}

}  // namespace xllm::kernel::cuda