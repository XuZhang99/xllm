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

#include "core/common/global_flags.h"
#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {
void batch_superpage_prefill(
    const std::string& uri,
    ffi::Array<int64_t> plan_info,
    torch::Tensor float_workspace_buffer,
    torch::Tensor int_workspace_buffer,
    torch::Tensor page_locked_int_workspace_buffer,
    torch::Tensor query,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor paged_kv_last_page_len,
    int64_t window_left,
    double sm_scale,
    torch::Tensor output,
    int64_t stride_ratio,  // stride_ratio = stride_block / stride_n
    int64_t block_size,
    int64_t max_kv_seq_len,
    std::optional<torch::Tensor>& output_lse,
    std::optional<torch::Tensor> qo_indptr,
    torch::Tensor kv_tile_indptr_buf,
    torch::Tensor kv_tile_contig_flags_buf,
    torch::Tensor vector_sparse_indptr_buf,
    torch::Tensor vector_sparse_indices_buf,
    torch::Tensor kv_lens_buffer) {
  VLOG(kGraphExecutorLogVerboseLevel) << "plan_info: " << plan_info;

  torch::Tensor qo_indptr_to_use;
  const int64_t batch_size = paged_kv_last_page_len.size(0);

  if (qo_indptr.has_value()) {
    // Use provided qo_indptr from attn_metadata
    // TODO: consturct qo_indptr in CUDA graph execution
    qo_indptr_to_use = qo_indptr.value();
    VLOG(kGraphExecutorLogVerboseLevel)
        << "use provided qo_indptr in CUDA graph execution";
  } else {
    // Create qo_indptr if not provided (backward compatibility)
    torch::Tensor qo_indptr_host =
        get_cache_buffer(batch_size + 1, torch::kCPU);
    qo_indptr_to_use = qo_indptr_host.to(torch::kCUDA);
  }

  torch::Tensor indices_buf = paged_kv_indices.to(torch::kInt32);

  torch::Tensor sparse_indptr;
  torch::Tensor sparse_indices;
  if (block_size == 1) {
    sparse_indptr = paged_kv_indptr;

    get_function("page",
                 "block_sparse_indices_to_vector_sparse_offsets_and_tile_"
                 "contig_flags")(to_ffi_tensor(indices_buf),
                                 to_ffi_tensor(paged_kv_indptr),
                                 to_ffi_tensor(vector_sparse_indices_buf),
                                 to_ffi_tensor(paged_kv_indptr),
                                 to_ffi_tensor(kv_lens_buffer),
                                 to_ffi_tensor(kv_tile_indptr_buf),
                                 to_ffi_tensor(kv_tile_contig_flags_buf),
                                 stride_ratio,
                                 /*stride_n=*/1,
                                 batch_size,
                                 block_size,
                                 /*kv_tile_size=*/128,
                                 max_kv_seq_len);
  } else {
    get_function("page",
                 "block_sparse_indices_to_vector_sparse_offsets_and_tile_"
                 "contig_flags")(to_ffi_tensor(indices_buf),
                                 to_ffi_tensor(paged_kv_indptr),
                                 to_ffi_tensor(vector_sparse_indices_buf),
                                 to_ffi_tensor(vector_sparse_indptr_buf),
                                 to_ffi_tensor(kv_lens_buffer),
                                 to_ffi_tensor(kv_tile_indptr_buf),
                                 to_ffi_tensor(kv_tile_contig_flags_buf),
                                 stride_ratio,
                                 /*stride_n=*/1,
                                 batch_size,
                                 block_size,
                                 /*kv_tile_size=*/128,
                                 max_kv_seq_len);

    sparse_indptr = vector_sparse_indptr_buf;
  }

  get_function(uri, "paged_run")(
      to_ffi_tensor(float_workspace_buffer),
      to_ffi_tensor(int_workspace_buffer),
      plan_info,
      to_ffi_tensor(query),
      to_ffi_tensor(k_cache),
      to_ffi_tensor(v_cache),
      to_ffi_tensor(qo_indptr_to_use),
      to_ffi_tensor(paged_kv_indptr),
      to_ffi_tensor(paged_kv_indices),
      to_ffi_tensor(paged_kv_last_page_len),
      to_ffi_tensor(output),
      output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                             : ffi::Optional<ffi::Tensor>(),
      /*mask_mode_code=*/1,  // CAUSAL
      /*kv_layout_code=*/0,  // NHD layout
      window_left,
      support_pdl(),
      /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      to_ffi_tensor(kv_tile_indptr_buf),
      to_ffi_tensor(kv_tile_contig_flags_buf),
      /*logits_soft_cap=*/0.0,
      sm_scale,
      /*token_pos_in_items_len=*/0);
}
}  // namespace xllm::kernel::cuda