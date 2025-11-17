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

#include "attention_metadata.h"

#include "core/common/global_flags.h"

namespace xllm {
namespace layer {

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           bool is_prefill) {
  return AttentionMetadata::build(params, "float", is_prefill);
}

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           const std::string& compute_dtype,
                                           bool is_prefill) {
  AttentionMetadata attn_metadata;
  attn_metadata.query_start_loc = params.q_seq_lens;
  attn_metadata.seq_start_loc = params.kv_seq_lens;
  attn_metadata.max_query_len = params.q_max_seq_len;
  attn_metadata.max_seq_len = params.kv_max_seq_len;
  attn_metadata.slot_mapping = params.new_cache_slots;
  attn_metadata.compute_dtype = compute_dtype;

  // for flashinfer
  attn_metadata.paged_kv_indptr = params.paged_kv_indptr;
  attn_metadata.paged_kv_indices = params.paged_kv_indices;
  attn_metadata.paged_kv_last_page_len = params.paged_kv_last_page_len;
  attn_metadata.q_cu_seq_lens = params.q_seq_lens;    // cumulative q seqlens
  attn_metadata.kv_cu_seq_lens = params.kv_seq_lens;  // cumulative kv seqlens

  bool is_start_loc_match = (params.q_seq_lens_vec == params.kv_seq_lens_vec);
  attn_metadata.is_chunked_prefill = is_prefill && !is_start_loc_match;
  attn_metadata.is_prefill = is_prefill && !attn_metadata.is_chunked_prefill;
  if (!attn_metadata.is_prefill || FLAGS_enable_mla) {
    attn_metadata.block_table = params.block_tables;
    attn_metadata.kv_seq_lens = torch::diff(params.kv_seq_lens);  // kv seqlens
  }

  attn_metadata.is_dummy = (params.q_max_seq_len == 0);

  return attn_metadata;
}

}  // namespace layer
}  // namespace xllm