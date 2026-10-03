/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <utility>

#include "core/framework/model/domain_attention_input.h"

namespace xllm {

class VlmModelParams final {
 public:
  VlmModelParams clone() const { return *this; }

  VlmModelParams to(const torch::Device& device) const {
    VlmModelParams params;
    params.meta = meta;
    params.attention = attention.to(device);
    params.embedding = embedding.to(device);
    params.block_copy = block_copy.to(device);
    params.multimodal = multimodal.to(device);
    params.parallel = parallel.to(device);
    params.expert = expert.to(device);
    params.graph = graph.to(device);
    params.linear_state_cache_ops = linear_state_cache_ops;
    params.linear_state_validity_mask = linear_state_validity_mask;
    params.is_spec_verify = is_spec_verify;
    params.prefill_without_cache = prefill_without_cache;
    params.num_accepted_tokens = safe_to(num_accepted_tokens, device, true);
    params.num_accepted_tokens_host = num_accepted_tokens_host;
    params.mtp_topk_state =
        mtp_topk_state == nullptr ? nullptr : mtp_topk_state->to(device);
    params.multi_block_tables.reserve(multi_block_tables.size());
    for (const auto& table : multi_block_tables) {
      params.multi_block_tables.emplace_back(
          safe_to(table, table.options().device(torch::kCPU), true));
    }
    params.mtp_shifted_token_ids = safe_to(mtp_shifted_token_ids, device, true);
    if (!params.embedding.linear_state_indices.defined() &&
        !params.embedding.linear_state_ids.empty()) {
      params.embedding.linear_state_indices =
          torch::tensor(params.embedding.linear_state_ids, torch::kInt)
              .to(device);
    }
#if defined(USE_MUSA)
    params.attn_metadata = attn_metadata;
#endif
    return params;
  }

  void clear_linear_attention_state() {
    embedding.linear_state_ids.clear();
    embedding.linear_state_indices = torch::Tensor();
    linear_state_cache_ops.clear();
    linear_state_validity_mask.clear();
  }

  int32_t get_q_seq_len(int32_t seq_idx) const {
#if defined(USE_NPU)
    CHECK_LT(seq_idx, static_cast<int32_t>(attention.host.q_seq_lens.size()));
    return attention.host.q_seq_lens[seq_idx];
#else
    CHECK_LT(seq_idx + 1,
             static_cast<int32_t>(attention.host.q_seq_lens.size()));
    return attention.host.q_seq_lens[seq_idx + 1] -
           attention.host.q_seq_lens[seq_idx];
#endif
  }

  bool synchronize_layer(int64_t layer_idx) const {
    if (parallel.layer_wise_load_synchronizer == nullptr) {
      return true;
    }
    CHECK_GE(layer_idx, 0);
    if (static_cast<uint64_t>(layer_idx) % parallel.layers_per_event == 0) {
      return parallel.layer_wise_load_synchronizer->synchronize_layer(
          layer_idx / parallel.layers_per_event);
    }
    return true;
  }

  bool synchronize_draft_layer() const {
    if (parallel.layer_wise_load_synchronizer == nullptr ||
        !parallel.draft_load_event_index.has_value()) {
      return true;
    }
    return parallel.layer_wise_load_synchronizer->synchronize_layer(
        static_cast<int64_t>(*parallel.draft_load_event_index));
  }

  bool record_layer(uint32_t layer_idx, const torch::Device& device) const {
#if defined(USE_MLU) || defined(USE_DCU)
    if (parallel.layer_synchronizer != nullptr) {
      return parallel.layer_synchronizer->record_current(layer_idx,
                                                         device.index());
    }
#else
    (void)layer_idx;
    (void)device;
#endif
    return true;
  }

  BatchInputMeta meta;
  VlmAttentionInput attention;
  ModelEmbeddingInput embedding;
  ParallelInput parallel;
  BlockCopyInput block_copy;
  MultiModalInput multimodal;
  ExpertInput expert;
  GraphInput graph;
  std::vector<torch::Tensor> multi_block_tables;
  torch::Tensor mtp_shifted_token_ids;
  std::vector<LinearStateCacheOp> linear_state_cache_ops;
  LinearStateValidityMask linear_state_validity_mask;
  bool is_spec_verify = false;
  bool prefill_without_cache = false;
  torch::Tensor num_accepted_tokens;
  MtpTopkStatePtr mtp_topk_state;
  std::vector<int64_t> num_accepted_tokens_host;
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
  std::shared_ptr<PythonAttentionMetadata> python_attention_metadata;
  bool enable_graph = false;
};

// Temporary synchronous adapter. The owner remains inaccessible until the
// projection restores all leaves, including mutations produced by execution.
class VlmLegacyExecutionProjection final {
 public:
  explicit VlmLegacyExecutionProjection(VlmModelParams& owner) : owner_(owner) {
    exchange_fields();
  }
  ~VlmLegacyExecutionProjection() { exchange_fields(); }
  VlmLegacyExecutionProjection(const VlmLegacyExecutionProjection&) = delete;
  VlmLegacyExecutionProjection& operator=(const VlmLegacyExecutionProjection&) =
      delete;

  ModelInputParams& params() { return params_; }

 private:
  void exchange_fields() {
    using std::swap;
    swap(owner_.meta, params_.meta);
    swap(owner_.attention.host, params_.attention.host);
    swap(owner_.attention.device, params_.attention.device);
    swap(owner_.attention.attention_host_buffer,
         params_.attention.attention_host_buffer);
    swap(owner_.attention.attention_device_buffer,
         params_.attention.attention_device_buffer);
    swap(owner_.attention.attention_buffer_bytes,
         params_.attention.attention_buffer_bytes);
    swap(owner_.attention.attention_buffer_capacity,
         params_.attention.attention_buffer_capacity);
    swap(owner_.attention.attention_buffer_owner,
         params_.attention.attention_buffer_owner);
    swap(owner_.embedding, params_.embedding);
    swap(owner_.parallel, params_.parallel);
    swap(owner_.block_copy, params_.block_copy);
    swap(owner_.multimodal, params_.multimodal);
    swap(owner_.expert, params_.expert);
    swap(owner_.graph, params_.graph);
    swap(owner_.multi_block_tables, params_.multi_block_tables);
    swap(owner_.mtp_shifted_token_ids, params_.mtp_shifted_token_ids);
    swap(owner_.linear_state_cache_ops, params_.linear_state_cache_ops);
    swap(owner_.linear_state_validity_mask, params_.linear_state_validity_mask);
    swap(owner_.is_spec_verify, params_.is_spec_verify);
    swap(owner_.prefill_without_cache, params_.prefill_without_cache);
    swap(owner_.num_accepted_tokens, params_.num_accepted_tokens);
    swap(owner_.mtp_topk_state, params_.mtp_topk_state);
    swap(owner_.num_accepted_tokens_host, params_.num_accepted_tokens_host);
    swap(owner_.attn_metadata, params_.attn_metadata);
    swap(owner_.python_attention_metadata, params_.python_attention_metadata);
    swap(owner_.enable_graph, params_.enable_graph);
  }

  VlmModelParams& owner_;
  ModelInputParams params_;
};

}  // namespace xllm
