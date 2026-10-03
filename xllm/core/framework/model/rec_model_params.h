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

#include "core/framework/model/model_input_params.h"

namespace xllm {

class RecModelParams final {
 public:
  RecModelParams to(const torch::Device& device) const {
    RecModelParams params;
    params.meta = meta;
    params.attention = attention.to(device);
    params.embedding = embedding.to(device);
    params.parallel = parallel.to(device);
    params.block_copy = block_copy.to(device);
    params.multimodal = multimodal.to(device);
    params.expert = expert.to(device);
    params.graph = graph.to(device);
    params.linear_state_cache_ops = linear_state_cache_ops;
    params.linear_state_validity_mask = linear_state_validity_mask;
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
    if (const auto* xattention = onerec_xattention_params()) {
      params.rec_params = xattention->to(device);
    } else if (const auto* onerec = onerec_params()) {
      params.rec_params = onerec->to(device);
    } else if (const auto* llmrec = llmrec_params()) {
      params.rec_params = llmrec->to(device);
    }
#if defined(USE_MUSA)
    params.attn_metadata = attn_metadata;
#endif
    return params;
  }

  const OneRecModelInputParams* onerec_params() const {
    if (const auto* params = std::get_if<OneRecModelInputParams>(&rec_params)) {
      return params;
    }
    return std::get_if<OneRecXAttentionParams>(&rec_params);
  }

  bool has_onerec_params() const { return onerec_params() != nullptr; }

  OneRecModelInputParams& mutable_onerec_params() {
    if (auto* params = std::get_if<OneRecModelInputParams>(&rec_params)) {
      return *params;
    }
    if (auto* params = std::get_if<OneRecXAttentionParams>(&rec_params)) {
      return *params;
    }
    return rec_params.emplace<OneRecModelInputParams>();
  }

  const OneRecXAttentionParams* onerec_xattention_params() const {
    return std::get_if<OneRecXAttentionParams>(&rec_params);
  }

  bool has_onerec_xattention_params() const {
    return onerec_xattention_params() != nullptr;
  }

  OneRecXAttentionParams& mutable_onerec_xattention_params() {
    if (auto* params = std::get_if<OneRecXAttentionParams>(&rec_params)) {
      return *params;
    }
    return rec_params.emplace<OneRecXAttentionParams>();
  }

  const LlmRecMultiRoundParams* llmrec_params() const {
    return std::get_if<LlmRecMultiRoundParams>(&rec_params);
  }

  bool has_llmrec_params() const { return llmrec_params() != nullptr; }

  LlmRecMultiRoundParams& mutable_llmrec_params() {
    if (auto* params = std::get_if<LlmRecMultiRoundParams>(&rec_params)) {
      return *params;
    }
    return rec_params.emplace<LlmRecMultiRoundParams>();
  }

  BatchInputMeta meta;
  AttentionInput attention;
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
  RecModelInputParams rec_params;
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
  std::shared_ptr<PythonAttentionMetadata> python_attention_metadata;
  bool enable_graph = false;
};

// Temporary synchronous executor adapter. The owner must not be accessed until
// destruction restores its fields, including executor-produced metadata.
class RecLegacyExecutionProjection final {
 public:
  explicit RecLegacyExecutionProjection(RecModelParams& owner) : owner_(owner) {
    exchange_fields();
  }

  ~RecLegacyExecutionProjection() { exchange_fields(); }

  RecLegacyExecutionProjection(const RecLegacyExecutionProjection&) = delete;
  RecLegacyExecutionProjection& operator=(const RecLegacyExecutionProjection&) =
      delete;

  ModelInputParams& params() { return params_; }

 private:
  void exchange_fields() {
    using std::swap;
    swap(owner_.meta, params_.meta);
    swap(owner_.attention, params_.attention);
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
    swap(owner_.rec_params, params_.rec_params);
    swap(owner_.attn_metadata, params_.attn_metadata);
    swap(owner_.python_attention_metadata, params_.python_attention_metadata);
    swap(owner_.enable_graph, params_.enable_graph);
  }

  RecModelParams& owner_;
  ModelInputParams params_;
};

}  // namespace xllm
