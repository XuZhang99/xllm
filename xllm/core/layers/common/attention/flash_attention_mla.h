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

#include <tuple>

#include "attention_metadata.h"
#include "framework/kv_cache/kv_cache.h"

namespace xllm {
namespace layer {

class AttentionMLAImpl : public torch::nn::Module {
 public:
  AttentionMLAImpl() = default;

  AttentionMLAImpl(int64_t num_heads,
                   int64_t head_size,
                   int64_t num_kv_heads,
                   int64_t v_head_dim,
                   int64_t sliding_window,
                   float scale,
                   bool use_fused_mla_qkv,
                   bool enable_lighting_indexer);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache);

  void prefill_forward(torch::Tensor& query,
                       torch::Tensor& key,
                       torch::Tensor& value,
                       torch::Tensor& output,
                       const torch::Tensor& k_cache,
                       const std::optional<torch::Tensor>& v_cache,
                       const AttentionMetadata& attn_metadata);

  void decode_forward(torch::Tensor& query,
                      torch::Tensor& output,
                      const torch::Tensor& k_cache,
                      const std::optional<torch::Tensor>& v_cache,
                      const AttentionMetadata& attn_metadata);

 private:
  int64_t num_heads_;
  int64_t head_size_;
  float scale_;
  int64_t num_kv_heads_;
  int64_t v_head_dim_;
  bool use_fused_mla_qkv_;
  bool enable_lighting_indexer_;
  int64_t sliding_window_;
};

TORCH_MODULE(AttentionMLA);

}  // namespace layer
}  // namespace xllm
