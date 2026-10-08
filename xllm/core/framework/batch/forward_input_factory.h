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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/runtime/forward_params.h"

namespace xllm {

class BatchGroup;
class ThreadPool;
struct ModelArgs;

struct ForwardInputFactoryOptions {
  uint32_t dp_size = 1;
  uint32_t cp_size = 1;
  int64_t max_tokens_per_batch = 0;
};

class PreparedLlmInputGroup final {
 public:
  PreparedLlmInputGroup() = default;
  PreparedLlmInputGroup(const PreparedLlmInputGroup&) = delete;
  PreparedLlmInputGroup& operator=(const PreparedLlmInputGroup&) = delete;
  PreparedLlmInputGroup(PreparedLlmInputGroup&&) noexcept = default;
  PreparedLlmInputGroup& operator=(PreparedLlmInputGroup&&) noexcept = default;

  std::vector<LlmForwardInput> inputs;
  std::vector<int32_t> dp_token_counts;
  // Facts from rank-local inputs before empty-rank forward type inheritance.
  bool has_non_empty_batch = false;
  bool all_non_empty_batches_are_decode = true;
  bool is_graph_warmup = false;
};

// One engine-owned instance retains rank identities and reuses preparation
// threads across scheduler steps. Calls must be serialized by its owner.
class ForwardInputFactory final {
 public:
  explicit ForwardInputFactory(ForwardInputFactoryOptions options);
  ~ForwardInputFactory();
  ForwardInputFactory(const ForwardInputFactory&) = delete;
  ForwardInputFactory& operator=(const ForwardInputFactory&) = delete;
  ForwardInputFactory(ForwardInputFactory&&) noexcept = default;
  ForwardInputFactory& operator=(ForwardInputFactory&&) noexcept = default;

  // Advances Batch/Sequence preparation state; prepare each group only once.
  PreparedLlmInputGroup create_inputs(BatchGroup& batches,
                                      const ModelArgs& model_args,
                                      bool enable_graph);

 private:
  ForwardInputFactoryOptions options_;
  std::vector<std::vector<int32_t>> dp_batch_embedding_ids_;
  std::vector<std::vector<std::string>> dp_batch_request_ids_;
  std::vector<uint64_t> dp_batch_generations_;
  std::unique_ptr<ThreadPool> threadpool_;
};

}  // namespace xllm
