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

#include "core/framework/batch/batch_assembly_plan.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/common/metrics.h"

namespace xllm {

BatchAssemblyPlan::BatchAssemblyPlan(
    int32_t dp_size,
    const std::vector<std::shared_ptr<Request>>& requests,
    const std::vector<Sequence*>& sequences,
    const std::vector<size_t>& budgets,
    bool group_input,
    std::vector<std::vector<BlockTransferInfo>>* swap_infos)
    : dp_size_(dp_size),
      requests_(requests),
      sequences_(sequences),
      budgets_(budgets),
      group_input_(group_input),
      retain_request_groups_(group_input),
      swap_infos_(swap_infos) {
  CHECK_GT(dp_size_, 0);
  sequence_counts_.resize(dp_size_);
  group_counts_.resize(dp_size_);
  CHECK_EQ(sequences_.size(), budgets_.size())
      << "Each scheduled sequence requires one token budget";
  if (swap_infos_ != nullptr) {
    CHECK_EQ(swap_infos_->size(), static_cast<size_t>(dp_size_));
  }

  // Validate and count before assembling or consuming pending block transfers.
  for (size_t i = 0; i < sequences_.size(); ++i) {
    auto* sequence = sequences_[i];
    CHECK(sequence != nullptr);
    CHECK(!sequence->finished());
    CHECK_GE(sequence->dp_rank(), 0);
    CHECK_LT(sequence->dp_rank(), dp_size_);
    // OneRec can schedule decoder embeddings without any decoder tokens.
    if (!group_input_) {
      CHECK_GT(budgets_[i], 0);
      CHECK_LE(budgets_[i], std::numeric_limits<uint32_t>::max());
    }
    ++sequence_counts_[sequence->dp_rank()];
    const size_t cached_tokens = sequence->kv_state().kv_cache_tokens_num();
    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() > cached_tokens
            ? sequence->num_prompt_tokens() - cached_tokens
            : 0;
    const size_t prompt_tokens = std::min(remaining_prompt_tokens, budgets_[i]);
    num_prompt_tokens_ += prompt_tokens;
    num_generated_tokens_ += budgets_[i] - prompt_tokens;
  }
  for (const auto& request : requests_) {
    CHECK(request != nullptr);
    retain_request_groups_ |= request->check_beam_search();
  }
  if (retain_request_groups_) {
    for (const auto& request : requests_) {
      auto* group = request->sequence_group();
      CHECK(group != nullptr);
      CHECK(!group->sequences().empty());
      CHECK_GE(group->dp_rank(), 0);
      CHECK_LT(group->dp_rank(), dp_size_);
      ++group_counts_[group->dp_rank()];
    }
  }
}

void BatchAssemblyPlan::populate(std::vector<Batch>& batches) const {
  CHECK_EQ(batches.size(), static_cast<size_t>(dp_size_));
  for (int32_t rank = 0; rank < dp_size_; ++rank) {
    batches[rank].reserve(group_input_ ? 0 : sequence_counts_[rank],
                          group_counts_[rank]);
  }
  for (size_t i = 0; i < sequences_.size(); ++i) {
    auto& batch = batches[sequences_[i]->dp_rank()];
    if (group_input_) {
      batch.set_batch_id();
      continue;
    }
    batch.add(sequences_[i], static_cast<uint32_t>(budgets_[i]));
  }
  if (retain_request_groups_) {
    for (const auto& request : requests_) {
      auto* group = request->sequence_group();
      batches[group->dp_rank()].add(group);
    }
  }
  if (swap_infos_ != nullptr) {
    for (int32_t rank = 0; rank < dp_size_; ++rank) {
      if (batches[rank].empty()) {
        continue;
      }
      batches[rank].set_swap_block_transfer_infos(
          std::move((*swap_infos_)[rank]));
      (*swap_infos_)[rank].clear();
    }
  }

  COUNTER_ADD(num_processing_tokens_total_prompt, num_prompt_tokens_);
  COUNTER_ADD(num_processing_tokens_total_generated, num_generated_tokens_);
  if (!sequences_.empty()) {
    HISTOGRAM_OBSERVE(
        num_prompt_tokens_per_request,
        static_cast<int64_t>(num_prompt_tokens_ / sequences_.size()));
    HISTOGRAM_OBSERVE(
        num_generated_tokens_per_request,
        static_cast<int64_t>(num_generated_tokens_ / sequences_.size()));
  }
}

}  // namespace xllm
