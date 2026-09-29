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

#include "core/framework/batch/batch_sequence_plan.h"

#include <glog/logging.h>

#include <utility>

namespace xllm {

void BatchSequencePlan::reserve(size_t count) {
  sequences_.reserve(count);
  budgets_.reserve(count);
}

void BatchSequencePlan::add(Sequence* sequence, uint32_t token_budget) {
  CHECK(sequence != nullptr);
  CHECK_GT(token_budget, 0);
  sequences_.emplace_back(sequence);
  budgets_.emplace_back(token_budget);
}

ScheduledSequence BatchSequencePlan::operator[](size_t index) const {
  CHECK_LT(index, size());
  return {sequences_[index], budgets_[index]};
}

void BatchSequencePlan::reorder(const std::vector<size_t>& source_indices) {
  CHECK_EQ(source_indices.size(), size());
  std::vector<bool> visited(size(), false);
  BatchSequencePlan reordered;
  reordered.reserve(size());
  for (size_t index : source_indices) {
    CHECK_LT(index, size());
    CHECK(!visited[index]) << "Batch sequence order must be a permutation";
    visited[index] = true;
    const auto entry = (*this)[index];
    reordered.add(entry.sequence, entry.token_budget);
  }
  *this = std::move(reordered);
}

}  // namespace xllm
