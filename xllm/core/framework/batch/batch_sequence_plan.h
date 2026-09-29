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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm {

class Sequence;

struct ScheduledSequence {
  Sequence* sequence;
  uint32_t token_budget;
};

// Keeps the scheduler's sequence order and token budgets aligned. Builders get
// read-only column views; all mutations go through this object.
class BatchSequencePlan final {
 public:
  void reserve(size_t count);
  void add(Sequence* sequence, uint32_t token_budget);
  void reorder(const std::vector<size_t>& source_indices);

  size_t size() const { return sequences_.size(); }
  bool empty() const { return sequences_.empty(); }
  ScheduledSequence operator[](size_t index) const;
  const std::vector<Sequence*>& sequences() const { return sequences_; }
  const std::vector<uint32_t>& budgets() const { return budgets_; }

 private:
  std::vector<Sequence*> sequences_;
  std::vector<uint32_t> budgets_;
};

}  // namespace xllm
