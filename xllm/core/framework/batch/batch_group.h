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
#include <vector>

#include "core/framework/batch/batch.h"

namespace xllm {

// A DP batch aggregate. Each element is the rank-local Batch for the same
// scheduler step, and its index is the corresponding DP rank.
class BatchGroup final {
 public:
  using Container = std::vector<Batch>;
  using iterator = Container::iterator;
  using const_iterator = Container::const_iterator;

  BatchGroup() = default;
  explicit BatchGroup(size_t dp_size);
  BatchGroup(size_t dp_size, BatchDomain domain, BatchInputType input_type);

  BatchGroup(const BatchGroup&) = delete;
  BatchGroup& operator=(const BatchGroup&) = delete;
  BatchGroup(BatchGroup&&) noexcept = default;
  BatchGroup& operator=(BatchGroup&&) noexcept = default;

  size_t size() const { return batches_.size(); }
  bool empty() const { return batches_.empty(); }

  Batch& front() { return batches_.front(); }
  const Batch& front() const { return batches_.front(); }
  Batch& back() { return batches_.back(); }
  const Batch& back() const { return batches_.back(); }

  Batch& operator[](size_t dp_rank) { return batches_[dp_rank]; }
  const Batch& operator[](size_t dp_rank) const { return batches_[dp_rank]; }
  Batch& at(size_t dp_rank) { return batches_.at(dp_rank); }
  const Batch& at(size_t dp_rank) const { return batches_.at(dp_rank); }

  iterator begin() { return batches_.begin(); }
  iterator end() { return batches_.end(); }
  const_iterator begin() const { return batches_.begin(); }
  const_iterator end() const { return batches_.end(); }
  const_iterator cbegin() const { return batches_.cbegin(); }
  const_iterator cend() const { return batches_.cend(); }

 private:
  Container batches_;
};

}  // namespace xllm
