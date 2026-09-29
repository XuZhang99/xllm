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

#include "core/framework/batch/sampling_input_builder.h"

#include <glog/logging.h>

#include <iterator>
#include <limits>

#include "core/util/utils.h"

namespace xllm {

void SamplingInputBuilder::reserve(size_t rows) {
  params_.reserve(rows);
  selected_token_indices_.reserve(rows);
  sample_indices_.reserve(rows);
  token_ids_.reserve(rows);
  token_counts_.reserve(rows);
  token_lengths_.reserve(rows);
}

void SamplingInputBuilder::append(const RequestSamplingParam* params,
                                  int32_t token_index,
                                  const TokenCounts* counts,
                                  const TokenCounts* excluded_counts,
                                  bool sample) {
  CHECK(params != nullptr);
  CHECK_GE(token_index, 0);
  CHECK_LT(size(), std::numeric_limits<int32_t>::max());
  if (sample) {
    sample_indices_.emplace_back(static_cast<int32_t>(size()));
  }
  params_.emplace_back(params);
  selected_token_indices_.emplace_back(token_index);
  auto& ids = token_ids_.emplace_back();
  auto& frequencies = token_counts_.emplace_back();
  if (counts != nullptr) {
    ids.reserve(counts->size());
    frequencies.reserve(counts->size());
    for (const auto& [token, count] : *counts) {
      CHECK_GE(count, 0);
      int32_t excluded = 0;
      if (excluded_counts != nullptr) {
        const auto it = excluded_counts->find(token);
        excluded = it == excluded_counts->end() ? 0 : it->second;
        if (count <= excluded) {
          continue;
        }
      }
      ids.emplace_back(token);
      frequencies.emplace_back(count - excluded);
    }
  }
  token_lengths_.emplace_back(static_cast<int32_t>(ids.size()));
}

void SamplingInputBuilder::merge(SamplingInputBuilder other,
                                 int32_t token_offset) {
  CHECK_GE(token_offset, 0);
  CHECK_LE(size() + other.size(), std::numeric_limits<int32_t>::max());
  const int32_t row_offset = static_cast<int32_t>(size());
  reserve(size() + other.size());
  for (int32_t index : other.selected_token_indices_) {
    CHECK_LE(index, std::numeric_limits<int32_t>::max() - token_offset);
    selected_token_indices_.emplace_back(index + token_offset);
  }
  for (int32_t index : other.sample_indices_) {
    sample_indices_.emplace_back(index + row_offset);
  }
  params_.insert(params_.end(), other.params_.begin(), other.params_.end());
  token_ids_.insert(token_ids_.end(),
                    std::make_move_iterator(other.token_ids_.begin()),
                    std::make_move_iterator(other.token_ids_.end()));
  token_counts_.insert(token_counts_.end(),
                       std::make_move_iterator(other.token_counts_.begin()),
                       std::make_move_iterator(other.token_counts_.end()));
  token_lengths_.insert(token_lengths_.end(),
                        other.token_lengths_.begin(),
                        other.token_lengths_.end());
}

SamplingParameters SamplingInputBuilder::build() {
  SamplingParameters result;
  if (empty()) {
    return result;
  }
  util::pad_2d_vector<int64_t>(token_ids_, /*pad_value=*/0);
  util::pad_2d_vector(token_counts_, /*pad_value=*/0);
  result.init(params_,
              selected_token_indices_,
              sample_indices_,
              token_ids_,
              token_counts_,
              token_lengths_);
  return result;
}

}  // namespace xllm
