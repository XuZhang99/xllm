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

#include "core/runtime/schedule_overlap_state.h"

#include <utility>

#include "core/runtime/json_object_output_rows.h"

namespace xllm {

bool ScheduleOverlapState::publish(std::optional<ForwardOutput> output,
                                   std::vector<std::string> request_ids,
                                   std::vector<std::string> sample_sequence_ids,
                                   bool retain_tokenless_output) {
  std::unique_lock<std::mutex> lock(mutex_);
  condition_.wait(lock, [this] { return !pending_ || shutting_down_; });
  if (shutting_down_) {
    return false;
  }
  update_prior_output(std::move(output),
                      std::move(request_ids),
                      std::move(sample_sequence_ids),
                      retain_tokenless_output);
  pending_ = true;
  condition_.notify_all();
  return true;
}

void ScheduleOverlapState::update(std::optional<ForwardOutput> output,
                                  std::vector<std::string> request_ids,
                                  std::vector<std::string> sample_sequence_ids,
                                  bool retain_tokenless_output) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (shutting_down_) {
    return;
  }
  update_prior_output(std::move(output),
                      std::move(request_ids),
                      std::move(sample_sequence_ids),
                      retain_tokenless_output);
}

ForwardOutput ScheduleOverlapState::consume(bool retain_tokenless_output) {
  std::unique_lock<std::mutex> lock(mutex_);
  condition_.wait(lock, [this] { return pending_ || shutting_down_; });

  ForwardOutput output;
  if (pending_ && (has_tokens_ || retain_tokenless_output ||
                   !prior_output_.json_object_errors.empty())) {
    output = prior_output_;
  }
  pending_ = false;
  condition_.notify_all();
  return output;
}

void ScheduleOverlapState::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutting_down_ = true;
  }
  condition_.notify_all();
}

bool ScheduleOverlapState::has_request_overlap(
    const std::vector<std::string>& request_ids) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return detail::has_request_id_overlap(request_ids, request_ids_);
}

void ScheduleOverlapState::update_prior_output(
    std::optional<ForwardOutput> output,
    std::vector<std::string> request_ids,
    std::vector<std::string> sample_sequence_ids,
    bool retain_tokenless_output) {
  prior_output_ =
      output.has_value() ? std::move(output.value()) : ForwardOutput();
  request_ids_ = std::move(request_ids);
  sample_sequence_ids_ = std::move(sample_sequence_ids);
  has_tokens_ = prior_output_.sample_output.next_tokens.defined();
  if (!output.has_value() || (!has_tokens_ && !retain_tokenless_output &&
                              prior_output_.json_object_errors.empty())) {
    request_ids_.clear();
    sample_sequence_ids_.clear();
  }
}

}  // namespace xllm
