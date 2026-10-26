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

#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "core/runtime/forward_params.h"

namespace xllm {

// Owns the one-step device continuation and the one-slot host publication
// protocol used by schedule overlap. The producer thread updates the prior
// output, while the host consumer only clears the publication flag. A
// consumed result therefore remains available for the next device step.
class ScheduleOverlapState final {
 public:
  ScheduleOverlapState() = default;
  ~ScheduleOverlapState() = default;

  ScheduleOverlapState(const ScheduleOverlapState&) = delete;
  ScheduleOverlapState& operator=(const ScheduleOverlapState&) = delete;

  // Publishes a result for the host consumer. It waits for the preceding
  // publication to be consumed and returns false once shutdown has started.
  bool publish(std::optional<ForwardOutput> output,
               std::vector<std::string> request_ids,
               std::vector<std::string> sample_sequence_ids,
               bool retain_tokenless_output);

  // Updates only the device continuation cache. This path is used by ranks
  // whose output is not read by the host controller. Returns false after
  // shutdown; the caller must drain device work before releasing the output.
  bool update(std::optional<ForwardOutput> output,
              std::vector<std::string> request_ids,
              std::vector<std::string> sample_sequence_ids,
              bool retain_tokenless_output);

  // Waits for a publication or shutdown, consumes the publication flag, and
  // returns a copy when the result is relevant to the caller. The cached prior
  // output is intentionally retained for the next overlapped device step.
  ForwardOutput consume(bool retain_tokenless_output);

  // Wakes blocked producer and consumer calls. Cached device state is kept
  // intact so destruction can drain already submitted work safely.
  void shutdown();

  // These accessors are for the single producer thread that executes the next
  // overlapped step. They borrow state owned by this object.
  const ForwardOutput& prior_output() const { return prior_output_; }
  const std::vector<std::string>& request_ids() const { return request_ids_; }
  const std::vector<std::string>& sample_sequence_ids() const {
    return sample_sequence_ids_;
  }
  bool has_tokens() const { return has_tokens_; }

  bool has_request_overlap(const std::vector<std::string>& request_ids) const;

 private:
  void update_prior_output(std::optional<ForwardOutput> output,
                           std::vector<std::string> request_ids,
                           std::vector<std::string> sample_sequence_ids,
                           bool retain_tokenless_output);

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  ForwardOutput prior_output_;
  std::vector<std::string> request_ids_;
  std::vector<std::string> sample_sequence_ids_;
  bool has_tokens_ = false;
  bool pending_ = false;
  bool shutting_down_ = false;
};

}  // namespace xllm
