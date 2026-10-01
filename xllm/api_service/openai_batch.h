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

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "core/framework/request/request_output.h"

namespace xllm::api_service {

// Serializes per-prompt callbacks and gives the HTTP request one terminal
// event.
class OpenAIBatch final {
 public:
  OpenAIBatch(size_t size, size_t choices_per_prompt, bool streaming)
      : remaining_(size),
        choices_per_prompt_(choices_per_prompt),
        streaming_(streaming),
        finished_(size, false),
        usages_(size) {}

  bool accept(size_t index, RequestOutput output, const OutputCallback& send) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || finished_[index]) {
      return false;
    }
    if (output.status.has_value() && !output.status->ok()) {
      closed_ = true;
      send(std::move(output));
      return false;
    }
    if (output.usage.has_value()) {
      const auto& previous = usages_[index];
      const auto& current = output.usage.value();
      total_.num_prompt_tokens +=
          current.num_prompt_tokens - previous.num_prompt_tokens;
      total_.num_generated_tokens +=
          current.num_generated_tokens - previous.num_generated_tokens;
      total_.num_total_tokens +=
          current.num_total_tokens - previous.num_total_tokens;
      total_.num_cached_tokens +=
          current.num_cached_tokens - previous.num_cached_tokens;
      usages_[index] = current;
    }
    for (auto& sequence : output.outputs) {
      sequence.index += index * choices_per_prompt_;
    }
    const bool terminal = output.finished || output.cancelled;
    if (terminal) {
      finished_[index] = true;
      --remaining_;
    }
    output.finished = remaining_ == 0;
    output.cancelled = false;
    output.usage = total_;
    if (!streaming_) {
      for (auto& sequence : output.outputs) {
        outputs_.emplace_back(std::move(sequence));
      }
      if (!output.finished) {
        return true;
      }
      std::sort(outputs_.begin(),
                outputs_.end(),
                [](const auto& left, const auto& right) {
                  return left.index < right.index;
                });
      output.outputs = std::move(outputs_);
    }
    closed_ = output.finished;
    if (!send(std::move(output))) {
      closed_ = true;
    }
    return !closed_;
  }

 private:
  std::mutex mutex_;
  size_t remaining_;
  size_t choices_per_prompt_;
  bool streaming_;
  bool closed_ = false;
  std::vector<bool> finished_;
  std::vector<Usage> usages_;
  Usage total_;
  std::vector<SequenceOutput> outputs_;
};

}  // namespace xllm::api_service
