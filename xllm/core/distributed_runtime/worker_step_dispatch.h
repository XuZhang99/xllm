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

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "core/runtime/forward_input_encoding.h"
#include "core/runtime/worker_client.h"

namespace xllm {

using WorkerStepResults =
    std::vector<folly::Try<std::optional<RawForwardOutput>>>;

// Inputs and their borrowed tensor storage stay alive until every RPC
// completes. Engines retain their own policy for absent outputs, including
// empty VLM ranks.
template <typename Input>
WorkerStepResults dispatch_worker_step(
    const std::vector<std::shared_ptr<WorkerClient>>& workers,
    const std::vector<Input>& inputs,
    uint32_t workers_per_dp_rank) {
  CHECK_GT(workers_per_dp_rank, 0);
  CHECK_EQ(workers.size(), inputs.size() * workers_per_dp_rank);

  std::vector<std::shared_ptr<ForwardInputEncoding>> encodings;
  encodings.reserve(inputs.size());
  for (size_t dp_rank = 0; dp_rank < inputs.size(); ++dp_rank) {
    encodings.emplace_back(workers_per_dp_rank > 1
                               ? std::make_shared<ForwardInputEncoding>()
                               : nullptr);
  }

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(workers.size());
  // Model-side CP consumes the full DP input, just like the other TP workers.
  for (size_t worker_rank = 0; worker_rank < workers.size(); ++worker_rank) {
    const size_t dp_rank = worker_rank / workers_per_dp_rank;
    futures.emplace_back(workers[worker_rank]->step_remote_async(
        inputs[dp_rank], encodings[dp_rank]));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    CHECK(result.hasValue()) << "Failed to execute model, worker RPC failed";
  }
  return results;
}

// A full DP-local TP/CP width selects drivers; stride 1 retains every EPLB
// shard.
template <typename GetResult>
WorkerStepResults collect_worker_step_results(
    const std::vector<std::shared_ptr<WorkerClient>>& workers,
    uint32_t stride,
    GetResult get_result) {
  // Only the driver engine owns worker clients in multi-node serving.
  if (workers.empty()) {
    return {};
  }
  CHECK_GT(stride, 0);
  CHECK_EQ(workers.size() % stride, 0);

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(workers.size() / stride);
  for (size_t worker_rank = 0; worker_rank < workers.size();
       worker_rank += stride) {
    futures.emplace_back(std::invoke(get_result, *workers[worker_rank]));
  }
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    CHECK(result.hasValue() && result.value().has_value())
        << "Failed to get last step results, result has no value";
  }
  return results;
}

}  // namespace xllm
