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

#include <c10/util/FunctionRef.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/runtime/schedule_overlap_state.h"

namespace xllm {

class Device;
class Stream;
class VlmForwardInput;

// Owns continuation provenance, stream dependencies and host publication on the
// worker's FIFO execution thread. The worker executes the model between
// prepare_continuation() and complete_step().
class ScheduleOverlap final {
 public:
  enum class DecodeContinuation : int8_t {
    MODEL_MANAGED = 0,
    REPLACE_TOKENS,
    REPLACE_GRAPH_TOKENS,
  };
  using PrepareModelContinuation =
      c10::function_ref<DecodeContinuation(const ForwardOutput&)>;

  ScheduleOverlap(bool is_driver,
                  bool enable_eplb,
                  bool retire_unconsumed_inputs);

  // The callback is borrowed only for this call. It runs on compute_stream
  // after continuation dependencies are ready, and only when reusing tokens.
  void prepare_continuation(LlmForwardInput& input,
                            Device& device,
                            Stream& compute_stream,
                            PrepareModelContinuation prepare_model);
  void prepare_continuation(VlmForwardInput& input,
                            Device& device,
                            Stream& compute_stream,
                            PrepareModelContinuation prepare_model);

  void complete_step(const LlmForwardInput& input,
                     Stream& compute_stream,
                     std::optional<ForwardOutput>& output);
  void complete_step(const VlmForwardInput& input,
                     Stream& compute_stream,
                     std::optional<ForwardOutput>& output);

  // May be called by the host consumer concurrently with the execution thread.
  ForwardOutput consume();
  void shutdown();

 private:
  template <typename Input>
  void prepare_continuation_impl(Input& input,
                                 Device& device,
                                 Stream& compute_stream,
                                 PrepareModelContinuation prepare_model);
  template <typename Input>
  void complete_step_impl(const Input& input,
                          Stream& compute_stream,
                          std::optional<ForwardOutput>& output);
  template <typename Input>
  static void replace_tokens(Input& input,
                             const ForwardOutput& prior_output,
                             bool use_graph_decode_input);

  void record_output(Stream& compute_stream,
                     const std::optional<ForwardOutput>& output,
                     const std::vector<std::string>& request_ids,
                     const std::vector<std::string>& sample_sequence_ids);
  void retire_prior_inputs() const;

  ScheduleOverlapState state_;
  const bool retire_unconsumed_inputs_;
};

}  // namespace xllm
