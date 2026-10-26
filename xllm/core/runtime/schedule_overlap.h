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

#include <optional>
#include <string>
#include <vector>

#include "core/runtime/schedule_overlap_state.h"

namespace xllm {

class Device;
class Stream;
class VlmForwardInput;
class VlmModelParams;

// Executes the one-step overlap protocol on the worker's FIFO execution thread.
// Owns continuation provenance, stream dependencies and host publication; model
// execution and model-specific decode rebuilding remain worker capabilities.
class ScheduleOverlap final {
 public:
  // Non-owning model capabilities. Execution resources are borrowed separately
  // for each step because workers may replace their compute stream at load.
  class ModelExecution {
   public:
    virtual std::optional<ForwardOutput> step_for_schedule_overlap(
        const LlmForwardInput& input) = 0;
    virtual std::optional<ForwardOutput> step_for_schedule_overlap(
        const VlmForwardInput& input) = 0;
    virtual bool prepare_model_decode_continuation(
        LlmForwardInput& input,
        const ForwardOutput& prior_output) = 0;
    virtual bool prepare_model_decode_continuation(
        VlmForwardInput& input,
        const ForwardOutput& prior_output) = 0;
    virtual bool can_prepare_npu_graph_decode_input(
        const LlmModelParams& input_params) const = 0;
    virtual bool can_prepare_npu_graph_decode_input(
        const VlmModelParams& input_params) const = 0;
    virtual void observe_execution_memory(bool is_warmup) = 0;

   protected:
    virtual ~ModelExecution() = default;
  };

  ScheduleOverlap(bool is_driver,
                  bool enable_eplb,
                  bool retire_unconsumed_inputs);

  std::optional<ForwardOutput> execute(ModelExecution& executor,
                                       LlmForwardInput& input,
                                       Device& device,
                                       Stream& compute_stream);
  std::optional<ForwardOutput> execute(ModelExecution& executor,
                                       VlmForwardInput& input,
                                       Device& device,
                                       Stream& compute_stream);

  // May be called by the host consumer concurrently with execute().
  ForwardOutput consume();
  void shutdown();

 private:
  template <typename Input>
  std::optional<ForwardOutput> execute_impl(ModelExecution& executor,
                                            Input& input,
                                            Device& device,
                                            Stream& compute_stream);
  template <typename Input>
  void prepare_continuation(ModelExecution& executor,
                            Input& input,
                            Device& device,
                            Stream& compute_stream);
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
