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

#include "core/runtime/schedule_overlap.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "core/platform/device.h"
#include "core/runtime/json_object_decode_continuation.h"
#include "core/runtime/vlm_forward_params.h"
#if defined(USE_NPU)
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#elif defined(USE_MUSA)
#include "core/kernels/musa/musa_ops_api.h"
#endif

namespace xllm {

ScheduleOverlap::ScheduleOverlap(bool is_driver,
                                 bool enable_eplb,
                                 bool retire_unconsumed_inputs)
    : state_(/*publish_to_host=*/is_driver || enable_eplb,
             /*retain_tokenless_output=*/enable_eplb),
      retire_unconsumed_inputs_(!is_driver && !enable_eplb &&
                                retire_unconsumed_inputs) {}

template <typename Input>
void ScheduleOverlap::prepare_continuation_impl(
    Input& input,
    Device& device,
    Stream& compute_stream,
    PrepareModelContinuation prepare_model) {
  if (input.token_ids.numel() == 0 ||
      !input.input_params.meta.batch_forward_type.has_decode()) {
    return;
  }

  const bool reuse_tokens =
      state_.has_tokens() &&
      (input.json_object_states.empty() ||
       state_.has_request_overlap(input.input_params.embedding.request_ids));
  // Only the continuation runs under this guard. Model execution retains its
  // own stream selection, including composite speculative workers.
  auto stream_guard = compute_stream.set_stream_guard();
  CHECK(compute_stream.wait_event(input.runtime.metadata_ready_event))
      << "failed to wait input metadata before overlap continuation";
  const ForwardOutput empty_output;
  const ForwardOutput& prior_output =
      state_.has_tokens() ? state_.prior_output() : empty_output;
  if (reuse_tokens) {
    CHECK(compute_stream.wait_event(prior_output.ready_event))
        << "failed to wait last step output before overlap continuation";
  }
  JsonObjectDecodeContinuation::prepare(input,
                                        prior_output,
                                        state_.sample_sequence_ids(),
                                        device,
                                        compute_stream);
  if (reuse_tokens) {
    const DecodeContinuation continuation = prepare_model(prior_output);
    if (continuation != DecodeContinuation::MODEL_MANAGED) {
      replace_tokens(input,
                     prior_output,
                     continuation == DecodeContinuation::REPLACE_GRAPH_TOKENS);
    }
  }
}

template <typename Input>
void ScheduleOverlap::complete_step_impl(const Input& input,
                                         Stream& compute_stream,
                                         std::optional<ForwardOutput>& output) {
  if (output.has_value()) {
#if defined(USE_NPU)
    // No token consumer will wait for this output. Retire its asynchronous
    // input owners before either a result copy or the continuation is dropped.
    if (!output->sample_output.next_tokens.defined() &&
        output->ready_event != nullptr && !output->retained_inputs.empty()) {
      CHECK(output->ready_event->synchronize())
          << "failed to retire asynchronous output without tokens";
    }
#endif
    output->is_graph_warmup = input.input_params.meta.is_graph_warmup;
    output->json_object_errors.insert(output->json_object_errors.end(),
                                      input.json_object_errors.begin(),
                                      input.json_object_errors.end());
  }
  record_output(compute_stream,
                output,
                input.input_params.embedding.request_ids,
                input.sample_sequence_ids);
}

void ScheduleOverlap::retire_prior_inputs() const {
#if defined(USE_NPU)
  // Non-driver DP speculative results are never queried by the controller.
  // Replacing the prior output must not release temporary eager inputs until
  // its device work completes. A device wait alone cannot protect host owners.
  const auto& prior_output = state_.prior_output();
  if (state_.has_tokens() && !prior_output.retained_inputs.empty() &&
      prior_output.ready_event != nullptr) {
    CHECK(prior_output.ready_event->synchronize())
        << "failed to retire previous DP speculative input";
  }
#endif
}

void ScheduleOverlap::record_output(
    Stream& compute_stream,
    const std::optional<ForwardOutput>& output,
    const std::vector<std::string>& request_ids,
    const std::vector<std::string>& sample_sequence_ids) {
  if (retire_unconsumed_inputs_ && output.has_value()) {
    retire_prior_inputs();
  }
  if (!state_.record(output, request_ids, sample_sequence_ids)) {
    // Shutdown rejects publication and cache-only updates alike. Drain before
    // the caller releases the output and its asynchronous input owners.
    CHECK_EQ(compute_stream.synchronize(), 0);
  }
}

void ScheduleOverlap::prepare_continuation(
    LlmForwardInput& input,
    Device& device,
    Stream& compute_stream,
    PrepareModelContinuation prepare_model) {
  prepare_continuation_impl(input, device, compute_stream, prepare_model);
}

void ScheduleOverlap::prepare_continuation(
    VlmForwardInput& input,
    Device& device,
    Stream& compute_stream,
    PrepareModelContinuation prepare_model) {
  prepare_continuation_impl(input, device, compute_stream, prepare_model);
}

void ScheduleOverlap::complete_step(const LlmForwardInput& input,
                                    Stream& compute_stream,
                                    std::optional<ForwardOutput>& output) {
  complete_step_impl(input, compute_stream, output);
}

void ScheduleOverlap::complete_step(const VlmForwardInput& input,
                                    Stream& compute_stream,
                                    std::optional<ForwardOutput>& output) {
  complete_step_impl(input, compute_stream, output);
}

ForwardOutput ScheduleOverlap::consume() { return state_.consume(); }

void ScheduleOverlap::shutdown() { state_.shutdown(); }

template <typename Input>
void ScheduleOverlap::replace_tokens(
    Input& input,
    const ForwardOutput& prior_output,
    [[maybe_unused]] bool use_graph_decode_input) {
#if defined(USE_NPU)
  // These APIs take a mutable Tensor handle without changing its storage.
  auto next_tokens = prior_output.sample_output.next_tokens;
  kernel::npu::replace_token(input.token_ids,
                             next_tokens,
                             /*synchronize_stream=*/!use_graph_decode_input);
  if (use_graph_decode_input) {
    input.input_params.graph.input_tokens_override = input.token_ids;
  }
#elif defined(USE_MUSA)
  auto next_tokens = prior_output.sample_output.next_tokens;
  kernel::musa::replace_token(input.token_ids,
                              next_tokens,
                              /*synchronize_stream=*/false);
#else
  auto& flatten_tokens = input.token_ids;
  auto neg_mask = flatten_tokens < 0;
  auto clamped_neg_indices = torch::clamp(-flatten_tokens, 0);
  auto replacement =
      prior_output.sample_output.next_tokens.index({clamped_neg_indices - 1});
  input.token_ids = torch::where(neg_mask, replacement, flatten_tokens);
#endif
}

}  // namespace xllm
