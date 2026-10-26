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

#include "core/framework/config/eplb_config.h"
#include "core/runtime/worker_impl.h"
#if defined(USE_NPU)
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#elif defined(USE_MUSA)
#include "core/kernels/musa/musa_ops_api.h"
#endif

namespace xllm {

template <typename Input>
void ScheduleOverlap::prepare_continuation(WorkerImpl& worker, Input& input) {
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
  auto stream_guard = worker.compute_stream_->set_stream_guard();
  CHECK(worker.compute_stream_->wait_event(input.runtime.metadata_ready_event))
      << "failed to wait input metadata before overlap continuation";
  const ForwardOutput empty_output;
  const ForwardOutput& prior_output =
      state_.has_tokens() ? state_.prior_output() : empty_output;
  if (reuse_tokens) {
    CHECK(worker.compute_stream_->wait_event(prior_output.ready_event))
        << "failed to wait last step output before overlap continuation";
  }
  worker.update_json_object_states_by_output(
      input, prior_output, state_.sample_sequence_ids());
  worker.sanitize_json_object_error_inputs(input);
  if (reuse_tokens) {
    input = worker.prepare_decode_continuation(input, prior_output);
  }
}

template <typename Input>
std::optional<ForwardOutput> ScheduleOverlap::execute_impl(WorkerImpl& worker,
                                                           Input& input) {
  prepare_continuation(worker, input);
  auto output = worker.step_for_schedule_overlap(input);
  worker.observe_execution_memory(input.input_params.meta.is_graph_warmup);
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
  record_output(worker,
                output,
                input.input_params.embedding.request_ids,
                input.sample_sequence_ids);
  return output;
}

void ScheduleOverlap::retire_prior_inputs(
    [[maybe_unused]] const WorkerImpl& worker) const {
#if defined(USE_NPU)
  // Non-driver DP speculative results are never queried by the controller.
  // Replacing the prior output must not release temporary eager inputs until
  // its device work completes. A device wait alone cannot protect host owners.
  const auto& prior_output = state_.prior_output();
  if (state_.has_tokens() && worker.options_.enable_speculative_decode() &&
      worker.parallel_args_.dp_size() > 1 &&
      !prior_output.retained_inputs.empty() &&
      prior_output.ready_event != nullptr) {
    CHECK(prior_output.ready_event->synchronize())
        << "failed to retire previous DP speculative input";
  }
#endif
}

void ScheduleOverlap::record_output(
    WorkerImpl& worker,
    const std::optional<ForwardOutput>& output,
    const std::vector<std::string>& request_ids,
    const std::vector<std::string>& sample_sequence_ids) {
  const bool retain_tokenless_output = EPLBConfig::get_instance().enable_eplb();
  if (worker.is_driver() || retain_tokenless_output) {
    if (!state_.publish(output,
                        request_ids,
                        sample_sequence_ids,
                        retain_tokenless_output)) {
      // Shutdown woke a blocked publication. Drain before the new output's
      // retained inputs can be released by its caller.
      CHECK_EQ(worker.compute_stream_->synchronize(), 0);
    }
    return;
  }
  if (output.has_value()) {
    retire_prior_inputs(worker);
  }
  state_.update(
      output, request_ids, sample_sequence_ids, retain_tokenless_output);
}

std::optional<ForwardOutput> ScheduleOverlap::execute(WorkerImpl& worker,
                                                      LlmForwardInput& input) {
  return execute_impl(worker, input);
}

std::optional<ForwardOutput> ScheduleOverlap::execute(WorkerImpl& worker,
                                                      VlmForwardInput& input) {
  return execute_impl(worker, input);
}

ForwardOutput ScheduleOverlap::consume() {
  return state_.consume(EPLBConfig::get_instance().enable_eplb());
}

void ScheduleOverlap::shutdown() { state_.shutdown(); }

template <typename Input>
Input ScheduleOverlap::replace_tokens_impl(
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
  return input.clone();
}

LlmForwardInput ScheduleOverlap::replace_tokens(
    LlmForwardInput& input,
    const ForwardOutput& prior_output,
    bool use_graph_decode_input) {
  return replace_tokens_impl(input, prior_output, use_graph_decode_input);
}

VlmForwardInput ScheduleOverlap::replace_tokens(
    VlmForwardInput& input,
    const ForwardOutput& prior_output,
    bool use_graph_decode_input) {
  return replace_tokens_impl(input, prior_output, use_graph_decode_input);
}

}  // namespace xllm
