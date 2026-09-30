/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/batch/batch.h"

#include <glog/logging.h>

namespace xllm {

Batch::Batch(BatchInputType input_type)
    : Batch(input_type == BatchInputType::SEQUENCE ? BatchDomain::SEQUENCE
                                                   : BatchDomain::REC,
            input_type) {}

Batch::Batch(BatchDomain domain, BatchInputType input_type) {
  if (domain == BatchDomain::SEQUENCE) {
    CHECK(input_type == BatchInputType::SEQUENCE)
        << "Sequence batches require sequence input";
    batch_.emplace<SequenceBatch>();
    return;
  }
  batch_.emplace<RecBatch>(input_type);
}

Batch::Batch(Sequence* sequence) { add(sequence); }
Batch::Batch(const std::vector<Sequence*>& sequences) { add(sequences); }

BatchState& Batch::state() {
  return std::visit([](auto& batch) -> BatchState& { return batch.state(); },
                    batch_);
}

const BatchState& Batch::state() const {
  return std::visit(
      [](const auto& batch) -> const BatchState& { return batch.state(); },
      batch_);
}

BatchInputType Batch::input_type() const {
  return std::visit([](const auto& batch) { return batch.input_type(); },
                    batch_);
}

void Batch::reserve(size_t sequence_count, size_t group_count) {
  state().reserve(sequence_count, group_count);
}

void Batch::add(Sequence* sequence, uint32_t allowed_max_token) {
  state().add(sequence, allowed_max_token);
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

void Batch::add(SequencesGroup* group) { state().add(group); }
void Batch::set_batch_id() { state().set_batch_id(); }
void Batch::update_forward_type(Sequence* sequence) {
  state().update_forward_type(sequence);
}
void Batch::refresh_forward_type() {
  state().refresh_forward_type(get_sequences());
}

size_t Batch::size() const {
  return std::visit([](const auto& batch) { return batch.size(); }, batch_);
}

Sequence* Batch::operator[](size_t index) const {
  return std::visit(
      [index](const auto& batch) { return batch.sequence(index); }, batch_);
}

std::vector<Sequence*> Batch::get_sequences() {
  return static_cast<const Batch&>(*this).get_sequences();
}
std::vector<Sequence*> Batch::get_sequences() const {
  return std::visit([](const auto& batch) { return batch.get_sequences(); },
                    batch_);
}

void Batch::refresh_sequences_from_groups() {
  std::visit([](auto& batch) { batch.refresh_sequences_from_groups(); },
             batch_);
}

ForwardInput Batch::prepare_forward_input(uint32_t num_decoding_tokens,
                                          uint32_t min_decoding_batch_size,
                                          const ModelArgs& args,
                                          int32_t cp_size) {
  return std::visit(
      [&](auto& batch) {
        return batch.prepare_forward_input(
            num_decoding_tokens, min_decoding_batch_size, args, cp_size);
      },
      batch_);
}

ForwardInput Batch::prepare_forward_input(const ModelArgs& args,
                                          ThreadPool* thread_pool,
                                          int32_t cp_size) {
  return std::visit(
      [&](auto& batch) {
        return batch.prepare_forward_input(args, thread_pool, cp_size);
      },
      batch_);
}

ForwardInput Batch::prepare_rec_forward_input(uint32_t num_decoding_tokens,
                                              uint32_t min_decoding_batch_size,
                                              const ModelArgs& args,
                                              MPMCThreadPool* thread_pool) {
  auto* batch = std::get_if<RecBatch>(&batch_);
  CHECK(batch != nullptr)
      << "Rec input requires an explicit Rec batch input type";
  return batch->prepare_rec_forward_input(
      num_decoding_tokens, min_decoding_batch_size, args, thread_pool);
}

void Batch::process_sample_output(const RawForwardOutput& output,
                                  bool replace_fake_token) {
  const auto sequences = get_sequences();
  state().output_handler().process_sample_output(
      {sequences, state().sequence_groups()}, output, replace_fake_token);
}

void Batch::process_sample_output(const SampleOutput& output,
                                  bool replace_fake_token,
                                  bool force_requested_beam_result_size) {
  const auto sequences = get_sequences();
  state().output_handler().process_sample_output(
      {sequences, state().sequence_groups()},
      output,
      replace_fake_token,
      force_requested_beam_result_size);
}

void Batch::process_beam_sequence_group(const ForwardOutput& output) {
  const auto sequences = get_sequences();
  state().output_handler().process_beam_sequence_group(
      {sequences, state().sequence_groups()}, output);
}

void Batch::process_beam_search_output(const RawForwardOutput& output,
                                       bool replace_fake_token) {
  const auto sequences = get_sequences();
  state().output_handler().process_beam_search_output(
      {sequences, state().sequence_groups()}, output, replace_fake_token);
}

void Batch::finish() {
  for (auto* sequence_group : state().sequence_groups()) {
    sequence_group->finish();
  }

  const auto sequences = get_sequences();
  for (auto* sequence : sequences) {
    sequence->finish();
  }
}
}  // namespace xllm
