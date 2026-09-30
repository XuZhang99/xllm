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

#include "core/framework/batch/batch_output_handler.h"

namespace xllm {

// Rec-domain output lifecycle. Composes token writeback with OneRec context
// targets and device-side multi-round beam results.
class RecBatchOutputHandler final {
 public:
  explicit RecBatchOutputHandler(BatchInputType input_type)
      : input_type_(input_type) {}

  void clear() { sequence_handler_.clear(); }
  void prepare(const BatchInputData& data);
  void process_sample_output(const BatchOutputData& data,
                             const RawForwardOutput& output,
                             bool replace_fake_token);
  void process_sample_output(const BatchOutputData& data,
                             const SampleOutput& output,
                             bool replace_fake_token,
                             bool force_requested_beam_result_size);
  void process_beam_search_output(const BatchOutputData& data,
                                  const RawForwardOutput& output,
                                  bool replace_fake_token);
  void process_beam_sequence_group(const BatchOutputData& data,
                                   const ForwardOutput& output);

 private:
  BatchInputType input_type_;
  BatchOutputHandler sequence_handler_;
};

}  // namespace xllm
