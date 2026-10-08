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

#include "core/framework/batch/forward_input_factory.h"

#include <glog/logging.h>

#include <limits>

#include "core/framework/batch/batch_group.h"
#include "core/framework/model/model_args.h"
#include "core/util/utils.h"
#include "core/util/threadpool.h"

namespace xllm {

ForwardInputFactory::ForwardInputFactory(ForwardInputFactoryOptions options)
    : options_(options) {
  CHECK_GT(options_.dp_size, 0);
  CHECK_GT(options_.cp_size, 0);
  CHECK_LE(options_.cp_size, std::numeric_limits<int32_t>::max());
  dp_batch_embedding_ids_.resize(options_.dp_size);
  dp_batch_request_ids_.resize(options_.dp_size);
  dp_batch_generations_.resize(options_.dp_size, 0);
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/16,
      /*cpu_binding=*/false,
      /*pool_name=*/"ForwardInputFactory.forward_input");
}

ForwardInputFactory::~ForwardInputFactory() = default;

PreparedLlmInputGroup ForwardInputFactory::create_inputs(
    BatchGroup& batches,
    const ModelArgs& model_args,
    bool enable_graph) {
  CHECK_EQ(batches.size(), options_.dp_size)
      << "Split DP batch failed with dp_size as " << options_.dp_size
      << " and actual batch size as " << batches.size() << ".";

  PreparedLlmInputGroup prepared;
  auto& inputs = prepared.inputs;
  inputs.reserve(options_.dp_size);
  prepared.dp_token_counts.resize(options_.dp_size);
  std::vector<int32_t> dp_sequence_counts(options_.dp_size);
  std::vector<int32_t> dp_kv_max_seq_lens(options_.dp_size);
  std::vector<int32_t> dp_is_decode(options_.dp_size, 0);
  BatchForwardType batch_forward_type;

  for (uint32_t dp_rank = 0; dp_rank < options_.dp_size; ++dp_rank) {
    inputs.emplace_back(batches[dp_rank].prepare_forward_input(
        model_args, threadpool_.get(), static_cast<int32_t>(options_.cp_size)));
    const auto& meta = inputs[dp_rank].input_params.meta;
    const BatchForwardType& current_forward_type = meta.batch_forward_type;
    prepared.dp_token_counts[dp_rank] =
        static_cast<int32_t>(inputs[dp_rank].host_token_ids().numel());
    dp_sequence_counts[dp_rank] = meta.num_sequences;
    dp_kv_max_seq_lens[dp_rank] = meta.kv_max_seq_len;
    if (util::is_deepseek_v4_model_type(model_args.model_type())) {
      const int64_t actual_scheduled_tokens =
          inputs[dp_rank].host_token_ids().numel();
      CHECK_LE(actual_scheduled_tokens, options_.max_tokens_per_batch)
          << "DSV4 actual scheduled tokens exceed max_tokens_per_batch used "
             "for SWA cache allocation. This can make the shared SWA burst "
             "pool smaller than the block/table consumer needs and may cause "
             "SWA KV rows to be overwritten or read from the wrong position. "
             "Please increase --max_tokens_per_batch, reduce scheduler token "
             "load, or check chunked-prefill padding. Details: dp_rank="
          << dp_rank << ", actual_scheduled_tokens=" << actual_scheduled_tokens
          << ", max_tokens_per_batch=" << options_.max_tokens_per_batch
          << ", q_max_seq_len=" << meta.q_max_seq_len
          << ", kv_max_seq_len=" << meta.kv_max_seq_len
          << ", batch_forward_type=" << current_forward_type.to_string();
    }
    if (batch_forward_type.is_empty() && !current_forward_type.is_empty()) {
      batch_forward_type = current_forward_type;
    }
    if (!current_forward_type.is_empty()) {
      prepared.has_non_empty_batch = true;
      prepared.all_non_empty_batches_are_decode =
          prepared.all_non_empty_batches_are_decode &&
          current_forward_type.is_decode();
    }
    prepared.is_graph_warmup = prepared.is_graph_warmup || meta.is_graph_warmup;
    dp_is_decode[dp_rank] =
        current_forward_type.is_decode() && meta.q_max_seq_len == 1;

    const auto& embedding = inputs[dp_rank].input_params.embedding;
    if (dp_batch_embedding_ids_[dp_rank] != embedding.embedding_ids ||
        dp_batch_request_ids_[dp_rank] != embedding.request_ids) {
      dp_batch_embedding_ids_[dp_rank] = embedding.embedding_ids;
      dp_batch_request_ids_[dp_rank] = embedding.request_ids;
      ++dp_batch_generations_[dp_rank];
    }
  }

  // Graph decode requires empty ranks to participate using Worker fake inputs.
  if (enable_graph && batch_forward_type.is_decode()) {
    for (uint32_t dp_rank = 0; dp_rank < options_.dp_size; ++dp_rank) {
      if (inputs[dp_rank].input_params.meta.batch_forward_type.is_empty() &&
          prepared.dp_token_counts[dp_rank] == 0) {
        dp_is_decode[dp_rank] = 1;
      }
    }
  }

  for (auto& input : inputs) {
    input.input_params.meta.is_graph_warmup = prepared.is_graph_warmup;
    auto& parallel = input.input_params.parallel;
    parallel.dp_global_token_nums = prepared.dp_token_counts;
    parallel.dp_global_sequence_nums = dp_sequence_counts;
    parallel.raw_dp_global_token_nums = prepared.dp_token_counts;
    parallel.dp_global_batch_generations = dp_batch_generations_;
    parallel.dp_global_kv_max_seq_lens = dp_kv_max_seq_lens;
    parallel.dp_is_decode = dp_is_decode;
    if (input.input_params.meta.batch_forward_type.is_empty()) {
      input.input_params.meta.batch_forward_type = batch_forward_type;
    }
  }
  return prepared;
}

}  // namespace xllm
