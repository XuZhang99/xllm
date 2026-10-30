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

#include "core/framework/sampling/sampling_params.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {

TEST(SamplingParamsTest, NormalizesGreedyAndSmallTemperatureRows) {
  RequestSamplingParam greedy;
  greedy.temperature = 0;
  greedy.top_k = 1;
  greedy.top_p = 0.2;
  greedy.min_p = 0.9;
  RequestSamplingParam random;
  random.temperature = 0.001;
  random.min_p = 0.2;
  SamplingParameters params;
  params.init({&greedy, &random}, {0, 1}, {}, {}, {});
  EXPECT_TRUE(torch::allclose(params.distribution.temperatures,
                              torch::tensor({0.0F, 0.01F})));
  EXPECT_TRUE(
      torch::allclose(params.distribution.min_p, torch::tensor({0.0F, 0.2F})));
  EXPECT_TRUE(
      torch::equal(params.selection.do_sample, torch::tensor({false, true})));
  EXPECT_FALSE(params.distribution.top_k.defined());
  EXPECT_FALSE(params.distribution.top_p.defined());
  EXPECT_FALSE(params.selection.sampling_mode == SamplingMode::RANDOM);
  EXPECT_FALSE(params.selection.sampling_mode == SamplingMode::GREEDY);
}

TEST(SamplingParamsTest, MinimumConstraintsExpandForVerifyAndStayOffDrafts) {
  SamplingParameters params;
  params.selection.selected_token_idxes = torch::arange(2, torch::kInt32);
  params.constraints.min_token_positions =
      torch::tensor({100, 200}, torch::kInt64);
  params.constraints.min_tokens_remaining =
      torch::tensor({3, 4}, torch::kInt64);
  params.constraints.min_token_ids =
      torch::tensor({{7, -1}, {8, 9}}, torch::kInt64);
  params.constraints.bad_words = torch::ones({2, 1, 2}, torch::kInt64);
  params.constraints.bad_word_suffix = torch::ones({2, 1}, torch::kInt64);
  params.random.seeds = torch::ones({2}, torch::kInt64);
  EXPECT_TRUE(params.has_sampling_fields());
  for (const auto& field : kSamplingFields) {
    if (!field.per_request) {
      repeat_interleave_if_defined(field.member(params), /*repeats=*/3);
    }
  }
  params.selection.selected_token_idxes = torch::arange(6, torch::kInt32);
  EXPECT_TRUE(torch::equal(
      params.constraints.min_token_positions,
      torch::tensor({100, 100, 100, 200, 200, 200}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(params.constraints.min_tokens_remaining,
                           torch::tensor({3, 3, 3, 4, 4, 4}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      params.constraints.min_token_ids,
      torch::tensor({{7, -1}, {7, -1}, {7, -1}, {8, 9}, {8, 9}, {8, 9}},
                    torch::kInt64)));
  for (const bool probabilistic : {false, true}) {
    SamplingParameters draft = params;
    prepare_draft_sampling(probabilistic, draft);
    EXPECT_FALSE(draft.constraints.min_token_positions.defined());
    EXPECT_FALSE(draft.constraints.min_tokens_remaining.defined());
    EXPECT_FALSE(draft.constraints.min_token_ids.defined());
    EXPECT_FALSE(draft.constraints.bad_words.defined());
    EXPECT_FALSE(draft.constraints.bad_word_suffix.defined());
    EXPECT_FALSE(draft.random.seeds.defined());
    EXPECT_FALSE(draft.has_sampling_fields());
  }
  EXPECT_TRUE(params.constraints.min_token_ids.defined());
}

TEST(SamplingParamsTest, DeviceTransferPreservesSamplingStateAndTensorTypes) {
  SamplingParameters params;
  params.selection.selected_token_idxes = torch::tensor({2, 5}, torch::kInt32);
  params.constraints.filter_mask =
      torch::tensor({{0.0F, -1.0F}, {-2.0F, 0.0F}});
  params.constraints.filter_bitmask = torch::tensor({{1}, {2}}, torch::kInt32);
  params.penalties.frequency_penalties = torch::tensor({0.5F, 0.25F});
  params.penalties.presence_penalties = torch::tensor({0.75F, 0.5F});
  params.penalties.repetition_penalties = torch::tensor({1.25F, 1.5F});
  params.distribution.temperatures = torch::tensor({0.5F, 1.0F});
  params.distribution.top_p = torch::tensor({0.75F, 1.0F});
  params.distribution.min_p = torch::tensor({0.25F, 0.0F});
  params.distribution.top_k = torch::tensor({3, 0}, torch::kInt64);
  params.constraints.logits_bias = torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}});
  params.constraints.min_token_positions =
      torch::tensor({10, 20}, torch::kInt64);
  params.constraints.min_tokens_remaining =
      torch::tensor({2, 3}, torch::kInt64);
  params.constraints.min_token_ids =
      torch::tensor({{6, -1}, {7, 8}}, torch::kInt64);
  params.random.seeds = torch::tensor({42, -1}, torch::kInt64);
  params.random.seed_offsets = torch::tensor({3, 4}, torch::kInt64);
  params.constraints.bad_words =
      torch::tensor({{{2, 3}}, {{4, 5}}}, torch::kInt64);
  params.constraints.bad_word_suffix = torch::tensor({{2}, {4}}, torch::kInt64);
  params.prefix_tokens = torch::tensor({{3}, {-1}}, torch::kInt64);
  params.penalties.unique_token_ids =
      torch::tensor({{1, 2}, {3, 4}}, torch::kInt64);
  params.penalties.unique_token_counts =
      torch::tensor({{2, 1}, {1, 3}}, torch::kInt32);
  params.penalties.unique_token_ids_lens = torch::tensor({2, 2}, torch::kInt32);
  params.selection.do_sample = torch::tensor({true, false});
  params.beam_search.acc_logprob = torch::tensor({{-0.5F}, {-1.0F}});
  params.selection.sampling_mode = SamplingMode::MIXED;
  params.output.logprobs = true;
  params.output.return_logits = true;
  params.output.is_embeddings = true;
  params.output.max_top_logprobs = 3;
  params.beam_search.num_return_sequences = 2;
  params.beam_search.use_beam_search = true;

  const SamplingParameters transferred =
      params.to(torch::Device(torch::kCPU), torch::kFloat64);

  EXPECT_TRUE(torch::equal(transferred.selection.selected_token_idxes,
                           params.selection.selected_token_idxes));
  EXPECT_EQ(transferred.selection.selected_token_idxes.scalar_type(),
            torch::kInt32);
  EXPECT_TRUE(torch::equal(transferred.constraints.filter_mask,
                           params.constraints.filter_mask));
  EXPECT_EQ(transferred.constraints.filter_mask.scalar_type(), torch::kFloat64);
  EXPECT_TRUE(transferred.constraints.filter_mask.is_contiguous());
  EXPECT_TRUE(torch::equal(transferred.constraints.filter_bitmask,
                           params.constraints.filter_bitmask));
  EXPECT_EQ(transferred.constraints.filter_bitmask.scalar_type(),
            torch::kInt32);
  EXPECT_TRUE(torch::equal(transferred.penalties.frequency_penalties,
                           params.penalties.frequency_penalties));
  EXPECT_EQ(transferred.penalties.frequency_penalties.scalar_type(),
            torch::kFloat64);
  EXPECT_TRUE(torch::equal(transferred.penalties.presence_penalties,
                           params.penalties.presence_penalties));
  EXPECT_TRUE(torch::equal(transferred.penalties.repetition_penalties,
                           params.penalties.repetition_penalties));
  EXPECT_TRUE(torch::equal(transferred.distribution.temperatures,
                           params.distribution.temperatures));
  EXPECT_TRUE(
      torch::equal(transferred.distribution.top_p, params.distribution.top_p));
  EXPECT_TRUE(
      torch::equal(transferred.distribution.min_p, params.distribution.min_p));
  EXPECT_TRUE(
      torch::equal(transferred.distribution.top_k, params.distribution.top_k));
  EXPECT_EQ(transferred.distribution.top_k.scalar_type(), torch::kInt64);
  EXPECT_TRUE(torch::equal(transferred.constraints.logits_bias,
                           params.constraints.logits_bias));
  EXPECT_EQ(transferred.constraints.logits_bias.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::equal(transferred.constraints.min_token_positions,
                           params.constraints.min_token_positions));
  EXPECT_TRUE(torch::equal(transferred.constraints.min_tokens_remaining,
                           params.constraints.min_tokens_remaining));
  EXPECT_TRUE(torch::equal(transferred.constraints.min_token_ids,
                           params.constraints.min_token_ids));
  EXPECT_TRUE(torch::equal(transferred.random.seeds, params.random.seeds));
  EXPECT_TRUE(torch::equal(transferred.random.seed_offsets,
                           params.random.seed_offsets));
  EXPECT_TRUE(torch::equal(transferred.constraints.bad_words,
                           params.constraints.bad_words));
  EXPECT_TRUE(torch::equal(transferred.constraints.bad_word_suffix,
                           params.constraints.bad_word_suffix));
  EXPECT_TRUE(torch::equal(transferred.prefix_tokens, params.prefix_tokens));
  EXPECT_TRUE(torch::equal(transferred.penalties.unique_token_ids,
                           params.penalties.unique_token_ids));
  EXPECT_TRUE(torch::equal(transferred.penalties.unique_token_counts,
                           params.penalties.unique_token_counts));
  EXPECT_EQ(transferred.penalties.unique_token_counts.scalar_type(),
            torch::kInt32);
  EXPECT_TRUE(torch::equal(transferred.penalties.unique_token_ids_lens,
                           params.penalties.unique_token_ids_lens));
  EXPECT_TRUE(torch::equal(transferred.selection.do_sample,
                           params.selection.do_sample));
  EXPECT_EQ(transferred.selection.do_sample.scalar_type(), torch::kBool);
  EXPECT_TRUE(torch::equal(transferred.beam_search.acc_logprob,
                           params.beam_search.acc_logprob));
  EXPECT_EQ(transferred.num_sample_seqs(), 2);
  EXPECT_EQ(transferred.selection.sampling_mode, SamplingMode::MIXED);
  EXPECT_TRUE(transferred.output.logprobs);
  EXPECT_TRUE(transferred.output.return_logits);
  EXPECT_TRUE(transferred.output.is_embeddings);
  EXPECT_EQ(transferred.output.max_top_logprobs, 3);
  EXPECT_EQ(transferred.beam_search.num_return_sequences, 2);
  EXPECT_TRUE(transferred.beam_search.use_beam_search);
}

TEST(SamplingParamsTest, ReinitializationClearsPreviousBatchState) {
  SamplingParameters params;
  for (const auto& field : kSamplingFields) {
    field.member(params) = torch::ones({2});
  }
  params.selection.do_sample = torch::tensor({true, false});
  params.beam_search.acc_logprob = torch::ones({2, 1});
  params.selection.sampling_mode = SamplingMode::MIXED;
  params.output.logprobs = true;
  params.output.return_logits = true;
  params.output.is_embeddings = true;
  params.output.max_top_logprobs = 5;
  params.beam_search.num_return_sequences = 3;
  params.beam_search.use_beam_search = true;
  RequestSamplingParam greedy;
  greedy.temperature = 0.0F;

  params.init({&greedy}, {7}, {}, {}, {});

  EXPECT_TRUE(torch::equal(params.selection.selected_token_idxes,
                           torch::tensor({7}, torch::kInt32)));
  for (const auto& field : kSamplingFields) {
    EXPECT_FALSE(field.member(params).defined());
  }
  EXPECT_FALSE(params.selection.do_sample.defined());
  EXPECT_FALSE(params.beam_search.acc_logprob.defined());
  EXPECT_EQ(params.selection.sampling_mode, SamplingMode::GREEDY);
  EXPECT_FALSE(params.output.logprobs);
  EXPECT_FALSE(params.output.return_logits);
  EXPECT_FALSE(params.output.is_embeddings);
  EXPECT_EQ(params.output.max_top_logprobs, 0);
  EXPECT_EQ(params.beam_search.num_return_sequences, 0);
  EXPECT_FALSE(params.beam_search.use_beam_search);
  EXPECT_TRUE(params.is_all_greedy());
}

}  // namespace xllm
