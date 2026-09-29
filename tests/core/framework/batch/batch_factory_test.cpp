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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/batch/rec_batch_factory.h"
#include "core/framework/batch/sequence_batch_factory.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/request/rec_sequence.h"
#include "core/framework/request/request.h"
#include "core/framework/request/stopping_checker.h"
#include "core/util/hash_util.h"

namespace xllm {
namespace {

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_value_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_value_; }

 private:
  T& value_;
  T old_value_;
};

std::shared_ptr<Request> make_request(int32_t rank,
                                      RecType rec_type = RecType::kNone,
                                      int32_t beam_width = 1) {
  RequestSamplingParam sampling;
  sampling.beam_width = beam_width;
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(/*max_generated_tokens=*/8);
  stopping.set_max_context_len(/*max_context_len=*/64);
  stopping.set_ignore_eos(true);
  RequestState state("prompt",
                     {1, 2, 3},
                     sampling,
                     SchedulerParam{},
                     stopping,
                     /*seq_capacity=*/64,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/true,
                     /*enable_schedule_overlap=*/false,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  state.rec_type = rec_type;
  auto request = std::make_shared<Request>("request", "", "", std::move(state));
  request->sequences().front()->set_dp_rank(rank);
  return request;
}

TEST(BatchFactoryTest, IndependentFactoriesKeepTheirOwnDpSize) {
  SequenceBatchFactory single(/*dp_size=*/1);
  SequenceBatchFactory multiple(/*dp_size=*/3);
  auto request = make_request(/*rank=*/2);
  auto batches = multiple.create_batches(
      {request}, {request->sequences().front().get()}, {2});
  ASSERT_EQ(batches.size(), 3);
  EXPECT_TRUE(batches[0].empty());
  EXPECT_TRUE(batches[1].empty());
  EXPECT_EQ(batches[2].size(), 1);
  EXPECT_EQ(single.create_batches({}, {}, {}).size(), 1);
  EXPECT_EQ(multiple.create_batches({}, {}, {}).size(), 3);
}

TEST(BatchFactoryTest, PreservesRankLocalSequenceOrderAndBudgets) {
  SequenceBatchFactory factory(/*dp_size=*/3);
  auto first = make_request(/*rank=*/2);
  auto second = make_request(/*rank=*/0);
  auto third = make_request(/*rank=*/2);
  auto* seq1 = first->sequences().front().get();
  auto* seq2 = second->sequences().front().get();
  auto* seq3 = third->sequences().front().get();
  auto batches = factory.create_batches(
      {first, second, third}, {seq1, seq2, seq3}, {1, 2, 3});
  EXPECT_EQ(batches[0].get_sequences(), (std::vector<Sequence*>{seq2}));
  EXPECT_EQ(batches[0].get_allowed_max_tokens(), (std::vector<uint32_t>{2}));
  EXPECT_TRUE(batches[1].empty());
  EXPECT_EQ(batches[2].get_sequences(), (std::vector<Sequence*>{seq1, seq3}));
  EXPECT_EQ(batches[2].get_allowed_max_tokens(), (std::vector<uint32_t>{1, 3}));
  EXPECT_NE(batches[0].batch_id(), UNINITIALIZED_BATCH_ID);
  EXPECT_NE(batches[0].batch_id(), batches[2].batch_id());
  EXPECT_EQ(batches[1].batch_id(), UNINITIALIZED_BATCH_ID);
}

TEST(BatchFactoryTest, BeamBatchRetainsAllRequestGroupsInOrder) {
  SequenceBatchFactory factory(/*dp_size=*/1);
  auto first = make_request(/*rank=*/0);
  auto beam = make_request(/*rank=*/0, RecType::kNone, /*beam_width=*/2);
  auto batches = factory.create_batches(
      {first, beam},
      {first->sequences()[0].get(), beam->sequences()[0].get()},
      {3, 3});
  ASSERT_EQ(batches[0].sequence_groups().size(), 2);
  EXPECT_EQ(batches[0].sequence_groups()[0], first->sequence_group());
  EXPECT_EQ(batches[0].sequence_groups()[1], beam->sequence_group());
}

TEST(BatchFactoryTest, ReusingFactoryDoesNotRetainPreviousBeamGroups) {
  SequenceBatchFactory factory(/*dp_size=*/1);
  auto beam = make_request(/*rank=*/0, RecType::kNone, /*beam_width=*/2);
  auto beam_batches =
      factory.create_batches({beam}, {beam->sequences()[0].get()}, {3});
  ASSERT_EQ(beam_batches[0].sequence_groups().size(), 1);

  auto request = make_request(/*rank=*/0);
  auto batches =
      factory.create_batches({request}, {request->sequences()[0].get()}, {3});
  EXPECT_TRUE(batches[0].sequence_groups().empty());
  EXPECT_EQ(batches[0].get_sequences(),
            (std::vector<Sequence*>{request->sequences()[0].get()}));
  EXPECT_EQ(batches[0].input_type(), BatchInputType::SEQUENCE);

  auto empty_batches = factory.create_batches({}, {}, {});
  EXPECT_TRUE(empty_batches[0].empty());
  EXPECT_EQ(empty_batches[0].batch_id(), UNINITIALIZED_BATCH_ID);
}

TEST(BatchFactoryTest, FactoriesKeepIndependentInputContractsAcrossCalls) {
  RecBatchFactory onerec_factory(/*dp_size=*/2, BatchInputType::ONEREC);
  RecBatchFactory multi_round_factory(/*dp_size=*/1,
                                      BatchInputType::REC_MULTI_ROUND);
  auto onerec = make_request(/*rank=*/1, RecType::kOneRec);
  auto multi_round = make_request(/*rank=*/0, RecType::kLlmRec);

  // Interleave calls so neither factory can replace the other's configuration.
  for (int32_t iteration = 0; iteration < 2; ++iteration) {
    auto onerec_batches = onerec_factory.create_batches(
        {onerec}, {onerec->sequences()[0].get()}, {0});
    ASSERT_EQ(onerec_batches.size(), 2);
    EXPECT_TRUE(onerec_batches[0].empty());
    EXPECT_EQ(onerec_batches[1].input_type(), BatchInputType::ONEREC);
    EXPECT_EQ(onerec_batches[1].sequence_groups(),
              (std::vector<SequencesGroup*>{onerec->sequence_group()}));

    auto multi_round_batches = multi_round_factory.create_batches(
        {multi_round}, {multi_round->sequences()[0].get()}, {2});
    ASSERT_EQ(multi_round_batches.size(), 1);
    EXPECT_EQ(multi_round_batches[0].input_type(),
              BatchInputType::REC_MULTI_ROUND);
    EXPECT_TRUE(multi_round_batches[0].sequence_groups().empty());
    EXPECT_EQ(multi_round_batches[0].get_sequences(),
              (std::vector<Sequence*>{multi_round->sequences()[0].get()}));
    EXPECT_EQ(multi_round_batches[0].get_allowed_max_tokens(),
              (std::vector<uint32_t>{2}));
  }
}

TEST(BatchFactoryTest, RecSequenceInputsWithoutBeamDoNotAttachGroups) {
  for (BatchInputType type :
       {BatchInputType::SEQUENCE, BatchInputType::REC_MULTI_ROUND}) {
    RecBatchFactory factory(/*dp_size=*/2, type);
    auto request = make_request(/*rank=*/1, RecType::kLlmRec);
    auto* sequence = request->sequences()[0].get();
    auto batches = factory.create_batches({request}, {sequence}, {2});
    ASSERT_EQ(batches.size(), 2);
    EXPECT_TRUE(batches[0].empty());
    EXPECT_EQ(batches[1].input_type(), type);
    EXPECT_EQ(batches[1].get_sequences(), (std::vector<Sequence*>{sequence}));
    EXPECT_EQ(batches[1].get_allowed_max_tokens(), (std::vector<uint32_t>{2}));
    EXPECT_TRUE(batches[1].sequence_groups().empty());
  }
}

TEST(BatchFactoryTest, RecMultiRoundBuilderUsesScheduledSequencesAndBudgets) {
  ScopedConfigValue<int32_t> decode_rounds(
      RecConfig::get_instance().max_decode_rounds(), 3);
  ScopedConfigValue<int32_t> beam_width(
      BeamSearchConfig::get_instance().beam_width(), 2);
  RecBatchFactory factory(/*dp_size=*/1, BatchInputType::REC_MULTI_ROUND);
  auto first = make_request(/*rank=*/0, RecType::kLlmRec, /*beam_width=*/2);
  auto second = make_request(/*rank=*/0, RecType::kLlmRec, /*beam_width=*/2);
  // A group may own sequences that were not scheduled. Their tokens must not
  // be flattened into the input or paired with another sequence's budget.
  auto finished = first->sequences()[0]->fork(/*index=*/1);
  finished->finish();
  first->sequences().emplace_back(std::move(finished));

  auto* first_sequence = first->sequences()[0].get();
  auto* second_sequence = second->sequences()[0].get();
  auto batches = factory.create_batches(
      {first, second}, {first_sequence, second_sequence}, {1, 2});
  ASSERT_EQ(batches[0].sequence_groups().size(), 2);
  ModelArgs args;
  auto input = batches[0].prepare_forward_input(/*num_decoding_tokens=*/1,
                                                /*min_decoding_batch_size=*/0,
                                                args);

  EXPECT_TRUE(
      torch::equal(input.token_ids, torch::tensor({1, 1, 2}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(input.positions, torch::tensor({0, 0, 1}, torch::kInt32)));
  EXPECT_EQ(input.input_params.meta.num_sequences, 2);
  EXPECT_EQ(input.input_params.meta.q_max_seq_len, 2);
  EXPECT_EQ(input.input_params.meta.batch_id, batches[0].batch_id());
  EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_prefill());
  ASSERT_TRUE(input.step_decode.has_value());
  EXPECT_EQ(input.step_decode->batch_size, 2);
  EXPECT_EQ(input.step_decode->beam_width, 2);
  EXPECT_EQ(input.step_decode->total_round, 3);
  EXPECT_EQ(batches[0].get_allowed_max_tokens(), (std::vector<uint32_t>{1, 2}));
}

TEST(BatchFactoryTest, RecMultiRoundOutputsFollowRankLocalRequestOrder) {
  RecBatchFactory factory(/*dp_size=*/2, BatchInputType::REC_MULTI_ROUND);
  auto first = make_request(/*rank=*/1, RecType::kLlmRec, /*beam_width=*/2);
  auto other_rank =
      make_request(/*rank=*/0, RecType::kLlmRec, /*beam_width=*/2);
  auto last = make_request(/*rank=*/1, RecType::kLlmRec, /*beam_width=*/2);
  auto batches = factory.create_batches({first, other_rank, last},
                                        {first->sequences()[0].get(),
                                         other_rank->sequences()[0].get(),
                                         last->sequences()[0].get()},
                                        {3, 3, 3});
  ASSERT_EQ(batches[1].sequence_groups().size(), 2);
  EXPECT_EQ(batches[1].sequence_groups()[0], first->sequence_group());
  EXPECT_EQ(batches[1].sequence_groups()[1], last->sequence_group());

  ForwardOutput output;
  // The final result may be wider than the requested beam width.
  output.beam_sequence_group =
      torch::arange(18, torch::kInt32).reshape({2, 3, 3});
  output.beam_search_output.out_logprobs =
      torch::tensor({-0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f});
  batches[1].process_beam_sequence_group(output);

  const auto& first_result =
      RecSequence::from(*first->sequences()[0]).beam_search_result();
  const auto& last_result =
      RecSequence::from(*last->sequences()[0]).beam_search_result();
  EXPECT_TRUE(first_result.ready());
  EXPECT_EQ(first_result.beam_width(), 3);
  EXPECT_EQ(first_result.total_rounds(), 3);
  EXPECT_EQ(
      first_result.beams(),
      (std::vector<std::vector<int32_t>>{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}));
  EXPECT_EQ(first_result.last_logprobs(),
            (std::vector<float>{-0.1f, -0.2f, -0.3f}));
  EXPECT_EQ(last_result.beams(),
            (std::vector<std::vector<int32_t>>{
                {9, 10, 11}, {12, 13, 14}, {15, 16, 17}}));
  EXPECT_EQ(last_result.last_logprobs(),
            (std::vector<float>{-0.4f, -0.5f, -0.6f}));
  EXPECT_FALSE(RecSequence::from(*other_rank->sequences()[0])
                   .beam_search_result()
                   .ready());
}

TEST(BatchFactoryTest, OneRecInputTypeSurvivesSequenceRefresh) {
  for (BatchInputType type :
       {BatchInputType::ONEREC, BatchInputType::ONEREC_XATTENTION}) {
    RecBatchFactory factory(/*dp_size=*/2, type);
    auto request = make_request(/*rank=*/1, RecType::kOneRec);
    auto* sequence = request->sequences()[0].get();
    // Embedding-only decoder prefill is allowed to have a zero token budget.
    auto batches = factory.create_batches({request}, {sequence}, {0});
    EXPECT_TRUE(batches[0].empty());
    EXPECT_EQ(batches[0].input_type(), type);
    EXPECT_FALSE(batches[1].empty());
    EXPECT_EQ(batches[1].size(), 0);
    EXPECT_EQ(batches[1].get_sequences(), (std::vector<Sequence*>{sequence}));
    const uint64_t batch_id = batches[1].batch_id();
    batches[1].refresh_sequences_from_groups();
    EXPECT_EQ(batches[1].size(), 1);
    EXPECT_EQ(batches[1].input_type(), type);
    batches[1].set_batch_id();
    EXPECT_EQ(batches[1].batch_id(), batch_id);
  }
}

TEST(BatchFactoryTest, EmptyRecRanksPrepareEmptyInputs) {
  ModelArgs args;
  for (BatchInputType type : {BatchInputType::ONEREC,
                              BatchInputType::ONEREC_XATTENTION,
                              BatchInputType::REC_MULTI_ROUND}) {
    RecBatchFactory factory(/*dp_size=*/2, type);
    auto batches = factory.create_batches({}, {}, {});
    for (auto& batch : batches) {
      auto input = batch.prepare_forward_input(/*num_decoding_tokens=*/1,
                                               /*min_decoding_batch_size=*/0,
                                               args);
      EXPECT_TRUE(input.input_params.meta.batch_forward_type.is_empty());
    }
  }
}

TEST(BatchFactoryTest, TransfersAreConsumedOnlyForActiveRanks) {
  SequenceBatchFactory factory(/*dp_size=*/2);
  auto request = make_request(/*rank=*/1);
  std::vector<std::vector<BlockTransferInfo>> transfers(2);
  uint8_t key[XXH3_128BITS_HASH_VALUE_LEN] = {};
  transfers[0].emplace_back(/*src_id=*/1, /*dst_id=*/2, key, TransferType::G2D);
  transfers[1].emplace_back(/*src_id=*/3, /*dst_id=*/4, key, TransferType::G2D);
  auto batches = factory.create_batches(
      {request}, {request->sequences()[0].get()}, {3}, &transfers);
  EXPECT_EQ(transfers[0].size(), 1);
  EXPECT_TRUE(transfers[1].empty());
  EXPECT_TRUE(batches[0].empty());
  EXPECT_FALSE(batches[1].empty());
}

TEST(BatchFactoryDeathTest, RejectsInvalidDpSize) {
  EXPECT_DEATH(SequenceBatchFactory(/*dp_size=*/0), "dp_size_");
  EXPECT_DEATH((RecBatchFactory(/*dp_size=*/-1, BatchInputType::ONEREC)),
               "dp_size_");
}

TEST(BatchFactoryDeathTest, RejectsUnsupportedInputTypeAtConstruction) {
  EXPECT_DEATH(
      (RecBatchFactory(/*dp_size=*/1, static_cast<BatchInputType>(-1))),
      "Unsupported batch input type");
}

TEST(BatchFactoryDeathTest, RejectsGroupOnlySequenceInput) {
  auto request = make_request(/*rank=*/0, RecType::kOneRec);
  Batch batch;
  batch.add(request->sequence_group());
  ModelArgs args;
  EXPECT_DEATH(batch.prepare_forward_input(/*num_decoding_tokens=*/1,
                                           /*min_decoding_batch_size=*/0,
                                           args),
               "requires an explicit OneRec batch input type");
  EXPECT_DEATH(batch.prepare_forward_input(args, /*thread_pool=*/nullptr),
               "requires an explicit OneRec batch input type");
}

TEST(BatchFactoryDeathTest, RejectsOneRecInputWithoutRequestGroups) {
  auto request = make_request(/*rank=*/0, RecType::kOneRec);
  Batch batch(BatchInputType::ONEREC);
  batch.add(request->sequences()[0].get());
  ModelArgs args;
  EXPECT_DEATH(batch.prepare_forward_input(/*num_decoding_tokens=*/1,
                                           /*min_decoding_batch_size=*/0,
                                           args),
               "OneRec input requires request groups");
}

TEST(BatchFactoryDeathTest, RejectsMismatchedBudgetsAndInvalidRanks) {
  SequenceBatchFactory factory(/*dp_size=*/2);
  auto request = make_request(/*rank=*/0);
  auto* sequence = request->sequences()[0].get();
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {}),
               "budgets.size");
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {0}), "budgets");
  const size_t overflow_budget =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {overflow_budget}),
               "budgets");
  sequence->set_dp_rank(-1);
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {1}), "dp_rank");
  sequence->set_dp_rank(2);
  EXPECT_DEATH(factory.create_batches({request}, {sequence}, {1}), "dp_rank");
}

TEST(BatchFactoryDeathTest, RejectsInvalidSwapRankCount) {
  SequenceBatchFactory factory(/*dp_size=*/2);
  std::vector<std::vector<BlockTransferInfo>> transfers(1);
  EXPECT_DEATH(factory.create_batches({}, {}, {}, &transfers), "swap_infos");
}

}  // namespace
}  // namespace xllm
