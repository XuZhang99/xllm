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

#include "core/runtime/schedule_overlap_state.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

using namespace std::chrono_literals;

ForwardOutput token_output(int32_t token_id) {
  ForwardOutput output;
  output.sample_output.next_tokens = torch::tensor({token_id}, torch::kInt64);
  return output;
}

TEST(ScheduleOverlapStateTest, ConsumedOutputRemainsAvailableForContinuation) {
  ScheduleOverlapState state;
  ASSERT_TRUE(state.publish(token_output(7),
                            {"request"},
                            {"sequence"},
                            /*retain_tokenless_output=*/false));

  ForwardOutput consumed = state.consume(/*retain_tokenless_output=*/false);
  ASSERT_TRUE(consumed.sample_output.next_tokens.defined());
  EXPECT_EQ(consumed.sample_output.next_tokens.item<int64_t>(), 7);
  EXPECT_TRUE(state.has_tokens());
  EXPECT_TRUE(state.prior_output().sample_output.next_tokens.defined());
  EXPECT_EQ(state.request_ids(), (std::vector<std::string>{"request"}));
  EXPECT_TRUE(state.has_request_overlap({"new", "request"}));
}

TEST(ScheduleOverlapStateTest, EmptyPublicationClearsPriorContinuation) {
  for (const bool retain_tokenless_output : {false, true}) {
    ScheduleOverlapState state;
    state.update(token_output(7),
                 {"prior_request"},
                 {"prior_sequence"},
                 retain_tokenless_output);
    ASSERT_TRUE(state.has_tokens());
    ASSERT_TRUE(state.publish(
        std::nullopt, {"request"}, {"sequence"}, retain_tokenless_output));

    const ForwardOutput consumed = state.consume(retain_tokenless_output);
    EXPECT_FALSE(consumed.sample_output.next_tokens.defined());
    EXPECT_FALSE(state.has_tokens());
    EXPECT_FALSE(state.prior_output().sample_output.next_tokens.defined());
    EXPECT_TRUE(state.request_ids().empty());
    EXPECT_TRUE(state.sample_sequence_ids().empty());
  }
}

TEST(ScheduleOverlapStateTest, TokenlessMetadataCanBePublishedAndConsumed) {
  ScheduleOverlapState state;
  ForwardOutput output;
  output.json_object_errors.push_back({"sequence", "row mismatch"});
  ASSERT_TRUE(state.publish(std::move(output),
                            {"request"},
                            {"sequence"},
                            /*retain_tokenless_output=*/false));

  const ForwardOutput consumed =
      state.consume(/*retain_tokenless_output=*/false);
  EXPECT_FALSE(consumed.sample_output.next_tokens.defined());
  ASSERT_EQ(consumed.json_object_errors.size(), 1U);
  EXPECT_EQ(state.sample_sequence_ids(),
            (std::vector<std::string>{"sequence"}));
}

TEST(ScheduleOverlapStateTest, RetainsTokenlessOutputWhenRequested) {
  ScheduleOverlapState state;
  ASSERT_TRUE(state.publish(ForwardOutput{},
                            {"request"},
                            {"sequence"},
                            /*retain_tokenless_output=*/true));

  const ForwardOutput consumed =
      state.consume(/*retain_tokenless_output=*/true);
  EXPECT_FALSE(consumed.sample_output.next_tokens.defined());
  EXPECT_EQ(state.request_ids(), (std::vector<std::string>{"request"}));
  EXPECT_EQ(state.sample_sequence_ids(),
            (std::vector<std::string>{"sequence"}));
}

TEST(ScheduleOverlapStateTest, PublishAppliesBackpressureUntilConsumed) {
  ScheduleOverlapState state;
  ASSERT_TRUE(state.publish(token_output(1),
                            {},
                            {},
                            /*retain_tokenless_output=*/false));

  auto second_publish = std::async(std::launch::async, [&state]() {
    return state.publish(token_output(2),
                         {},
                         {},
                         /*retain_tokenless_output=*/false);
  });
  EXPECT_EQ(second_publish.wait_for(50ms), std::future_status::timeout);
  EXPECT_EQ(state.consume(/*retain_tokenless_output=*/false)
                .sample_output.next_tokens.item<int64_t>(),
            1);
  ASSERT_EQ(second_publish.wait_for(1s), std::future_status::ready);
  EXPECT_TRUE(second_publish.get());
  EXPECT_EQ(state.consume(/*retain_tokenless_output=*/false)
                .sample_output.next_tokens.item<int64_t>(),
            2);
}

TEST(ScheduleOverlapStateTest, ShutdownWakesWaitingConsumerAndProducer) {
  ScheduleOverlapState state;
  auto consumer = std::async(std::launch::async, [&state]() {
    return state.consume(/*retain_tokenless_output=*/false);
  });
  EXPECT_EQ(consumer.wait_for(50ms), std::future_status::timeout);
  state.shutdown();
  ASSERT_EQ(consumer.wait_for(1s), std::future_status::ready);
  EXPECT_FALSE(consumer.get().sample_output.next_tokens.defined());

  EXPECT_FALSE(state.publish(token_output(3),
                             {},
                             {},
                             /*retain_tokenless_output=*/false));
}

TEST(ScheduleOverlapStateTest, ShutdownUnblocksPublicationBackpressure) {
  ScheduleOverlapState state;
  ASSERT_TRUE(state.publish(token_output(1),
                            {},
                            {},
                            /*retain_tokenless_output=*/false));
  auto second_publish = std::async(std::launch::async, [&state]() {
    return state.publish(token_output(2),
                         {},
                         {},
                         /*retain_tokenless_output=*/false);
  });
  EXPECT_EQ(second_publish.wait_for(50ms), std::future_status::timeout);
  state.shutdown();
  ASSERT_EQ(second_publish.wait_for(1s), std::future_status::ready);
  EXPECT_FALSE(second_publish.get());
}

TEST(ScheduleOverlapStateTest, RetainedInputsOutliveConsumedCopy) {
  std::weak_ptr<int> weak_owner;
  {
    ScheduleOverlapState state;
    auto owner = std::make_shared<int>(42);
    weak_owner = owner;
    ForwardOutput output = token_output(9);
    output.retained_inputs.emplace_back(owner);
    ASSERT_TRUE(state.publish(std::move(output),
                              {},
                              {},
                              /*retain_tokenless_output=*/false));
    owner.reset();
    ForwardOutput consumed = state.consume(/*retain_tokenless_output=*/false);
    consumed.retained_inputs.clear();
    EXPECT_FALSE(weak_owner.expired());
  }
  EXPECT_TRUE(weak_owner.expired());
}

}  // namespace
}  // namespace xllm
