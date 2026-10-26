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

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/constrained/json_object_grammar.h"
#include "core/runtime/worker_impl.h"

namespace xllm {
namespace {

using namespace std::chrono_literals;

ForwardOutput token_output(int32_t token_id) {
  ForwardOutput output;
  output.sample_output.next_tokens = torch::tensor({token_id}, torch::kInt64);
  return output;
}

class OverlapTestWorker final : public WorkerImpl {
 public:
  enum class ContinuationMode : int8_t {
    CUSTOM_REBUILD,
    DEFAULT_REPLACEMENT,
    MODEL_MANAGED
  };

  explicit OverlapTestWorker(
      std::vector<std::optional<ForwardOutput>> outputs,
      bool driver = true,
      ContinuationMode continuation_mode = ContinuationMode::CUSTOM_REBUILD)
      : WorkerImpl(ParallelArgs(/*rank=*/driver ? 0 : 1,
                                /*world_size=*/2,
                                /*dp_size=*/1,
                                /*process_group=*/nullptr),
                   torch::Device(torch::kPrivateUse1, /*index=*/0),
                   options()),
        outputs_(std::move(outputs)),
        continuation_mode_(continuation_mode) {}

  ~OverlapTestWorker() override { drain_pending_execution(); }

  bool init_model(ModelContext& /*context*/) override { return false; }

  void prepare_work_before_execute(const LlmForwardInput& input,
                                   LlmForwardInput& prepared) override {
    prepare_input(input, prepared);
  }

  void prepare_work_before_execute(const VlmForwardInput& input,
                                   VlmForwardInput& prepared) override {
    prepare_input(input, prepared);
  }

  int32_t continuation_count() const { return continuation_count_; }
  const torch::Tensor& executed_tokens() const { return executed_tokens_; }
  const std::vector<int32_t>& executed_grammar_tokens() const {
    return executed_grammar_tokens_;
  }
  bool executed_grammar_initialized() const {
    return executed_grammar_initialized_;
  }

 protected:
  bool prepare_model_decode_continuation(
      LlmForwardInput& input,
      const ForwardOutput& prior_output) override {
    return prepare_continuation(input, prior_output);
  }

  bool prepare_model_decode_continuation(
      VlmForwardInput& input,
      const ForwardOutput& prior_output) override {
    return prepare_continuation(input, prior_output);
  }

  std::optional<ForwardOutput> step_for_schedule_overlap(
      const LlmForwardInput& input) override {
    return execute(input);
  }

  std::optional<ForwardOutput> step_for_schedule_overlap(
      const VlmForwardInput& input) override {
    return execute(input);
  }

 private:
  static runtime::Options options() {
    runtime::Options options;
    options.enable_schedule_overlap(true);
    return options;
  }

  template <typename Input>
  void prepare_input(const Input& input, Input& prepared) const {
    prepared = input.clone();
    if (continuation_mode_ == ContinuationMode::DEFAULT_REPLACEMENT) {
      prepared.token_ids = input.token_ids.to(device());
    }
  }

  template <typename Input>
  bool prepare_continuation(Input& input, const ForwardOutput& prior_output) {
    ++continuation_count_;
    if (continuation_mode_ == ContinuationMode::DEFAULT_REPLACEMENT) {
      return WorkerImpl::prepare_model_decode_continuation(input, prior_output);
    }
    if (continuation_mode_ == ContinuationMode::CUSTOM_REBUILD) {
      input.token_ids =
          prior_output.sample_output.next_tokens.to(torch::kInt32);
    }
    return true;
  }

  template <typename Input>
  std::optional<ForwardOutput> execute(const Input& input) {
    executed_tokens_ = input.token_ids;
    executed_grammar_initialized_ =
        !input.json_object_states.empty() &&
        input.json_object_states.front().initialized();
    executed_grammar_tokens_.clear();
    if (executed_grammar_initialized_) {
      executed_grammar_tokens_ =
          input.json_object_states.front().snapshot().token_ids;
    }
    CHECK_LT(step_index_, outputs_.size());
    return std::move(outputs_[step_index_++]);
  }

  std::vector<std::optional<ForwardOutput>> outputs_;
  ContinuationMode continuation_mode_;
  size_t step_index_ = 0;
  int32_t continuation_count_ = 0;
  torch::Tensor executed_tokens_;
  std::vector<int32_t> executed_grammar_tokens_;
  bool executed_grammar_initialized_ = false;
};

template <typename Input>
Input step_input(bool decode,
                 int32_t token_id = 7,
                 std::string request_id = "request") {
  Input input;
  input.token_ids = torch::tensor({token_id}, torch::kInt32);
  input.input_params.meta.batch_forward_type =
      decode ? BatchForwardType::DECODE : BatchForwardType::PREFILL;
  input.input_params.embedding.request_ids = {request_id};
  input.sample_sequence_ids = {request_id + "#0"};
  return input;
}

template <typename Input>
class ScheduleOverlapTest : public ::testing::Test {};

using ForwardInputTypes = ::testing::Types<LlmForwardInput, VlmForwardInput>;
TYPED_TEST_SUITE(ScheduleOverlapTest, ForwardInputTypes);

TYPED_TEST(ScheduleOverlapTest, ConsumedResultRemainsAvailableForContinuation) {
  OverlapTestWorker worker({token_output(17), token_output(23)});
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  ASSERT_EQ(worker.get_last_step_result()
                .sample_output.next_tokens.template item<int64_t>(),
            17);

  auto second = worker.step_async(
      step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1));
  ASSERT_TRUE(std::move(second).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 17);
  EXPECT_EQ(worker.get_last_step_result()
                .sample_output.next_tokens.template item<int64_t>(),
            23);
}

TYPED_TEST(ScheduleOverlapTest,
           DefaultContinuationReplacesIndexedPlaceholders) {
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  ForwardOutput prior_output;
  prior_output.sample_output.next_tokens =
      torch::tensor({17, 23}, torch::kInt64).to(device);
  OverlapTestWorker worker(
      {std::move(prior_output), token_output(31)},
      /*driver=*/true,
      OverlapTestWorker::ContinuationMode::DEFAULT_REPLACEMENT);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true);
  input.token_ids = torch::tensor({-2, 9, -1}, torch::kInt32);
  auto second = worker.step_async(input);
  ASSERT_TRUE(std::move(second).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_tokens().device(), device);
  EXPECT_TRUE(torch::equal(worker.executed_tokens().to(torch::kCPU),
                           torch::tensor({23, 9, 17}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(input.token_ids, torch::tensor({-2, 9, -1}, torch::kInt32)));
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, ModelManagedContinuationPreservesModelInput) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  OverlapTestWorker worker({token_output(0), token_output(1)},
                           /*driver=*/true,
                           OverlapTestWorker::ContinuationMode::MODEL_MANAGED);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1);
  input.json_object_states = {grammar.initial_state()};
  input.sample_prior_output_rows = {0};
  auto second = worker.step_async(input);
  const auto output = std::move(second).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->json_object_errors.empty());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), -1);
  EXPECT_EQ(worker.executed_grammar_tokens(), std::vector<int32_t>({0}));
  EXPECT_EQ(input.token_ids.template item<int32_t>(), -1);
  EXPECT_TRUE(input.json_object_states.front().snapshot().token_ids.empty());
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, EmptyStepClearsPriorContinuation) {
  OverlapTestWorker worker({token_output(17), std::nullopt, token_output(23)});
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  TypeParam empty;
  empty.token_ids = torch::empty({0}, torch::kInt32);
  auto idle = worker.step_async(empty);
  EXPECT_FALSE(std::move(idle).get().has_value());
  EXPECT_FALSE(
      worker.get_last_step_result().sample_output.next_tokens.defined());

  auto next = worker.step_async(
      step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1));
  ASSERT_TRUE(std::move(next).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 0);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), -1);
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, AddsWarmupAndJsonErrorsToBothResultPaths) {
  ForwardOutput produced = token_output(17);
  produced.json_object_errors.emplace_back(
      JsonObjectOutputError{"model#0", "model error"});
  OverlapTestWorker worker({std::move(produced)});
  auto input = step_input<TypeParam>(/*decode=*/false);
  input.input_params.meta.is_graph_warmup = true;
  input.json_object_errors.emplace_back(
      JsonObjectOutputError{"request#0", "input error"});

  auto step = worker.step_async(input);
  const auto returned = std::move(step).get();
  ASSERT_TRUE(returned.has_value());
  EXPECT_TRUE(returned->is_graph_warmup);
  ASSERT_EQ(returned->json_object_errors.size(), 2U);
  EXPECT_EQ(returned->json_object_errors[0].message, "model error");
  EXPECT_EQ(returned->json_object_errors[1].message, "input error");

  const ForwardOutput published = worker.get_last_step_result();
  EXPECT_TRUE(published.is_graph_warmup);
  ASSERT_EQ(published.json_object_errors.size(), 2U);
  EXPECT_EQ(published.json_object_errors[0].message, "model error");
  EXPECT_EQ(published.json_object_errors[1].message, "input error");
}

TYPED_TEST(ScheduleOverlapTest, ConstrainedDecodeRejectsUnrelatedPriorResult) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  OverlapTestWorker worker({token_output(0), token_output(1)});
  auto first = worker.step_async(step_input<TypeParam>(
      /*decode=*/false, /*token_id=*/7, /*request_id=*/"previous"));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(
      /*decode=*/true, /*token_id=*/-1, /*request_id=*/"current");
  input.json_object_states = {grammar.initial_state()};
  input.sample_prior_output_rows = {0};
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(worker.continuation_count(), 0);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 0);
  EXPECT_FALSE(worker.executed_grammar_initialized());
  ASSERT_EQ(output->json_object_errors.size(), 1U);
  EXPECT_EQ(output->json_object_errors.front().sample_sequence_id, "current#0");
  EXPECT_NE(output->json_object_errors.front().message.find("missing"),
            std::string::npos);
  ASSERT_EQ(worker.get_last_step_result().json_object_errors.size(), 1U);
}

TYPED_TEST(ScheduleOverlapTest, ConstrainedContinuationAdvancesGrammar) {
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  OverlapTestWorker worker({token_output(0), token_output(1)});
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false));
  ASSERT_TRUE(std::move(first).get().has_value());
  worker.get_last_step_result();

  auto input = step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1);
  input.json_object_states = {grammar.initial_state()};
  input.sample_prior_output_rows = {0};
  auto step = worker.step_async(input);
  const auto output = std::move(step).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->json_object_errors.empty());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_grammar_tokens(), std::vector<int32_t>({0}));
  EXPECT_TRUE(input.json_object_states.front().snapshot().token_ids.empty());
  worker.get_last_step_result();
}

TYPED_TEST(ScheduleOverlapTest, NonDriverContinuesWithoutHostConsumption) {
  OverlapTestWorker worker({token_output(17), token_output(23)},
                           /*driver=*/false);
  auto first = worker.step_async(step_input<TypeParam>(/*decode=*/false))
                   .toUnsafeFuture();
  first.wait(2s);
  const bool first_ready = first.isReady();
  if (!first_ready) {
    worker.begin_shutdown();
  }
  ASSERT_TRUE(first_ready);
  ASSERT_TRUE(std::move(first).get().has_value());

  auto second =
      worker.step_async(step_input<TypeParam>(/*decode=*/true, /*token_id=*/-1))
          .toUnsafeFuture();
  second.wait(2s);
  const bool second_ready = second.isReady();
  if (!second_ready) {
    worker.begin_shutdown();
  }
  EXPECT_TRUE(second_ready);
  ASSERT_TRUE(std::move(second).get().has_value());
  EXPECT_EQ(worker.continuation_count(), 1);
  EXPECT_EQ(worker.executed_tokens().template item<int32_t>(), 17);
}

}  // namespace
}  // namespace xllm
