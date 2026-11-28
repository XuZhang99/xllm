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

#include "core/distributed_runtime/worker_step_dispatch.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/runtime/vlm_forward_params.h"

namespace xllm {
namespace {

RawForwardOutput output_for_worker(int64_t worker_id) {
  RawForwardOutput output;
  output.prepared_token = worker_id;
  output.expert_load_data = {worker_id * 7};
  return output;
}

class RecordingStepWorker final : public WorkerClient {
 public:
  explicit RecordingStepWorker(int64_t worker_id) : worker_id_(worker_id) {}

  folly::SemiFuture<std::optional<RawForwardOutput>> step_remote_async(
      const LlmForwardInput& input,
      std::shared_ptr<ForwardInputEncoding> encoding) override {
    ++llm_step_calls_;
    return record_step(input, std::move(encoding));
  }

  folly::SemiFuture<std::optional<RawForwardOutput>> step_remote_async(
      const VlmForwardInput& input,
      std::shared_ptr<ForwardInputEncoding> encoding) override {
    ++vlm_step_calls_;
    return record_step(input, std::move(encoding));
  }

  folly::SemiFuture<std::optional<RawForwardOutput>>
  get_last_llm_step_result_async() override {
    ++llm_result_calls_;
    return response();
  }

  folly::SemiFuture<std::optional<RawForwardOutput>>
  get_last_vlm_step_result_async() override {
    ++vlm_result_calls_;
    return response();
  }

  int32_t llm_step_calls_ = 0;
  int32_t vlm_step_calls_ = 0;
  int32_t llm_result_calls_ = 0;
  int32_t vlm_result_calls_ = 0;
  const void* seen_input_ = nullptr;
  int64_t seen_tokens_ = -1;
  std::shared_ptr<ForwardInputEncoding> encoding_;
  bool rpc_failure_ = false;
  bool absent_output_ = false;
  std::shared_ptr<folly::Promise<std::optional<RawForwardOutput>>> pending_;
  std::function<void()> after_dispatch_;

 private:
  template <typename Input>
  folly::SemiFuture<std::optional<RawForwardOutput>> record_step(
      const Input& input,
      std::shared_ptr<ForwardInputEncoding> encoding) {
    seen_input_ = &input;
    seen_tokens_ = input.token_ids.defined() ? input.token_ids.numel() : 0;
    encoding_ = std::move(encoding);
    auto future = response();
    if (after_dispatch_) {
      after_dispatch_();
    }
    return future;
  }

  folly::SemiFuture<std::optional<RawForwardOutput>> response() {
    if (pending_) {
      return pending_->getSemiFuture();
    }
    folly::Promise<std::optional<RawForwardOutput>> promise;
    auto future = promise.getSemiFuture();
    if (rpc_failure_) {
      promise.setException(std::runtime_error("worker RPC error"));
    } else if (absent_output_) {
      promise.setValue(std::nullopt);
    } else {
      promise.setValue(output_for_worker(worker_id_));
    }
    return future;
  }

  int64_t worker_id_;
};

template <typename Input>
class WorkerStepDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }

  void make_workers(uint32_t count) {
    workers_.reserve(count);
    recordings_.reserve(count);
    for (uint32_t rank = 0; rank < count; ++rank) {
      auto worker = std::make_shared<RecordingStepWorker>(rank);
      workers_.emplace_back(worker);
      recordings_.emplace_back(std::move(worker));
    }
  }

  static auto result_getter() {
    if constexpr (std::is_same_v<Input, LlmForwardInput>) {
      return &WorkerClient::get_last_llm_step_result_async;
    } else {
      return &WorkerClient::get_last_vlm_step_result_async;
    }
  }

  void expect_calls(uint32_t rank, int32_t steps, int32_t results) {
    const auto& worker = recordings_[rank];
    EXPECT_EQ(worker->llm_step_calls_,
              (std::is_same_v<Input, LlmForwardInput> ? steps : 0));
    EXPECT_EQ(worker->vlm_step_calls_,
              (std::is_same_v<Input, VlmForwardInput> ? steps : 0));
    EXPECT_EQ(worker->llm_result_calls_,
              (std::is_same_v<Input, LlmForwardInput> ? results : 0));
    EXPECT_EQ(worker->vlm_result_calls_,
              (std::is_same_v<Input, VlmForwardInput> ? results : 0));
  }

  std::vector<std::shared_ptr<WorkerClient>> workers_;
  std::vector<std::shared_ptr<RecordingStepWorker>> recordings_;
};

using StepInputs = ::testing::Types<LlmForwardInput, VlmForwardInput>;
TYPED_TEST_SUITE(WorkerStepDispatchTest, StepInputs);

TYPED_TEST(WorkerStepDispatchTest,
           SharesFullDpInputAndEncodingAcrossTpCpWorkers) {
  this->make_workers(/*count=*/6);
  std::vector<TypeParam> inputs(2);
  inputs[0].token_ids = torch::tensor({11, 12, 13}, torch::kInt32);
  inputs[1].token_ids = torch::tensor({21, 22, 23, 24}, torch::kInt32);
  auto results =
      dispatch_worker_step(this->workers_, inputs, /*workers_per_dp_rank=*/3);
  ASSERT_EQ(results.size(), 6u);
  for (uint32_t rank = 0; rank < 6; ++rank) {
    const uint32_t dp_rank = rank / 3;
    this->expect_calls(rank, /*steps=*/1, /*results=*/0);
    EXPECT_EQ(this->recordings_[rank]->seen_input_, &inputs[dp_rank]);
    EXPECT_EQ(this->recordings_[rank]->seen_tokens_,
              inputs[dp_rank].token_ids.numel());
    EXPECT_NE(this->recordings_[rank]->encoding_, nullptr);
    EXPECT_EQ(this->recordings_[rank]->encoding_,
              this->recordings_[dp_rank * 3]->encoding_);
    ASSERT_TRUE(results[rank].hasValue());
    ASSERT_TRUE(results[rank].value().has_value());
    EXPECT_EQ(results[rank].value()->prepared_token, rank);
  }
  EXPECT_NE(this->recordings_[0]->encoding_, this->recordings_[3]->encoding_);
}

TYPED_TEST(WorkerStepDispatchTest, AvoidsEncodingCacheForSingleWorkerGroups) {
  this->make_workers(/*count=*/2);
  std::vector<TypeParam> inputs(2);
  const auto results =
      dispatch_worker_step(this->workers_, inputs, /*workers_per_dp_rank=*/1);
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(this->recordings_[0]->encoding_, nullptr);
  EXPECT_EQ(this->recordings_[1]->encoding_, nullptr);
}

TYPED_TEST(WorkerStepDispatchTest,
           DispatchesEmptyRanksAndPreservesAbsentOutputs) {
  this->make_workers(/*count=*/4);
  std::vector<TypeParam> inputs(2);
  inputs[0].token_ids = torch::tensor({11, 12}, torch::kInt32);
  this->recordings_[2]->absent_output_ = true;
  this->recordings_[3]->absent_output_ = true;
  auto results =
      dispatch_worker_step(this->workers_, inputs, /*workers_per_dp_rank=*/2);
  ASSERT_EQ(results.size(), 4u);
  for (uint32_t rank = 2; rank < 4; ++rank) {
    this->expect_calls(rank, /*steps=*/1, /*results=*/0);
    EXPECT_EQ(this->recordings_[rank]->seen_input_, &inputs[1]);
    EXPECT_EQ(this->recordings_[rank]->seen_tokens_, 0);
    EXPECT_TRUE(results[rank].hasValue());
    EXPECT_FALSE(results[rank].value().has_value());
  }
}

TYPED_TEST(WorkerStepDispatchTest, RetainsWorkerOrderForReversedRpcCompletion) {
  this->make_workers(/*count=*/4);
  for (const auto& worker : this->recordings_) {
    worker->pending_ =
        std::make_shared<folly::Promise<std::optional<RawForwardOutput>>>();
  }
  this->recordings_.back()->after_dispatch_ = [&]() {
    for (int32_t rank = 3; rank >= 0; --rank) {
      this->recordings_[rank]->pending_->setValue(output_for_worker(rank));
    }
  };
  std::vector<TypeParam> inputs(2);
  auto results =
      dispatch_worker_step(this->workers_, inputs, /*workers_per_dp_rank=*/2);
  ASSERT_EQ(results.size(), 4u);
  for (uint32_t rank = 0; rank < 4; ++rank) {
    ASSERT_TRUE(results[rank].value().has_value());
    EXPECT_EQ(results[rank].value()->prepared_token, rank);
    EXPECT_EQ(results[rank].value()->expert_load_data,
              (std::vector<int64_t>{rank * 7}));
  }
}

TYPED_TEST(WorkerStepDispatchTest, RejectsInvalidDispatchTopology) {
  this->make_workers(/*count=*/3);
  std::vector<TypeParam> inputs(2);
  EXPECT_DEATH(dispatch_worker_step(this->workers_,
                                    inputs,
                                    /*workers_per_dp_rank=*/2),
               "Check failed");
  EXPECT_DEATH(dispatch_worker_step(this->workers_,
                                    inputs,
                                    /*workers_per_dp_rank=*/0),
               "Check failed");
}

TYPED_TEST(WorkerStepDispatchTest, RejectsRpcFailureOnNonDriverWorker) {
  this->make_workers(/*count=*/4);
  this->recordings_[1]->rpc_failure_ = true;
  std::vector<TypeParam> inputs(2);
  EXPECT_DEATH(dispatch_worker_step(this->workers_,
                                    inputs,
                                    /*workers_per_dp_rank=*/2),
               "Failed to execute model, worker RPC failed");
}

TYPED_TEST(WorkerStepDispatchTest, CollectsDriversWithTheFullTpCpStride) {
  this->make_workers(/*count=*/6);
  this->recordings_[1]->rpc_failure_ = true;
  this->recordings_[2]->rpc_failure_ = true;
  this->recordings_[4]->rpc_failure_ = true;
  this->recordings_[5]->rpc_failure_ = true;
  auto results = collect_worker_step_results(
      this->workers_, /*stride=*/3, TestFixture::result_getter());
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].value()->prepared_token, 0);
  EXPECT_EQ(results[1].value()->prepared_token, 3);
  for (uint32_t rank = 0; rank < 6; ++rank) {
    this->expect_calls(rank,
                       /*steps=*/0,
                       /*results=*/rank % 3 == 0 ? 1 : 0);
  }
}

TYPED_TEST(WorkerStepDispatchTest,
           CollectsAllShardOutputsWithoutConsumingThem) {
  this->make_workers(/*count=*/4);
  auto results = collect_worker_step_results(
      this->workers_, /*stride=*/1, TestFixture::result_getter());
  ASSERT_EQ(results.size(), 4u);
  for (uint32_t rank = 0; rank < 4; ++rank) {
    this->expect_calls(rank, /*steps=*/0, /*results=*/1);
    EXPECT_EQ(results[rank].value()->prepared_token, rank);
    EXPECT_EQ(results[rank].value()->expert_load_data,
              (std::vector<int64_t>{rank * 7}));
  }
}

TYPED_TEST(WorkerStepDispatchTest, RejectsMissingCollectedDriverOutput) {
  this->make_workers(/*count=*/4);
  this->recordings_[2]->absent_output_ = true;
  EXPECT_DEATH(collect_worker_step_results(
                   this->workers_, /*stride=*/2, TestFixture::result_getter()),
               "Failed to get last step results, result has no value");
}

TYPED_TEST(WorkerStepDispatchTest, RejectsCollectedDriverRpcFailure) {
  this->make_workers(/*count=*/4);
  this->recordings_[2]->rpc_failure_ = true;
  EXPECT_DEATH(collect_worker_step_results(
                   this->workers_, /*stride=*/2, TestFixture::result_getter()),
               "Failed to get last step results, result has no value");
}

TYPED_TEST(WorkerStepDispatchTest, RejectsInvalidCollectionStride) {
  this->make_workers(/*count=*/6);
  EXPECT_DEATH(collect_worker_step_results(
                   this->workers_, /*stride=*/0, TestFixture::result_getter()),
               "Check failed");
  EXPECT_DEATH(collect_worker_step_results(
                   this->workers_, /*stride=*/4, TestFixture::result_getter()),
               "Check failed");
}

TYPED_TEST(WorkerStepDispatchTest, HasNoOutputsOnAnEngineWithoutWorkers) {
  auto results = collect_worker_step_results(
      this->workers_, /*stride=*/0, TestFixture::result_getter());
  EXPECT_TRUE(results.empty());
}

}  // namespace
}  // namespace xllm
