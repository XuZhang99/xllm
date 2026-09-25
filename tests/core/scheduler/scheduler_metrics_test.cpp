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

#include "scheduler/scheduler_metrics.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"

namespace xllm {
namespace {

class FakeEngine final : public Engine {
 public:
  FakeEngine() {
    BlockManagerPool::Options options;
    options.num_blocks(8).block_size(2).enable_prefix_cache(false);
    block_manager_ = std::make_unique<BlockManagerPool>(options, /*dp_size=*/1);
  }

  ForwardOutput step(std::vector<Batch>& /*batch*/) override { return {}; }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {}

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

 private:
  std::unique_ptr<BlockManagerPool> block_manager_;
};

std::shared_ptr<Request> make_request() {
  const std::vector<int32_t> prompt_token_ids{1, 2, 3, 4};
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(64);
  stopping_checker.set_ignore_eos(true);
  RequestState state("prompt",
                     prompt_token_ids,
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     prompt_token_ids.size() + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/false,
                     /*enable_schedule_overlap=*/true,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  return std::make_shared<Request>(
      "req", "x-request-id", "x-request-time", std::move(state));
}

}  // namespace

TEST(SchedulerMetricsTest, CollectsAndDrainsLatencySamples) {
  FakeEngine engine;
  SchedulerMetrics metrics(/*engine=*/&engine,
                           /*kv_cache_manager=*/engine.block_manager_pool(),
                           /*dp_size=*/1,
                           /*num_speculative_tokens=*/0,
                           /*collect_recent_latency=*/true);
  std::shared_ptr<Request> request = make_request();
  Sequence* sequence = request->sequences().front().get();
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(-1));
  const absl::Time start = absl::Now() - absl::Seconds(5);
  sequence->tbt_microseconds(start);
  std::vector<Sequence*> sequences = {sequence};
  std::vector<int64_t> ttft;
  std::vector<int64_t> tbt;

  sequence->update_last_step_token(Token(10), /*token_offset=*/0);
  metrics.update_token_latency_metrics(sequences);
  metrics.get_latency_metrics(ttft, tbt);
  ASSERT_EQ(ttft.size(), 1u);
  EXPECT_GE(ttft.front(), 1000);
  EXPECT_TRUE(tbt.empty());

  metrics.get_latency_metrics(ttft, tbt);
  EXPECT_TRUE(ttft.empty());
  EXPECT_TRUE(tbt.empty());

  sequence->append_token(Token(-1));
  sequence->update_last_step_token(Token(11), /*token_offset=*/0);
  metrics.update_token_latency_metrics(sequences);
  metrics.get_latency_metrics(ttft, tbt);
  EXPECT_TRUE(ttft.empty());
  ASSERT_EQ(tbt.size(), 1u);

  metrics.get_latency_metrics(ttft, tbt);
  EXPECT_TRUE(ttft.empty());
  EXPECT_TRUE(tbt.empty());
}

TEST(SchedulerMetricsTest, SkipsRecentSamplesWhenCollectionDisabled) {
  FakeEngine engine;
  SchedulerMetrics metrics(/*engine=*/&engine,
                           /*kv_cache_manager=*/engine.block_manager_pool(),
                           /*dp_size=*/1,
                           /*num_speculative_tokens=*/0,
                           /*collect_recent_latency=*/false);
  std::shared_ptr<Request> request = make_request();
  Sequence* sequence = request->sequences().front().get();
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(-1));
  sequence->tbt_microseconds(absl::Now() - absl::Seconds(5));
  sequence->update_last_step_token(Token(10), /*token_offset=*/0);
  std::vector<Sequence*> sequences = {sequence};

  metrics.update_token_latency_metrics(sequences);

  std::vector<int64_t> ttft;
  std::vector<int64_t> tbt;
  metrics.get_latency_metrics(ttft, tbt);
  EXPECT_TRUE(ttft.empty());
  EXPECT_TRUE(tbt.empty());
  EXPECT_GT(sequence->time_to_first_token_latency_seconds(), 0.0);
}

TEST(SchedulerMetricsTest, AmortizedTokenLatencyRoundsHalfUp) {
  // Amortized per-token latency is round(latency / n) via (latency + n/2) / n.
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(100, 4), 25);
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(101, 4), 25);
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(102, 4), 26);
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(50, 5), 10);
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(53, 5), 11);
  // With a single committed token amortized latency equals the raw latency.
  EXPECT_EQ(SchedulerMetrics::amortized_token_latency(37, 1), 37);
}

}  // namespace xllm
