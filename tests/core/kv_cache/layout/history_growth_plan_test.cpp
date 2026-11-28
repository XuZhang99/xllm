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

#include "core/kv_cache/layout/history_growth_plan.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {

TEST(HistoryGrowthPlanTest, PartialTailRequiresAnotherBlock) {
  const HistoryGrowthPlan plan(/*token_stride=*/16,
                               /*held_blocks=*/2,
                               /*target_tokens=*/33);
  EXPECT_EQ(plan.required_blocks(), 3u);
  EXPECT_EQ(plan.additional_blocks(), 1u);
  EXPECT_FALSE(plan.is_satisfied_by(/*appended_blocks=*/0));
  EXPECT_TRUE(plan.is_satisfied_by(/*appended_blocks=*/1));
}

TEST(HistoryGrowthPlanTest, SmallerTargetsKeepExistingHistory) {
  for (const size_t tokens : {size_t{0}, size_t{16}, size_t{32}}) {
    const HistoryGrowthPlan plan(/*token_stride=*/16,
                                 /*held_blocks=*/2,
                                 /*target_tokens=*/tokens);
    EXPECT_EQ(plan.additional_blocks(), 0u);
    EXPECT_TRUE(plan.is_satisfied_by(/*appended_blocks=*/0));
  }
}

TEST(HistoryGrowthPlanTest, LargeTokenCountsDoNotWrapCoverage) {
  constexpr size_t kMaxTokens = std::numeric_limits<size_t>::max();
  const HistoryGrowthPlan plan(/*token_stride=*/2,
                               /*held_blocks=*/0,
                               /*target_tokens=*/kMaxTokens);
  EXPECT_EQ(plan.required_blocks(), kMaxTokens / 2 + 1);
  EXPECT_FALSE(plan.is_satisfied_by(kMaxTokens / 2));
  EXPECT_TRUE(plan.is_satisfied_by(kMaxTokens / 2 + 1));
}

TEST(HistoryGrowthPlanTest, RejectsZeroTokenStride) {
  EXPECT_DEATH((HistoryGrowthPlan(/*token_stride=*/0,
                                  /*held_blocks=*/0,
                                  /*target_tokens=*/1)),
               "token_stride");
}

}  // namespace
}  // namespace xllm
