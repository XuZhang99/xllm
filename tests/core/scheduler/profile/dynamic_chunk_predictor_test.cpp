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

#include "core/scheduler/profile/dynamic_chunk_predictor.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {
std::vector<std::pair<int32_t, double>> measurements(double a,
                                                     double b,
                                                     double c) {
  std::vector<std::pair<int32_t, double>> samples;
  samples.reserve(16);
  for (int32_t length = 256; length <= 4096; length += 256) {
    samples.emplace_back(length, a * length * length + b * length + c);
  }
  return samples;
}

TEST(DynamicChunkPredictorTest, EqualizesIncrementalLatency) {
  DynamicChunkPredictor predictor(4096, 64, 1.0, 64);
  constexpr double kA = 0.00001;
  constexpr double kB = 0.01;
  ASSERT_TRUE(predictor.fit(measurements(kA, kB, 12.0)));
  EXPECT_NEAR(
      predictor.target_latency_ms(), kA * 4096 * 4096 + kB * 4096, 1e-8);
  size_t previous = 4096;
  for (size_t history : {0, 4096, 16384, 65536, 1048576}) {
    const size_t chunk = predictor.predict(history, 2097152, 8192);
    EXPECT_LE(chunk, previous);
    EXPECT_EQ(chunk % 64, 0u);
    const double latency = kA * chunk * (chunk + 2.0 * history) + kB * chunk;
    if (chunk > 64) {
      EXPECT_LE(latency, predictor.target_latency_ms() + 1e-8);
      const double next = chunk + 64;
      EXPECT_GT(kA * next * (next + 2.0 * history) + kB * next,
                predictor.target_latency_ms());
    }
    previous = chunk;
  }
}

TEST(DynamicChunkPredictorTest, CapsBudgetsAndAllowsFinalPartialBlock) {
  DynamicChunkPredictor predictor(4096, 256, 1.0, 256);
  ASSERT_TRUE(predictor.fit(measurements(0.00001, 0.01, 12.0)));
  EXPECT_EQ(predictor.predict(0, 10000, 700), 512u);
  EXPECT_EQ(predictor.predict(0, 10000, 255), 0u);
  EXPECT_EQ(predictor.predict(0, 31, 1000), 31u);
  EXPECT_EQ(predictor.predict(4096, 129, 1000), 129u);
  EXPECT_EQ(predictor.predict(4096, 0, 1000), 0u);
  EXPECT_EQ(predictor.predict(4096, 1000, 0), 0u);
  EXPECT_EQ(predictor.predict(1048576, 10000, 4096), 256u);
}

TEST(DynamicChunkPredictorTest, SmoothsAndHandlesLinearCost) {
  DynamicChunkPredictor full(4096, 256, 1.0, 64);
  DynamicChunkPredictor smooth(4096, 256, 0.5, 64);
  ASSERT_TRUE(full.fit(measurements(0.00001, 0.01, 12.0)));
  ASSERT_TRUE(smooth.fit(measurements(0.00001, 0.01, 12.0)));
  EXPECT_GT(smooth.predict(65536, 10000, 4096),
            full.predict(65536, 10000, 4096));
  ASSERT_TRUE(full.fit(measurements(0.0, 0.01, 12.0)));
  EXPECT_GE(full.predict(1048576, 10000, 4096), 4032u);
}

TEST(DynamicChunkPredictorTest, RejectsBadDataAndResetsReadiness) {
  DynamicChunkPredictor predictor(4096, 256, 1.0, 64);
  EXPECT_FALSE(predictor.fit({}));
  EXPECT_FALSE(
      predictor.fit(std::vector<std::pair<int32_t, double>>(16, {64, 1.0})));
  EXPECT_FALSE(predictor.fit(measurements(0, 0, 12)));
  ASSERT_TRUE(predictor.fit(measurements(0.00001, 0.01, 12)));
  auto invalid = measurements(0.00001, 0.01, 12);
  invalid[4].second = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(predictor.fit(invalid));
  EXPECT_FALSE(predictor.ready());
  EXPECT_EQ(predictor.predict(100000, 77, 1024), 77u);
  EXPECT_EQ(predictor.predict(100000, 10000, 1024), 1024u);
}

TEST(DynamicChunkPredictorTest, ValidatesParameters) {
  EXPECT_DEATH(DynamicChunkPredictor(4096, 0, 1.0, 64), "min_chunk");
  EXPECT_DEATH(DynamicChunkPredictor(4096, 8192, 1.0, 64), "min_chunk");
  EXPECT_DEATH(DynamicChunkPredictor(4096, 256, 0.0, 64), "smooth_factor");
  EXPECT_DEATH(DynamicChunkPredictor(32, 16, 1.0, 64), "base_chunk");
}
}  // namespace
}  // namespace xllm
