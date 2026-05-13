/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <filesystem>
#include <string_view>

#include "core/framework/config/xllm_config.h"

namespace xllm {
namespace {

inline constexpr std::string_view kInlineConfig = R"json({
  "block_size": 16,
  "max_memory_utilization": 0.5,
  "enable_prefix_cache": false,
  "max_tokens_per_batch": 8192,
  "max_seqs_per_batch": 64
})json";

TEST(XllmConfigJsonTest, MissingFileUsesFlagDefaults) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_missing.json";
  std::filesystem::remove(config_path);

  const XllmConfig config = XllmConfig::from_json_file(config_path.string());

  EXPECT_EQ(config.kv_cache_config().block_size(), 128);
  EXPECT_DOUBLE_EQ(config.kv_cache_config().max_memory_utilization(), 0.8);
  EXPECT_EQ(config.scheduler_config().max_tokens_per_batch(), 10240);
  EXPECT_EQ(config.scheduler_config().max_seqs_per_batch(), 1024);
}

TEST(XllmConfigJsonTest, LoadsFlatJsonFile) {
  const std::filesystem::path config_path =
      std::filesystem::path(__FILE__).parent_path() / "qwen3/config.json";

  const XllmConfig config = XllmConfig::from_json_file(config_path.string());

  EXPECT_EQ(config.kv_cache_config().block_size(), 32);
  EXPECT_DOUBLE_EQ(config.kv_cache_config().max_memory_utilization(), 0.6);
  EXPECT_TRUE(config.kv_cache_config().enable_prefix_cache());
  EXPECT_EQ(config.scheduler_config().max_tokens_per_batch(), 4096);
  EXPECT_EQ(config.scheduler_config().max_seqs_per_batch(), 128);
  EXPECT_EQ(config.scheduler_config().max_tokens_per_chunk_for_prefill(), 2048);
}

TEST(XllmConfigJsonTest, LoadsInlineConfigJson) {
  const XllmConfig config = XllmConfig::from_json_string(kInlineConfig);

  EXPECT_EQ(config.kv_cache_config().block_size(), 16);
  EXPECT_DOUBLE_EQ(config.kv_cache_config().max_memory_utilization(), 0.5);
  EXPECT_FALSE(config.kv_cache_config().enable_prefix_cache());
  EXPECT_EQ(config.scheduler_config().max_tokens_per_batch(), 8192);
  EXPECT_EQ(config.scheduler_config().max_seqs_per_batch(), 64);

  EXPECT_EQ(config.scheduler_config().max_decode_token_per_sequence(), 256);
  EXPECT_EQ(config.kv_cache_config().kv_cache_dtype(), "auto");
}

}  // namespace
}  // namespace xllm
