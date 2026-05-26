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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"

namespace xllm {
namespace {

inline constexpr std::string_view kInlineConfig = R"json({
  "block_size": 16,
  "max_memory_utilization": 0.5,
  "enable_prefix_cache": false,
  "max_tokens_per_batch": 8192,
  "max_seqs_per_batch": 64
})json";

inline constexpr std::string_view kUpdatedConfig = R"json({
  "block_size": 32,
  "max_tokens_per_batch": 4096
})json";

inline constexpr std::string_view kMalformedConfig = R"json({
  "block_size":
})json";

inline constexpr std::string_view kRankBoundConfig = R"json({
  "port": [29000, 29001],
  "devices": ["cuda:0", "cuda:1"],
  "draft_devices": ["cuda:0", "cuda:1"],
  "node_rank": [0, 1],
  "nnodes": 2
})json";

class ConfigJsonFileFlagGuard final {
 public:
  explicit ConfigJsonFileFlagGuard(const std::string& config_json_file)
      : old_config_json_file_(FLAGS_config_json_file) {
    FLAGS_config_json_file = config_json_file;
  }

  ~ConfigJsonFileFlagGuard() { FLAGS_config_json_file = old_config_json_file_; }

 private:
  std::string old_config_json_file_;
};

class NodeRankFlagGuard final {
 public:
  explicit NodeRankFlagGuard(int32_t node_rank)
      : old_node_rank_(FLAGS_node_rank) {
    FLAGS_node_rank = node_rank;
  }

  ~NodeRankFlagGuard() { FLAGS_node_rank = old_node_rank_; }

 private:
  int32_t old_node_rank_;
};

class DumpConfigJsonFlagGuard final {
 public:
  explicit DumpConfigJsonFlagGuard(const std::string& dump_config_json_file)
      : old_enable_dump_config_json_(FLAGS_enable_dump_config_json),
        old_dump_config_json_file_(FLAGS_dump_config_json_file) {
    FLAGS_dump_config_json_file = dump_config_json_file;
  }

  ~DumpConfigJsonFlagGuard() {
    FLAGS_enable_dump_config_json = old_enable_dump_config_json_;
    FLAGS_dump_config_json_file = old_dump_config_json_file_;
  }

 private:
  bool old_enable_dump_config_json_;
  std::string old_dump_config_json_file_;
};

class RankBoundStartupConfigGuard final {
 public:
  RankBoundStartupConfigGuard()
      : service_config_(ServiceConfig::get_instance()),
        model_config_(ModelConfig::get_instance()),
        speculative_config_(SpeculativeConfig::get_instance()),
        distributed_config_(DistributedConfig::get_instance()),
        old_port_(service_config_.port()),
        old_devices_(model_config_.devices()),
        old_draft_devices_(speculative_config_.draft_devices()),
        old_nnodes_(distributed_config_.nnodes()),
        old_node_rank_(distributed_config_.node_rank()) {}

  ~RankBoundStartupConfigGuard() {
    service_config_.port(old_port_);
    model_config_.devices(old_devices_);
    speculative_config_.draft_devices(old_draft_devices_);
    distributed_config_.nnodes(old_nnodes_).node_rank(old_node_rank_);
  }

 private:
  ServiceConfig& service_config_;
  ModelConfig& model_config_;
  SpeculativeConfig& speculative_config_;
  DistributedConfig& distributed_config_;
  int32_t old_port_;
  std::string old_devices_;
  std::string old_draft_devices_;
  int32_t old_nnodes_;
  int32_t old_node_rank_;
};

class StartupConfigGuard final {
 public:
  StartupConfigGuard()
      : kv_cache_config_(KVCacheConfig::get_instance()),
        scheduler_config_(SchedulerConfig::get_instance()),
        old_block_size_(kv_cache_config_.block_size()),
        old_enable_prefix_cache_(kv_cache_config_.enable_prefix_cache()),
        old_max_tokens_per_batch_(scheduler_config_.max_tokens_per_batch()),
        old_enable_chunked_prefill_(
            scheduler_config_.enable_chunked_prefill()) {}

  ~StartupConfigGuard() {
    kv_cache_config_.block_size(old_block_size_)
        .enable_prefix_cache(old_enable_prefix_cache_);
    scheduler_config_.max_tokens_per_batch(old_max_tokens_per_batch_)
        .enable_chunked_prefill(old_enable_chunked_prefill_);
  }

 private:
  KVCacheConfig& kv_cache_config_;
  SchedulerConfig& scheduler_config_;
  int32_t old_block_size_;
  bool old_enable_prefix_cache_;
  int32_t old_max_tokens_per_batch_;
  bool old_enable_chunked_prefill_;
};

void write_config_file(const std::filesystem::path& config_path,
                       std::string_view config_json) {
  std::ofstream config_file(config_path);
  config_file << config_json;
}

nlohmann::ordered_json read_json_file(const std::filesystem::path& file_path) {
  std::ifstream input_file(file_path);
  EXPECT_TRUE(input_file.is_open()) << file_path;
  return nlohmann::ordered_json::parse(input_file);
}

std::filesystem::path config_test_file_path() {
  const std::filesystem::path source_config_path =
      std::filesystem::path(__FILE__).parent_path() / "config_test.json";
  if (std::filesystem::exists(source_config_path)) {
    return source_config_path;
  }

  const std::filesystem::path copied_config_path = "config_test.json";
  if (std::filesystem::exists(copied_config_path)) {
    return copied_config_path;
  }

  return std::filesystem::path("tests/core/framework/config/config_test.json");
}

TEST(ConfigJsonTest, FromJsonUsesParsedOverrides) {
  const JsonReader json = config::parse_json_string(kInlineConfig);

  KVCacheConfig kv_cache_config;
  kv_cache_config.from_flags();
  kv_cache_config.from_json(json);

  SchedulerConfig scheduler_config;
  scheduler_config.from_flags();
  scheduler_config.from_json(json);

  EXPECT_EQ(kv_cache_config.block_size(), 16);
  EXPECT_DOUBLE_EQ(kv_cache_config.max_memory_utilization(), 0.5);
  EXPECT_FALSE(kv_cache_config.enable_prefix_cache());
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 8192);
  EXPECT_EQ(scheduler_config.max_seqs_per_batch(), 64);

  EXPECT_EQ(kv_cache_config.kv_cache_dtype(), "auto");
  EXPECT_EQ(scheduler_config.max_decode_token_per_sequence(), 256);
}

TEST(ConfigJsonTest, LoadJsonFileReadsConfigFixture) {
  const std::filesystem::path config_path = config_test_file_path();
  ASSERT_TRUE(std::filesystem::exists(config_path)) << config_path;

  const JsonReader json = config::load_json_file(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.from_flags();
  kv_cache_config.from_json(json);

  SchedulerConfig scheduler_config;
  scheduler_config.from_flags();
  scheduler_config.from_json(json);

  EXPECT_EQ(kv_cache_config.block_size(), 24);
  EXPECT_EQ(kv_cache_config.max_cache_size(), 1048576);
  EXPECT_DOUBLE_EQ(kv_cache_config.max_memory_utilization(), 0.65);
  EXPECT_EQ(kv_cache_config.kv_cache_dtype(), "int8");
  EXPECT_FALSE(kv_cache_config.enable_prefix_cache());
  EXPECT_EQ(kv_cache_config.xxh3_128bits_seed(), 2048);
  EXPECT_TRUE(kv_cache_config.enable_xtensor());
  EXPECT_EQ(kv_cache_config.phy_page_granularity_size(), 4096);

  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 2048);
  EXPECT_EQ(scheduler_config.max_seqs_per_batch(), 32);
  EXPECT_TRUE(scheduler_config.enable_schedule_overlap());
  EXPECT_DOUBLE_EQ(scheduler_config.prefill_scheduling_memory_usage_threshold(),
                   0.75);
  EXPECT_FALSE(scheduler_config.enable_chunked_prefill());
  EXPECT_EQ(scheduler_config.max_tokens_per_chunk_for_prefill(), 512);
  EXPECT_EQ(scheduler_config.chunked_match_frequency(), 3);
  EXPECT_TRUE(scheduler_config.use_zero_evict());
  EXPECT_EQ(scheduler_config.max_decode_token_per_sequence(), 128);
  EXPECT_EQ(scheduler_config.priority_strategy(), "priority");
  EXPECT_TRUE(scheduler_config.use_mix_scheduler());
  EXPECT_FALSE(scheduler_config.enable_online_preempt_offline());
  EXPECT_DOUBLE_EQ(scheduler_config.aggressive_coeff(), 1.5);
  EXPECT_DOUBLE_EQ(scheduler_config.starve_threshold(), 2.0);
  EXPECT_FALSE(scheduler_config.enable_starve_prevent());
}

TEST(ConfigJsonTest, InitializeLoadsConfigJsonFileFromFlag) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() / "xllm_config_json_test.json";
  write_config_file(config_path, kInlineConfig);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 16);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 8192);

  std::filesystem::remove(config_path);
}

TEST(ConfigJsonTest, InitializeReusesCachedConfigJsonForSameFile) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_cached.json";
  write_config_file(config_path, kInlineConfig);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  write_config_file(config_path, kUpdatedConfig);

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 16);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 8192);

  std::filesystem::remove(config_path);
}

TEST(ConfigJsonTest, MalformedJsonFileKeepsFlagDefaults) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_malformed.json";
  write_config_file(config_path, kMalformedConfig);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 128);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 10240);

  std::filesystem::remove(config_path);
}

TEST(ConfigJsonTest, MissingJsonFileKeepsFlagDefaults) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_missing.json";
  std::filesystem::remove(config_path);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 128);
  EXPECT_DOUBLE_EQ(kv_cache_config.max_memory_utilization(), 0.8);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 10240);
  EXPECT_EQ(scheduler_config.max_seqs_per_batch(), 1024);
}

TEST(ConfigJsonTest, InitializeReadsRankBoundValuesFromArray) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_rank_bound.json";
  write_config_file(config_path, kRankBoundConfig);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());
  NodeRankFlagGuard node_rank_guard(1);

  ServiceConfig service_config;
  service_config.initialize();

  ModelConfig model_config;
  model_config.initialize();

  SpeculativeConfig speculative_config;
  speculative_config.initialize();

  DistributedConfig distributed_config;
  distributed_config.initialize();

  EXPECT_EQ(service_config.port(), 29001);
  EXPECT_EQ(model_config.devices(), "cuda:1");
  EXPECT_EQ(speculative_config.draft_devices(), "cuda:1");
  EXPECT_EQ(distributed_config.nnodes(), 2);
  EXPECT_EQ(distributed_config.node_rank(), 1);

  std::filesystem::remove(config_path);
}

TEST(ConfigJsonTest, FromJsonReadsRankZeroWhenNodeRankExplicitlySet) {
  GFLAGS_NAMESPACE::FlagSaver flag_saver;
  ASSERT_FALSE(
      GFLAGS_NAMESPACE::SetCommandLineOption("node_rank", "0").empty());
  const JsonReader json = config::parse_json_string(kRankBoundConfig);

  ServiceConfig service_config;
  service_config.from_flags();
  service_config.from_json(json);

  ModelConfig model_config;
  model_config.from_flags();
  model_config.from_json(json);

  DistributedConfig distributed_config;
  distributed_config.from_flags();
  distributed_config.from_json(json);

  EXPECT_EQ(service_config.port(), 29000);
  EXPECT_EQ(model_config.devices(), "cuda:0");
  EXPECT_EQ(distributed_config.node_rank(), 0);
}

TEST(ConfigJsonTest, RankBoundJsonRequiresExplicitNodeRankSelector) {
  GFLAGS_NAMESPACE::FlagSaver flag_saver;
  FLAGS_node_rank = 0;
  const JsonReader json = config::parse_json_string(kRankBoundConfig);

  EXPECT_DEATH(
      {
        ServiceConfig service_config;
        service_config.from_flags();
        service_config.from_json(json);
      },
      "--node_rank");
}

TEST(ConfigJsonTest, DumpStartupConfigSkipsWhenDisabled) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_disabled.json";
  std::filesystem::remove(dump_path);
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  FLAGS_enable_dump_config_json = false;

  config::dump_startup_config();

  EXPECT_FALSE(std::filesystem::exists(dump_path));
}

TEST(ConfigJsonTest, DumpStartupConfigWritesNonDefaultValuesOnly) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_non_default.json";
  std::filesystem::remove(dump_path);
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  StartupConfigGuard startup_config_guard;

  KVCacheConfig::get_instance().block_size(256).enable_prefix_cache(false);
  SchedulerConfig::get_instance()
      .max_tokens_per_batch(2048)
      .enable_chunked_prefill(false);
  FLAGS_enable_dump_config_json = true;

  config::dump_startup_config();

  ASSERT_TRUE(std::filesystem::exists(dump_path));
  const nlohmann::ordered_json config_json = read_json_file(dump_path);
  EXPECT_EQ(config_json.at("block_size").get<int32_t>(), 256);
  EXPECT_FALSE(config_json.at("enable_prefix_cache").get<bool>());
  EXPECT_EQ(config_json.at("max_tokens_per_batch").get<int32_t>(), 2048);
  EXPECT_FALSE(config_json.at("enable_chunked_prefill").get<bool>());

  EXPECT_FALSE(config_json.contains("max_cache_size"));
  EXPECT_FALSE(config_json.contains("kv_cache_dtype"));
  EXPECT_FALSE(config_json.contains("max_seqs_per_batch"));
  EXPECT_FALSE(config_json.contains("priority_strategy"));

  std::filesystem::remove(dump_path);
}

TEST(ConfigJsonTest, DumpStartupConfigMergesRankBoundValuesIntoArrays) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_rank_bound.json";
  std::filesystem::remove(dump_path);
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  StartupConfigGuard startup_config_guard;
  RankBoundStartupConfigGuard rank_bound_startup_config_guard;

  FLAGS_enable_dump_config_json = true;

  DistributedConfig::get_instance().nnodes(2).node_rank(1);
  ServiceConfig::get_instance().port(29001);
  ModelConfig::get_instance().devices("cuda:1");
  SpeculativeConfig::get_instance().draft_devices("cuda:1");
  KVCacheConfig::get_instance().block_size(512);
  config::dump_startup_config();

  DistributedConfig::get_instance().nnodes(2).node_rank(0);
  ServiceConfig::get_instance().port(29000);
  ModelConfig::get_instance().devices("cuda:0");
  SpeculativeConfig::get_instance().draft_devices("cuda:0");
  KVCacheConfig::get_instance().block_size(256);
  config::dump_startup_config();

  ASSERT_TRUE(std::filesystem::exists(dump_path));
  const nlohmann::ordered_json config_json = read_json_file(dump_path);
  ASSERT_TRUE(config_json.at("port").is_array());
  ASSERT_TRUE(config_json.at("devices").is_array());
  ASSERT_TRUE(config_json.at("draft_devices").is_array());
  ASSERT_TRUE(config_json.at("node_rank").is_array());

  EXPECT_EQ(config_json.at("port").at(0).get<int32_t>(), 29000);
  EXPECT_EQ(config_json.at("port").at(1).get<int32_t>(), 29001);
  EXPECT_EQ(config_json.at("devices").at(0).get<std::string>(), "cuda:0");
  EXPECT_EQ(config_json.at("devices").at(1).get<std::string>(), "cuda:1");
  EXPECT_EQ(config_json.at("draft_devices").at(0).get<std::string>(), "cuda:0");
  EXPECT_EQ(config_json.at("draft_devices").at(1).get<std::string>(), "cuda:1");
  EXPECT_EQ(config_json.at("node_rank").at(0).get<int32_t>(), 0);
  EXPECT_EQ(config_json.at("node_rank").at(1).get<int32_t>(), 1);
  EXPECT_EQ(config_json.at("nnodes").get<int32_t>(), 2);
  EXPECT_EQ(config_json.at("block_size").get<int32_t>(), 256);

  std::filesystem::remove(dump_path);
}

TEST(ConfigJsonTest, DumpStartupConfigOmitsRankValuesMatchingRankZero) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_rank_zero_default.json";
  std::filesystem::remove(dump_path);
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  StartupConfigGuard startup_config_guard;
  RankBoundStartupConfigGuard rank_bound_startup_config_guard;

  FLAGS_enable_dump_config_json = true;

  DistributedConfig::get_instance().nnodes(2).node_rank(1);
  ServiceConfig::get_instance().port(29000);
  ModelConfig::get_instance().devices("cuda:0");
  config::dump_startup_config();

  DistributedConfig::get_instance().nnodes(2).node_rank(0);
  ServiceConfig::get_instance().port(29000);
  ModelConfig::get_instance().devices("cuda:0");
  config::dump_startup_config();

  ASSERT_TRUE(std::filesystem::exists(dump_path));
  const nlohmann::ordered_json config_json = read_json_file(dump_path);
  ASSERT_TRUE(config_json.at("port").is_array());
  ASSERT_TRUE(config_json.at("devices").is_array());
  EXPECT_FALSE(config_json.contains("draft_devices"));
  EXPECT_EQ(config_json.at("port").size(), 1);
  EXPECT_EQ(config_json.at("devices").size(), 1);
  EXPECT_EQ(config_json.at("port").at(0).get<int32_t>(), 29000);
  EXPECT_EQ(config_json.at("devices").at(0).get<std::string>(), "cuda:0");

  const JsonReader json = config::parse_json_string(config_json.dump());
  NodeRankFlagGuard node_rank_guard(1);

  ServiceConfig service_config;
  service_config.from_flags();
  service_config.from_json(json);

  ModelConfig model_config;
  model_config.from_flags();
  model_config.from_json(json);

  SpeculativeConfig speculative_config;
  speculative_config.from_flags();
  speculative_config.from_json(json);

  EXPECT_EQ(service_config.port(), 29000);
  EXPECT_EQ(model_config.devices(), "cuda:0");
  EXPECT_EQ(speculative_config.draft_devices(), "npu:0");

  std::filesystem::remove(dump_path);
}

TEST(ConfigJsonTest,
     DumpStartupConfigOmitsDefaultRankValuesAfterRankZeroFirst) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_rank_zero_first_default.json";
  std::filesystem::remove(dump_path);
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  StartupConfigGuard startup_config_guard;
  RankBoundStartupConfigGuard rank_bound_startup_config_guard;

  FLAGS_enable_dump_config_json = true;

  DistributedConfig::get_instance().nnodes(2).node_rank(0);
  ServiceConfig::get_instance().port(29000);
  ModelConfig::get_instance().devices("cuda:0");
  config::dump_startup_config();

  DistributedConfig::get_instance().nnodes(2).node_rank(1);
  ServiceConfig::get_instance().port(29000);
  ModelConfig::get_instance().devices("cuda:0");
  config::dump_startup_config();

  ASSERT_TRUE(std::filesystem::exists(dump_path));
  const nlohmann::ordered_json config_json = read_json_file(dump_path);
  EXPECT_FALSE(config_json.contains("draft_devices"));
  EXPECT_FALSE(config_json.contains("__rank_dump_seen"));

  std::filesystem::remove(dump_path);
}

TEST(ConfigJsonTest,
     DumpStartupConfigKeepsDefaultRankValueWhenRankZeroDiffers) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_rank_default_override.json";
  std::filesystem::remove(dump_path);
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  StartupConfigGuard startup_config_guard;
  RankBoundStartupConfigGuard rank_bound_startup_config_guard;

  FLAGS_enable_dump_config_json = true;

  DistributedConfig::get_instance().nnodes(2).node_rank(1);
  ServiceConfig::get_instance().port(29001);
  ModelConfig::get_instance().devices("cuda:1");
  SpeculativeConfig::get_instance().draft_devices("npu:0");
  config::dump_startup_config();

  DistributedConfig::get_instance().nnodes(2).node_rank(0);
  ServiceConfig::get_instance().port(29000);
  ModelConfig::get_instance().devices("cuda:0");
  SpeculativeConfig::get_instance().draft_devices("cuda:0");
  config::dump_startup_config();

  ASSERT_TRUE(std::filesystem::exists(dump_path));
  const nlohmann::ordered_json config_json = read_json_file(dump_path);
  ASSERT_TRUE(config_json.at("draft_devices").is_array());
  EXPECT_EQ(config_json.at("draft_devices").at(0).get<std::string>(), "cuda:0");
  EXPECT_EQ(config_json.at("draft_devices").at(1).get<std::string>(), "npu:0");
  EXPECT_FALSE(config_json.contains("__rank_dump_seen"));

  std::filesystem::remove(dump_path);
}

TEST(ConfigJsonTest, DumpStartupConfigClearsStaleSharedValuesOnRankZero) {
  const std::filesystem::path dump_path =
      std::filesystem::temp_directory_path() /
      "xllm_dump_config_json_test_clear_stale_shared.json";
  std::filesystem::remove(dump_path);
  write_config_file(dump_path, R"json({
    "block_size": 256,
    "port": [28000, 28001],
    "devices": ["cuda:0", "cuda:1"],
    "node_rank": [0, 1]
  })json");
  DumpConfigJsonFlagGuard flag_guard(dump_path.string());
  StartupConfigGuard startup_config_guard;
  RankBoundStartupConfigGuard rank_bound_startup_config_guard;
  const KVCacheConfig default_kv_cache_config;

  FLAGS_enable_dump_config_json = true;

  KVCacheConfig::get_instance().block_size(
      default_kv_cache_config.block_size());
  DistributedConfig::get_instance().nnodes(2).node_rank(0);
  ServiceConfig::get_instance().port(29000);
  ModelConfig::get_instance().devices("cuda:0");
  config::dump_startup_config();

  DistributedConfig::get_instance().nnodes(2).node_rank(1);
  ServiceConfig::get_instance().port(29001);
  ModelConfig::get_instance().devices("cuda:1");
  config::dump_startup_config();

  ASSERT_TRUE(std::filesystem::exists(dump_path));
  const nlohmann::ordered_json config_json = read_json_file(dump_path);
  EXPECT_FALSE(config_json.contains("block_size"));
  EXPECT_EQ(config_json.at("port").at(0).get<int32_t>(), 29000);
  EXPECT_EQ(config_json.at("port").at(1).get<int32_t>(), 29001);

  std::filesystem::remove(dump_path);
}

}  // namespace
}  // namespace xllm
