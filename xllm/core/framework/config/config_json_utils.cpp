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

#include "core/framework/config/config_json_utils.h"

#include <fcntl.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/file.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>

#include "core/common/global_flags.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/profile_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"

DEFINE_string(config_json_file,
              "",
              "Path to a JSON config file. Values in the file override "
              "command-line flag values. Rank-bound values can be arrays "
              "indexed by --node_rank.");

DEFINE_bool(enable_dump_config_json,
            false,
            "Whether to dump the resolved startup config as JSON.");

DEFINE_string(dump_config_json_file,
              "xllm_config.json",
              "Path to write the resolved startup config as JSON. Used only "
              "when enable_dump_config_json is true. Multi-rank dumps are "
              "merged into this single file.");

namespace xllm::config {
namespace {

std::mutex& parsed_json_config_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<std::once_flag>& parsed_json_config_once() {
  static std::unique_ptr<std::once_flag> once_flag =
      std::make_unique<std::once_flag>();
  return once_flag;
}

std::string& parsed_json_config_path() {
  static std::string config_path;
  return config_path;
}

std::optional<JsonReader>& parsed_json_config() {
  static std::optional<JsonReader> json_config;
  return json_config;
}

inline constexpr const char* kRankDumpSeenKey = "__rank_dump_seen";

bool is_rank_bound_config_key(const std::string& key) {
  return key == "port" || key == "devices" || key == "draft_devices" ||
         key == "node_rank";
}

bool is_rank_dump_state_key(const std::string& key) {
  return key == kRankDumpSeenKey;
}

class FileLock final {
 public:
  explicit FileLock(const std::filesystem::path& lock_path) {
    fd_ = ::open(lock_path.string().c_str(), O_CREAT | O_RDWR, 0644);
    if (fd_ < 0) {
      LOG(FATAL) << "Failed to open startup config dump file: "
                 << lock_path.string() << ", error: " << std::strerror(errno);
    }
    if (::flock(fd_, LOCK_EX) != 0) {
      LOG(FATAL) << "Failed to lock startup config dump file: "
                 << lock_path.string() << ", error: " << std::strerror(errno);
    }
  }

  ~FileLock() {
    if (fd_ >= 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
    }
  }

  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

 private:
  int fd_ = -1;
};

void load_parsed_json_config() {
  const std::string& config_path = parsed_json_config_path();
  if (config_path.empty()) {
    return;
  }

  JsonReader reader;
  try {
    if (!reader.parse(config_path)) {
      LOG(ERROR) << "Failed to load JSON config file: " << config_path;
      return;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to parse JSON config file: " << config_path
               << ", error: " << e.what();
    return;
  }

  parsed_json_config() = reader;
}

void reset_parsed_json_config_if_path_changed() {
  if (parsed_json_config_path() == FLAGS_config_json_file) {
    return;
  }

  parsed_json_config_path() = FLAGS_config_json_file;
  parsed_json_config().reset();
  parsed_json_config_once() = std::make_unique<std::once_flag>();
}

nlohmann::ordered_json build_startup_config_json() {
  nlohmann::ordered_json config_json = nlohmann::ordered_json::object();

  ServiceConfig::get_instance().append_config_json(config_json);
  ModelConfig::get_instance().append_config_json(config_json);
  LoadConfig::get_instance().append_config_json(config_json);
  KVCacheConfig::get_instance().append_config_json(config_json);
  KVCacheStoreConfig::get_instance().append_config_json(config_json);
  BeamSearchConfig::get_instance().append_config_json(config_json);
  SchedulerConfig::get_instance().append_config_json(config_json);
  ParallelConfig::get_instance().append_config_json(config_json);
  EPLBConfig::get_instance().append_config_json(config_json);
  DistributedConfig::get_instance().append_config_json(config_json);
  DisaggPDConfig::get_instance().append_config_json(config_json);
  SpeculativeConfig::get_instance().append_config_json(config_json);
  ProfileConfig::get_instance().append_config_json(config_json);
  ExecutionConfig::get_instance().append_config_json(config_json);
  KernelConfig::get_instance().append_config_json(config_json);
  DiTConfig::get_instance().append_config_json(config_json);
  RecConfig::get_instance().append_config_json(config_json);

  return config_json;
}

nlohmann::ordered_json load_existing_dump_config(
    const std::filesystem::path& dump_path) {
  if (!std::filesystem::exists(dump_path)) {
    return nlohmann::ordered_json::object();
  }
  if (std::filesystem::file_size(dump_path) == 0) {
    return nlohmann::ordered_json::object();
  }

  std::ifstream input_stream(dump_path);
  if (!input_stream.is_open()) {
    LOG(FATAL) << "Failed to open existing startup config dump file: "
               << dump_path.string();
  }

  try {
    nlohmann::ordered_json config_json =
        nlohmann::ordered_json::parse(input_stream);
    if (!config_json.is_object()) {
      LOG(WARNING) << "Existing startup config dump file is not a JSON object, "
                   << "overwriting it: " << dump_path.string();
      return nlohmann::ordered_json::object();
    }
    return config_json;
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse existing startup config dump file: "
               << dump_path.string() << ", error: " << e.what();
  }
  return nlohmann::ordered_json::object();
}

void write_dump_config_json(const std::filesystem::path& dump_path,
                            const nlohmann::ordered_json& config_json) {
  std::filesystem::path temp_path = dump_path;
  temp_path += ".tmp." + std::to_string(::getpid());

  std::ofstream output_stream(temp_path);
  if (!output_stream.is_open()) {
    LOG(FATAL) << "Failed to open startup config dump file: "
               << temp_path.string();
  }

  output_stream << config_json.dump(2) << "\n";
  output_stream.close();
  if (!output_stream.good()) {
    LOG(FATAL) << "Failed to write startup config dump file: "
               << temp_path.string();
  }

  std::error_code error_code;
  std::filesystem::rename(temp_path, dump_path, error_code);
  if (error_code) {
    std::filesystem::remove(temp_path);
    LOG(FATAL) << "Failed to replace startup config dump file: "
               << dump_path.string() << ", error: " << error_code.message();
  }
}

void write_locked_dump_config_json(const std::filesystem::path& dump_path,
                                   const nlohmann::ordered_json& config_json) {
  std::ofstream output_stream(dump_path, std::ios::trunc);
  if (!output_stream.is_open()) {
    LOG(FATAL) << "Failed to open startup config dump file: "
               << dump_path.string();
  }

  output_stream << config_json.dump(2) << "\n";
  output_stream.close();
  if (!output_stream.good()) {
    LOG(FATAL) << "Failed to write startup config dump file: "
               << dump_path.string();
  }
}

void mark_rank_dump_seen(nlohmann::ordered_json& config_json,
                         int32_t rank,
                         int32_t nnodes) {
  nlohmann::ordered_json seen_array = nlohmann::ordered_json::array();
  if (config_json.contains(kRankDumpSeenKey) &&
      config_json[kRankDumpSeenKey].is_array()) {
    seen_array = config_json[kRankDumpSeenKey];
  }

  while (seen_array.size() < static_cast<std::size_t>(nnodes)) {
    seen_array.push_back(false);
  }
  while (seen_array.size() > static_cast<std::size_t>(nnodes)) {
    seen_array.erase(seen_array.size() - 1);
  }

  seen_array[static_cast<std::size_t>(rank)] = true;
  config_json[kRankDumpSeenKey] = seen_array;
}

bool are_all_ranks_dumped(const nlohmann::ordered_json& config_json,
                          int32_t nnodes) {
  if (!config_json.contains(kRankDumpSeenKey) ||
      !config_json[kRankDumpSeenKey].is_array()) {
    return false;
  }

  const nlohmann::ordered_json& seen_array = config_json[kRankDumpSeenKey];
  if (seen_array.size() < static_cast<std::size_t>(nnodes)) {
    return false;
  }

  for (int32_t rank = 0; rank < nnodes; ++rank) {
    const nlohmann::ordered_json& seen_value =
        seen_array[static_cast<std::size_t>(rank)];
    if (!seen_value.is_boolean() || !seen_value.get<bool>()) {
      return false;
    }
  }
  return true;
}

nlohmann::ordered_json merge_rank_startup_config_json(
    const std::filesystem::path& dump_path,
    const nlohmann::ordered_json& current_config_json) {
  const ServiceConfig& service_config = ServiceConfig::get_instance();
  const ModelConfig& model_config = ModelConfig::get_instance();
  const SpeculativeConfig& speculative_config =
      SpeculativeConfig::get_instance();
  const DistributedConfig& distributed_config =
      DistributedConfig::get_instance();
  const ServiceConfig default_service_config;
  const ModelConfig default_model_config;
  const SpeculativeConfig default_speculative_config;
  const DistributedConfig default_distributed_config;
  const int32_t nnodes = distributed_config.nnodes();
  const int32_t rank = distributed_config.node_rank();
  CHECK(nnodes > 1) << "Multi-rank startup config dump requires nnodes > 1.";
  CHECK(rank >= 0 && rank < nnodes)
      << "Invalid node_rank " << rank << " for nnodes " << nnodes;

  nlohmann::ordered_json merged_config_json =
      load_existing_dump_config(dump_path);
  if (rank == 0) {
    for (auto it = merged_config_json.begin();
         it != merged_config_json.end();) {
      const std::string key = it.key();
      if (!is_rank_bound_config_key(key) && !is_rank_dump_state_key(key)) {
        it = merged_config_json.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = current_config_json.begin(); it != current_config_json.end();
         ++it) {
      const std::string key = it.key();
      if (!is_rank_bound_config_key(key)) {
        merged_config_json[key] = it.value();
      }
    }
  }
  mark_rank_dump_seen(merged_config_json, rank, nnodes);
  const bool all_ranks_dumped =
      are_all_ranks_dumped(merged_config_json, nnodes);

  APPEND_RANK_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(merged_config_json,
                                               service_config,
                                               default_service_config,
                                               port,
                                               rank,
                                               nnodes,
                                               all_ranks_dumped);
  APPEND_RANK_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(merged_config_json,
                                               model_config,
                                               default_model_config,
                                               devices,
                                               rank,
                                               nnodes,
                                               all_ranks_dumped);
  APPEND_RANK_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(merged_config_json,
                                               speculative_config,
                                               default_speculative_config,
                                               draft_devices,
                                               rank,
                                               nnodes,
                                               all_ranks_dumped);
  APPEND_RANK_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(merged_config_json,
                                               distributed_config,
                                               default_distributed_config,
                                               node_rank,
                                               rank,
                                               nnodes,
                                               all_ranks_dumped);

  if (all_ranks_dumped) {
    merged_config_json.erase(kRankDumpSeenKey);
  }

  return merged_config_json;
}

}  // namespace

JsonReader load_json_file(const std::string& config_path) {
  JsonReader reader;
  if (!config_path.empty()) {
    reader.parse(config_path);
  }
  return reader;
}

JsonReader parse_json_string(std::string_view config_json) {
  JsonReader reader;
  if (!config_json.empty()) {
    reader.parse_text(std::string(config_json));
  }
  return reader;
}

const std::optional<JsonReader>& get_parsed_json_config() {
  std::lock_guard<std::mutex> lock(parsed_json_config_mutex());
  reset_parsed_json_config_if_path_changed();
  std::call_once(*parsed_json_config_once(), load_parsed_json_config);
  return parsed_json_config();
}

int32_t get_rank_config_index() { return FLAGS_node_rank; }

bool is_rank_config_index_explicit() {
  GFLAGS_NAMESPACE::CommandLineFlagInfo flag_info;
  if (!GFLAGS_NAMESPACE::GetCommandLineFlagInfo("node_rank", &flag_info)) {
    return false;
  }
  return !flag_info.is_default;
}

void validate_rank_config_index(const std::string& key,
                                std::size_t rank_values_size) {
  if (rank_values_size <= 1) {
    return;
  }

  if (get_rank_config_index() != 0 || is_rank_config_index_explicit()) {
    return;
  }

  LOG(FATAL) << "JSON config key \"" << key
             << "\" is a rank-bound array, but --node_rank was not "
             << "specified. Pass --node_rank=$i together with "
             << "--config_json_file so each xllm process can select its "
             << "rank-specific values.";
}

void dump_startup_config() {
  if (!FLAGS_enable_dump_config_json) {
    return;
  }

  const std::filesystem::path dump_path =
      std::filesystem::path(FLAGS_dump_config_json_file).lexically_normal();
  if (dump_path.has_parent_path()) {
    std::error_code error_code;
    std::filesystem::create_directories(dump_path.parent_path(), error_code);
    if (error_code) {
      LOG(FATAL) << "Failed to create startup config dump directory: "
                 << dump_path.parent_path().string()
                 << ", error: " << error_code.message();
    }
  }

  const nlohmann::ordered_json config_json = build_startup_config_json();
  if (DistributedConfig::get_instance().nnodes() > 1) {
    const FileLock file_lock(dump_path);
    static_cast<void>(file_lock);
    const nlohmann::ordered_json merged_config_json =
        merge_rank_startup_config_json(dump_path, config_json);
    write_locked_dump_config_json(dump_path, merged_config_json);
    LOG(INFO) << "Merged startup config for rank "
              << DistributedConfig::get_instance().node_rank() << " to "
              << dump_path.string();
    return;
  }

  write_dump_config_json(dump_path, config_json);
  LOG(INFO) << "Dumped startup config to " << dump_path.string();
}

}  // namespace xllm::config
