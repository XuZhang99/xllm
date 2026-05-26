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

#pragma once

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>

#include "core/util/json_reader.h"

namespace xllm::config {

JsonReader load_json_file(const std::string& config_path);

JsonReader parse_json_string(std::string_view config_json);

const std::optional<JsonReader>& get_parsed_json_config();

int32_t get_rank_config_index();

void validate_rank_config_index(const std::string& key,
                                std::size_t rank_values_size);

template <typename T, typename T2>
void append_rank_json_value_if_not_default(nlohmann::ordered_json& config_json,
                                           const std::string& key,
                                           const T& value,
                                           const T2& default_value,
                                           int32_t rank,
                                           int32_t nnodes,
                                           bool all_ranks_dumped) {
  if (rank < 0 || rank >= nnodes || nnodes <= 0) {
    return;
  }

  nlohmann::ordered_json old_array = nlohmann::ordered_json::array();
  if (config_json.contains(key) && config_json[key].is_array()) {
    old_array = config_json[key];
  }

  nlohmann::ordered_json new_array = nlohmann::ordered_json::array();
  for (int32_t idx = 0; idx < nnodes; ++idx) {
    if (static_cast<std::size_t>(idx) < old_array.size()) {
      new_array.push_back(old_array[static_cast<std::size_t>(idx)]);
    } else {
      new_array.push_back(nullptr);
    }
  }

  new_array[static_cast<std::size_t>(rank)] = value;

  if (!all_ranks_dumped) {
    while (new_array.size() > 1 && new_array.back().is_null()) {
      new_array.erase(new_array.size() - 1);
    }
    config_json[key] = new_array;
    return;
  }

  nlohmann::ordered_json rank0_value = default_value;
  if (!new_array.empty() && !new_array.front().is_null()) {
    rank0_value = new_array.front();
  } else if (!new_array.empty()) {
    new_array.front() = rank0_value;
  }

  bool has_rank_override = false;
  for (int32_t idx = 1; idx < nnodes; ++idx) {
    nlohmann::ordered_json& rank_value =
        new_array[static_cast<std::size_t>(idx)];
    if (rank_value.is_null()) {
      continue;
    }
    if (rank_value == rank0_value) {
      rank_value = nullptr;
    } else {
      has_rank_override = true;
    }
  }

  while (new_array.size() > 1 && new_array.back().is_null()) {
    new_array.erase(new_array.size() - 1);
  }

  const bool rank0_is_non_default = rank0_value != default_value;
  if (rank0_is_non_default || has_rank_override) {
    config_json[key] = new_array;
  } else {
    config_json.erase(key);
  }
}

template <typename T, typename T2>
T rank_value_or(const JsonReader& json,
                const std::string& key,
                T2 default_value) {
  const std::optional<nlohmann::json> value_json = json.json_value(key);
  if (!value_json) {
    return default_value;
  }

  if (!value_json->is_array()) {
    if (value_json->is_null() || value_json->is_object()) {
      return default_value;
    }
    return value_json->get<T>();
  }

  validate_rank_config_index(key, value_json->size());
  const int32_t rank_config_index = get_rank_config_index();
  if (rank_config_index >= 0 &&
      static_cast<std::size_t>(rank_config_index) < value_json->size()) {
    const nlohmann::json& rank_value =
        (*value_json)[static_cast<std::size_t>(rank_config_index)];
    if (!rank_value.is_null() && !rank_value.is_object()) {
      return rank_value.get<T>();
    }
  }

  if (value_json->empty()) {
    return default_value;
  }
  const nlohmann::json& rank0_value = (*value_json)[0];
  if (rank0_value.is_null() || rank0_value.is_object()) {
    return default_value;
  }
  return rank0_value.get<T>();
}

void dump_startup_config();

}  // namespace xllm::config

#define APPEND_JSON_VALUE_IF_NOT_DEFAULT(       \
    config_json, key, value, default_value)     \
  do {                                          \
    const auto& config_json_value = (value);    \
    if (config_json_value != (default_value)) { \
      (config_json)[key] = config_json_value;   \
    }                                           \
  } while (false)

#define APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT( \
    config_json, default_config, property)       \
  APPEND_JSON_VALUE_IF_NOT_DEFAULT(              \
      config_json, #property, property(), (default_config).property())

#define APPEND_RANK_JSON_VALUE_IF_NOT_DEFAULT(                              \
    config_json, key, value, default_value, rank, nnodes, all_ranks_dumped) \
  do {                                                                      \
    const auto& config_json_value = (value);                                \
    const auto& config_json_default_value = (default_value);                \
    ::xllm::config::append_rank_json_value_if_not_default(                  \
        (config_json),                                                      \
        (key),                                                              \
        config_json_value,                                                  \
        config_json_default_value,                                          \
        (rank),                                                             \
        (nnodes),                                                           \
        (all_ranks_dumped));                                                \
  } while (false)

#define APPEND_RANK_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json,      \
                                                     config,           \
                                                     default_config,   \
                                                     property,         \
                                                     rank,             \
                                                     nnodes,           \
                                                     all_ranks_dumped) \
  APPEND_RANK_JSON_VALUE_IF_NOT_DEFAULT(config_json,                   \
                                        #property,                     \
                                        (config).property(),           \
                                        (default_config).property(),   \
                                        rank,                          \
                                        nnodes,                        \
                                        all_ranks_dumped)
