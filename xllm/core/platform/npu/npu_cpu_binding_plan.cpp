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

#include <algorithm>
#include <charconv>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "core/platform/npu/npu_cpu_binding.h"

namespace xllm::npu {
namespace {

std::optional<int32_t> parse_id(std::string_view text) {
  int32_t value = -1;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size() ||
      value < 0) {
    return std::nullopt;
  }
  return value;
}

std::vector<int32_t> sorted_unique(std::vector<int32_t> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::vector<int32_t> intersect(const std::vector<int32_t>& left,
                               const std::vector<int32_t>& right) {
  const auto sorted = sorted_unique(left);
  std::vector<int32_t> result;
  result.reserve(std::min(left.size(), right.size()));
  std::set_intersection(sorted.begin(),
                        sorted.end(),
                        right.begin(),
                        right.end(),
                        std::back_inserter(result));
  return result;
}

std::vector<int32_t> slice(const std::vector<int32_t>& cpus,
                           size_t index,
                           size_t count) {
  const size_t base = cpus.size() / count;
  const size_t extra = cpus.size() % count;
  const size_t start = index * base + std::min(index, extra);
  const size_t length = base + (index < extra ? 1 : 0);
  return {cpus.begin() + start, cpus.begin() + start + length};
}

}  // namespace

std::optional<std::vector<int32_t>> parse_cpu_list(const std::string& text) {
  if (text.empty()) {
    return std::nullopt;
  }
  std::vector<int32_t> result;
  result.reserve(text.size());
  std::istringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const size_t dash = item.find('-');
    const auto first = parse_id(item.substr(0, dash));
    const auto last =
        dash == std::string::npos ? first : parse_id(item.substr(dash + 1));
    // Bound expansion of malformed topology output before allocating memory.
    if (!first || !last || *last < *first || *last >= 1024 * 1024) {
      return std::nullopt;
    }
    for (int32_t cpu = *first; cpu <= *last; ++cpu) {
      result.emplace_back(cpu);
    }
  }
  if (text.back() == ',' || result.empty()) {
    return std::nullopt;
  }
  return sorted_unique(std::move(result));
}

std::vector<NpuIdentity> parse_npu_inventory(const std::string& text) {
  std::vector<NpuIdentity> devices;
  devices.reserve(16);
  // Header positions matter: never treat an MCU or a physical ID as a logical
  // ID.
  const std::regex separator(R"(\s{2,})");
  std::istringstream stream(text);
  std::string line;
  if (!std::getline(stream, line)) {
    return {};
  }
  const size_t first = line.find_first_not_of(" \t\r");
  if (first == std::string::npos) {
    return {};
  }
  line = line.substr(first);
  const std::vector<std::string> headers{
      std::sregex_token_iterator(line.begin(), line.end(), separator, -1),
      std::sregex_token_iterator()};
  const auto column = [&headers](const std::string& name) -> size_t {
    const auto it = std::find(headers.begin(), headers.end(), name);
    return static_cast<size_t>(std::distance(headers.begin(), it));
  };
  const size_t card = column("NPU ID");
  const size_t chip = column("Chip ID");
  const size_t logic = column("Chip Logic ID");
  const size_t physical = column("Chip Phy-ID");
  if (card == headers.size() || chip == headers.size() ||
      logic == headers.size()) {
    return {};
  }
  std::unordered_set<int32_t> seen;
  while (std::getline(stream, line)) {
    std::istringstream fields(line);
    const std::vector<std::string> values{
        std::istream_iterator<std::string>(fields),
        std::istream_iterator<std::string>()};
    if (values.size() <= std::max({card, chip, logic})) {
      continue;
    }
    const auto card_id = parse_id(values[card]);
    const auto chip_id = parse_id(values[chip]);
    const auto logical_id = parse_id(values[logic]);
    if (!logical_id) {
      continue;
    }
    if (!card_id || !chip_id || !seen.insert(*logical_id).second) {
      return {};
    }
    const auto physical_id =
        physical < values.size() ? parse_id(values[physical]) : logical_id;
    if (!physical_id) {
      return {};
    }
    devices.emplace_back(
        NpuIdentity{*card_id, *chip_id, *logical_id, *physical_id});
  }
  std::sort(
      devices.begin(), devices.end(), [](const auto& left, const auto& right) {
        return left.logical_id < right.logical_id;
      });
  return devices;
}

std::unordered_map<int32_t, std::vector<int32_t>> parse_npu_affinity(
    const std::string& text,
    const std::vector<NpuIdentity>& devices) {
  std::unordered_map<int32_t, std::vector<int32_t>> result;
  // A2 reports NPU<n>; newer drivers can label rows with physical IDs.
  const std::regex row(
      R"(^\s*(NPU|Phy-ID)([0-9]+)\s+.*?\s+([0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*)\s*$)");
  if (text.find("Affinity") == std::string::npos) {
    return result;
  }
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    std::smatch match;
    if (!std::regex_match(line, match, row)) {
      continue;
    }
    const auto id = parse_id(match[2].str());
    const auto cpus = parse_cpu_list(match[3].str());
    if (!id || !cpus) {
      continue;
    }
    const bool physical = match[1].str() == "Phy-ID";
    const auto device =
        std::find_if(devices.begin(), devices.end(), [&](const auto& item) {
          return (physical ? item.physical_id : item.logical_id) == *id;
        });
    if (device != devices.end()) {
      result.emplace(device->logical_id, *cpus);
    }
  }
  return result;
}

std::optional<int32_t> resolve_npu_logical_id(
    int32_t device_index,
    const std::string& visible_devices,
    const std::vector<NpuIdentity>& devices) {
  if (device_index < 0) {
    return std::nullopt;
  }
  int32_t logical_id = device_index;
  if (!visible_devices.empty()) {
    std::istringstream stream(visible_devices);
    std::string field;
    std::vector<int32_t> ids;
    ids.reserve(devices.size());
    std::unordered_set<int32_t> seen;
    while (std::getline(stream, field, ',')) {
      const auto id = parse_id(field);
      if (!id || !seen.insert(*id).second) {
        return std::nullopt;
      }
      ids.emplace_back(*id);
    }
    if (visible_devices.back() == ',' ||
        static_cast<size_t>(device_index) >= ids.size()) {
      return std::nullopt;
    }
    logical_id = ids[device_index];
  }
  const auto found = std::find_if(
      devices.begin(), devices.end(), [logical_id](const auto& device) {
        return device.logical_id == logical_id;
      });
  return found == devices.end() ? std::nullopt
                                : std::optional<int32_t>(logical_id);
}

std::optional<CpuBindingPlan> make_cpu_binding_plan(
    const CpuBindingTopology& topology,
    int32_t logical_id,
    bool global_slice,
    bool reserve_irq_cpus,
    std::string* error) {
  const auto fail =
      [error](const std::string& reason) -> std::optional<CpuBindingPlan> {
    if (error != nullptr) {
      *error = reason;
    }
    return std::nullopt;
  };
  const auto allowed = sorted_unique(topology.allowed_cpus);
  if (allowed.empty() || allowed.front() < 0 || topology.devices.empty()) {
    return fail("empty or invalid CPU/NPU inventory");
  }
  std::vector<int32_t> ids;
  ids.reserve(topology.devices.size());
  for (const auto& device : topology.devices) {
    ids.emplace_back(device.logical_id);
  }
  ids = sorted_unique(std::move(ids));
  const auto device = std::find(ids.begin(), ids.end(), logical_id);
  if (device == ids.end() || ids.front() < 0 ||
      ids.size() != topology.devices.size()) {
    return fail("invalid or duplicate logical NPU ID");
  }
  const size_t minimum = reserve_irq_cpus ? 5 : 3;
  std::vector<int32_t> pool;
  std::string mode = "global_slice";
  if (global_slice || topology.affinity.empty()) {
    if (allowed.size() / ids.size() < minimum) {
      return fail("insufficient allowed CPUs for all logical NPUs");
    }
    pool =
        slice(allowed, static_cast<size_t>(device - ids.begin()), ids.size());
  } else {
    mode = "topo_affinity";
    // Ordered groups provide deterministic allocation, including hidden NPUs.
    std::map<std::vector<int32_t>, std::vector<int32_t>> groups;
    std::set<int32_t> nodes;
    for (const auto& entry : topology.cpu_nodes) {
      if (entry.second >= 0) {
        nodes.insert(entry.second);
      }
    }
    for (int32_t id : ids) {
      const auto affinity = topology.affinity.find(id);
      if (affinity == topology.affinity.end()) {
        return fail("incomplete topology affinity; cannot ensure isolation");
      }
      auto cpus = intersect(affinity->second, allowed);
      if (cpus.empty()) {
        if (id == logical_id) {
          return fail("NPU affinity does not intersect the allowed cpuset");
        }
        continue;
      }
      std::set<int32_t> affinity_nodes;
      for (int32_t cpu : cpus) {
        const auto node = topology.cpu_nodes.find(cpu);
        if (node == topology.cpu_nodes.end()) {
          return fail("incomplete CPU NUMA map");
        }
        affinity_nodes.insert(node->second);
      }
      if (affinity_nodes.size() == 1 && nodes.size() > 1) {
        auto next = nodes.upper_bound(*affinity_nodes.begin());
        if (next == nodes.end()) {
          next = nodes.begin();
        }
        for (int32_t cpu : allowed) {
          const auto node = topology.cpu_nodes.find(cpu);
          if (node != topology.cpu_nodes.end() && node->second == *next) {
            cpus.emplace_back(cpu);
          }
        }
        cpus = sorted_unique(std::move(cpus));
      }
      groups[cpus].emplace_back(id);
    }
    std::unordered_set<int32_t> claimed;
    for (const auto& [cpus, members] : groups) {
      for (int32_t cpu : cpus) {
        if (!claimed.insert(cpu).second) {
          return fail("partially overlapping topology groups");
        }
      }
      const auto member = std::find(members.begin(), members.end(), logical_id);
      if (member == members.end()) {
        continue;
      }
      if (cpus.size() / members.size() < minimum) {
        return fail("insufficient CPUs in the shared topology affinity group");
      }
      pool = slice(
          cpus, static_cast<size_t>(member - members.begin()), members.size());
    }
  }
  if (pool.size() < minimum) {
    return fail("CPU pool is too small for worker/ACL/release roles");
  }
  CpuBindingPlan plan;
  plan.logical_id = logical_id;
  plan.mode = std::move(mode);
  plan.worker_cpus.assign(pool.begin() + (reserve_irq_cpus ? 2 : 0),
                          pool.end() - 2);
  if (reserve_irq_cpus) {
    plan.irq_cpus.assign(pool.begin(), pool.begin() + 2);
  }
  plan.acl_cpu = pool[pool.size() - 2];
  plan.release_cpu = pool.back();
  const auto node = topology.cpu_nodes.find(pool.front());
  if (node != topology.cpu_nodes.end()) {
    plan.memory_node = node->second;
  }
  return plan;
}

}  // namespace xllm::npu
