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

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm::npu {

struct NpuIdentity {
  int32_t card_id = -1;
  int32_t chip_id = -1;
  int32_t logical_id = -1;
  int32_t physical_id = -1;
};

struct CpuBindingTopology {
  std::vector<int32_t> allowed_cpus;
  std::vector<NpuIdentity> devices;
  std::unordered_map<int32_t, std::vector<int32_t>> affinity;
  std::unordered_map<int32_t, int32_t> cpu_nodes;
};

struct CpuBindingPlan {
  int32_t logical_id = -1;
  std::string mode;
  std::vector<int32_t> worker_cpus;
  std::vector<int32_t> irq_cpus;
  int32_t acl_cpu = -1;
  int32_t release_cpu = -1;
  int32_t memory_node = -1;
};

// Pure parsers/planner, also used by CPU-only tests.
std::optional<std::vector<int32_t>> parse_cpu_list(const std::string& text);
std::vector<NpuIdentity> parse_npu_inventory(const std::string& text);
std::unordered_map<int32_t, std::vector<int32_t>> parse_npu_affinity(
    const std::string& text,
    const std::vector<NpuIdentity>& devices);
std::optional<int32_t> resolve_npu_logical_id(
    int32_t device_index,
    const std::string& visible_devices,
    const std::vector<NpuIdentity>& devices);
std::optional<CpuBindingPlan> make_cpu_binding_plan(
    const CpuBindingTopology& topology,
    int32_t logical_id,
    bool global_slice,
    bool reserve_irq_cpus,
    std::string* error);

// Applies one validated plan to the current process; restores prior masks if
// a thread cannot be bound. Does not change memory policy or IRQ placement.
bool apply_cpu_binding_plan(const CpuBindingPlan& plan);

// Standalone serving owns one NPU per process. Initialize before model loading;
// refresh after runtime initialization/loading to place newly created threads.
// Only the current process's threads and its device's IRQs are eligible.
class NpuCpuBinding final {
 public:
  static NpuCpuBinding& get_instance();

  void initialize(int32_t device_index,
                  const std::string& soc_name,
                  bool bind_irq);
  void refresh_threads();
  void refresh_after_first_forward();

 private:
  std::mutex mutex_;
  std::once_flag first_forward_;
  std::optional<CpuBindingPlan> plan_;
};

}  // namespace xllm::npu
