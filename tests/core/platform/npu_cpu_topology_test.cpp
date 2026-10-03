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

#include "core/platform/npu/npu_cpu_topology.h"

#include <gtest/gtest.h>

#include <numeric>

namespace xllm::npu {
namespace {

std::vector<NpuIdentity> inventory(int32_t count) {
  std::vector<NpuIdentity> result;
  result.reserve(count);
  for (int32_t id = 0; id < count; ++id) {
    result.emplace_back(NpuIdentity{id / 2, id % 2, id, id});
  }
  return result;
}

TEST(NpuCpuTopologyTest, ParsesA3InventoryAndIgnoresMcu) {
  const auto devices = parse_npu_inventory(
      "  NPU ID    Chip ID    Chip Logic ID    Chip Phy-ID    Chip Name\n"
      "  1         1          3                13             Ascend910\n"
      "  1         2          -                -              Mcu\n"
      "  0         0          0                10             Ascend910\n");
  ASSERT_EQ(devices.size(), 2U);
  EXPECT_EQ(devices[0].logical_id, 0);
  EXPECT_EQ(devices[1].logical_id, 3);
  EXPECT_EQ(devices[1].physical_id, 13);
  EXPECT_EQ(devices[1].card_id, 1);
  EXPECT_EQ(devices[1].chip_id, 1);
}

TEST(NpuCpuTopologyTest, RejectsUnknownAndDuplicateInventory) {
  EXPECT_TRUE(parse_npu_inventory("driver unavailable").empty());
  EXPECT_TRUE(
      parse_npu_inventory("NPU ID    Chip ID    Chip Logic ID\n0 0 1\n1 0 1\n")
          .empty());
}

TEST(NpuCpuTopologyTest, VisibleDeviceOrderIsNotSorted) {
  const auto devices = inventory(16);
  EXPECT_EQ(resolve_npu_logical_id(0, "12,3,7", devices), 12);
  EXPECT_EQ(resolve_npu_logical_id(1, "12,3,7", devices), 3);
  EXPECT_EQ(resolve_npu_logical_id(15, "", devices), 15);
  EXPECT_FALSE(resolve_npu_logical_id(3, "12,3,7", devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "3,3", devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "3,", devices));
  EXPECT_FALSE(resolve_npu_logical_id(-1, "", devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "99", devices));
}

TEST(NpuCpuTopologyTest, HiddenNpusKeepTheirGlobalSlices) {
  const auto devices = inventory(16);
  const auto id = resolve_npu_logical_id(0, "15", devices);
  ASSERT_TRUE(id);
  CpuBindingTopology host;
  host.allowed_cpus.resize(640);
  std::iota(host.allowed_cpus.begin(), host.allowed_cpus.end(), 0);
  host.device_ids.resize(16);
  std::iota(host.device_ids.begin(), host.device_ids.end(), 0);
  const auto plan = make_cpu_binding_plan(
      host,
      *id,
      npu_cpu_binding_options(/*global_slice=*/true, /*bind_irq=*/false));
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->worker_cpus.front(), 600);
}

TEST(NpuCpuTopologyTest, ParsesTopologyByPhysicalId) {
  auto devices = inventory(2);
  devices[0].physical_id = 10;
  devices[1].physical_id = 11;
  const auto affinity = parse_npu_affinity(
      "  Phy-ID10  Phy-ID11  CPU Affinity\n"
      "Phy-ID10  X    SYS    0-19,40-59\n"
      "Phy-ID11  SYS  X      20-39,60-79\n",
      devices);
  ASSERT_EQ(affinity.size(), 2U);
  EXPECT_EQ(affinity.at(0).front(), 0);
  EXPECT_EQ(affinity.at(0).back(), 59);
  EXPECT_TRUE(parse_npu_affinity("Phy-ID10 X SIO\n", devices).empty());
}

TEST(NpuCpuTopologyTest, AscendRolesAndMemoryPolicyAreBackendOptions) {
  const auto a3 =
      npu_cpu_binding_options(/*global_slice=*/true, /*bind_irq=*/false);
  EXPECT_EQ(a3.mode, CpuBindingMode::GLOBAL_SLICE);
  EXPECT_EQ(a3.reserved_cpu_count, 0);
  EXPECT_EQ(a3.dedicated_threads,
            (std::vector<std::string>{"acl_thread", "release_thread"}));
  EXPECT_EQ(a3.memory_policy, numa::MemoryPolicy::PREFERRED);
  const auto a2 =
      npu_cpu_binding_options(/*global_slice=*/false, /*bind_irq=*/true);
  EXPECT_EQ(a2.mode, CpuBindingMode::TOPO_AFFINITY);
  EXPECT_TRUE(a2.extend_numa_pool);
  EXPECT_EQ(a2.reserved_cpu_count, 2);
}

}  // namespace
}  // namespace xllm::npu
