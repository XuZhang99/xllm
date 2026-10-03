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

#include "core/platform/npu/npu_cpu_binding.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <numeric>
#include <unordered_set>

namespace xllm::npu {
namespace {

CpuBindingTopology topology(int32_t cpu_count, int32_t device_count) {
  CpuBindingTopology result;
  result.allowed_cpus.resize(cpu_count);
  std::iota(result.allowed_cpus.begin(), result.allowed_cpus.end(), 0);
  result.devices.reserve(device_count);
  for (int32_t id = 0; id < device_count; ++id) {
    result.devices.emplace_back(NpuIdentity{id / 2, id % 2, id, id});
  }
  for (int32_t cpu = 0; cpu < cpu_count; ++cpu) {
    result.cpu_nodes.emplace(cpu, cpu / 40);
  }
  return result;
}

std::vector<int32_t> all_cpus(const CpuBindingPlan& plan) {
  auto result = plan.worker_cpus;
  result.insert(result.end(), plan.irq_cpus.begin(), plan.irq_cpus.end());
  result.emplace_back(plan.acl_cpu);
  result.emplace_back(plan.release_cpu);
  std::sort(result.begin(), result.end());
  return result;
}

TEST(NpuCpuBindingTest, ParsesSparseCpuSet) {
  EXPECT_EQ(parse_cpu_list("8-10,2,4-5,4"),
            (std::vector<int32_t>{2, 4, 5, 8, 9, 10}));
}

TEST(NpuCpuBindingTest, RejectsMalformedCpuSets) {
  for (const auto& text : {"",
                           "-1",
                           "4-2",
                           "1,",
                           "1,,2",
                           "1x",
                           "1-2-3",
                           "2147483648",
                           "0-1048576"}) {
    EXPECT_FALSE(parse_cpu_list(text)) << text;
  }
}

TEST(NpuCpuBindingTest, ParsesA3InventoryAndIgnoresMcu) {
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

TEST(NpuCpuBindingTest, RejectsUnknownAndDuplicateInventory) {
  EXPECT_TRUE(parse_npu_inventory("driver unavailable").empty());
  EXPECT_TRUE(
      parse_npu_inventory("NPU ID    Chip ID    Chip Logic ID\n0 0 1\n1 0 1\n")
          .empty());
}

TEST(NpuCpuBindingTest, VisibleDeviceOrderIsNotSorted) {
  const auto host = topology(640, 16);
  EXPECT_EQ(resolve_npu_logical_id(0, "12,3,7", host.devices), 12);
  EXPECT_EQ(resolve_npu_logical_id(1, "12,3,7", host.devices), 3);
  EXPECT_EQ(resolve_npu_logical_id(15, "", host.devices), 15);
  EXPECT_FALSE(resolve_npu_logical_id(3, "12,3,7", host.devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "3,3", host.devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "3,", host.devices));
  EXPECT_FALSE(resolve_npu_logical_id(-1, "", host.devices));
  EXPECT_FALSE(resolve_npu_logical_id(0, "99", host.devices));
}

TEST(NpuCpuBindingTest, A3MatchesDocumentedFortyCpuSlices) {
  const auto host = topology(640, 16);
  std::unordered_set<int32_t> used;
  std::string error;
  for (int32_t id = 0; id < 16; ++id) {
    const auto plan = make_cpu_binding_plan(host, id, true, true, &error);
    ASSERT_TRUE(plan) << error;
    EXPECT_EQ(plan->mode, "global_slice");
    EXPECT_EQ(plan->worker_cpus.size(), 36U);
    EXPECT_EQ(plan->worker_cpus.front(), id * 40 + 2);
    EXPECT_EQ(plan->worker_cpus.back(), id * 40 + 37);
    EXPECT_EQ(plan->irq_cpus, (std::vector<int32_t>{id * 40, id * 40 + 1}));
    EXPECT_EQ(plan->acl_cpu, id * 40 + 38);
    EXPECT_EQ(plan->release_cpu, id * 40 + 39);
    EXPECT_EQ(plan->memory_node, id);
    for (int32_t cpu : all_cpus(*plan)) {
      EXPECT_TRUE(used.insert(cpu).second);
    }
  }
  EXPECT_EQ(used.size(), 640U);
}

TEST(NpuCpuBindingTest, DoesNotReserveIrqCpusWhenDisabled) {
  const auto plan =
      make_cpu_binding_plan(topology(640, 16), 7, true, false, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->worker_cpus.size(), 38U);
  EXPECT_EQ(plan->worker_cpus.front(), 280);
  EXPECT_TRUE(plan->irq_cpus.empty());
}

TEST(NpuCpuBindingTest, SplitsSparseRestrictedCpusetAndRemainder) {
  auto host = topology(0, 3);
  host.allowed_cpus = {1, 3, 7, 9, 11, 14, 19, 23, 25, 27, 33};
  const auto first = make_cpu_binding_plan(host, 0, true, false, nullptr);
  const auto last = make_cpu_binding_plan(host, 2, true, false, nullptr);
  ASSERT_TRUE(first);
  ASSERT_TRUE(last);
  EXPECT_EQ(all_cpus(*first), (std::vector<int32_t>{1, 3, 7, 9}));
  EXPECT_EQ(all_cpus(*last), (std::vector<int32_t>{25, 27, 33}));
}

TEST(NpuCpuBindingTest, HiddenNpusKeepTheirGlobalSlices) {
  const auto host = topology(640, 16);
  const auto id = resolve_npu_logical_id(0, "15", host.devices);
  ASSERT_TRUE(id);
  const auto plan = make_cpu_binding_plan(host, *id, true, false, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->worker_cpus.front(), 600);
}

TEST(NpuCpuBindingTest, SupportsNonContiguousLogicalInventory) {
  auto host = topology(12, 2);
  host.devices[0].logical_id = 4;
  host.devices[1].logical_id = 8;
  const auto plan = make_cpu_binding_plan(host, 8, true, true, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(all_cpus(*plan), (std::vector<int32_t>{6, 7, 8, 9, 10, 11}));
}

TEST(NpuCpuBindingTest, RejectsInsufficientCpusetBeforeBinding) {
  std::string error;
  EXPECT_FALSE(make_cpu_binding_plan(topology(79, 16), 0, true, true, &error));
  EXPECT_NE(error.find("insufficient"), std::string::npos);
  EXPECT_FALSE(
      make_cpu_binding_plan(topology(47, 16), 15, true, false, nullptr));
  EXPECT_FALSE(
      make_cpu_binding_plan(topology(640, 16), 16, true, false, nullptr));
}

TEST(NpuCpuBindingTest, ParsesTopologyByPhysicalId) {
  auto host = topology(80, 2);
  host.devices[0].physical_id = 10;
  host.devices[1].physical_id = 11;
  const auto affinity = parse_npu_affinity(
      "  Phy-ID10  Phy-ID11  CPU Affinity\n"
      "Phy-ID10  X    SYS    0-19,40-59\n"
      "Phy-ID11  SYS  X      20-39,60-79\n",
      host.devices);
  ASSERT_EQ(affinity.size(), 2U);
  EXPECT_EQ(affinity.at(0).front(), 0);
  EXPECT_EQ(affinity.at(0).back(), 59);
  EXPECT_TRUE(parse_npu_affinity("Phy-ID10 X SIO\n", host.devices).empty());
}

TEST(NpuCpuBindingTest, SharedAffinityIncludesHiddenNpus) {
  auto host = topology(80, 2);
  host.affinity[0] = *parse_cpu_list("0-39");
  host.affinity[1] = *parse_cpu_list("0-39");
  const auto first = make_cpu_binding_plan(host, 0, false, true, nullptr);
  const auto second = make_cpu_binding_plan(host, 1, false, true, nullptr);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->mode, "topo_affinity");
  EXPECT_EQ(all_cpus(*first), *parse_cpu_list("0-39"));
  EXPECT_EQ(all_cpus(*second), *parse_cpu_list("40-79"));
}

TEST(NpuCpuBindingTest, RespectsNonContiguousNumaIds) {
  auto host = topology(80, 2);
  for (auto& entry : host.cpu_nodes) {
    entry.second = entry.first < 40 ? 2 : 7;
  }
  host.affinity[0] = *parse_cpu_list("0-39");
  host.affinity[1] = *parse_cpu_list("0-39");
  const auto plan = make_cpu_binding_plan(host, 1, false, false, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->memory_node, 7);
}

TEST(NpuCpuBindingTest, RejectsIncompleteAndPartiallyOverlappingTopology) {
  auto host = topology(120, 3);
  host.affinity[0] = *parse_cpu_list("0-39");
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, false, true, nullptr));
  host.affinity[1] = *parse_cpu_list("40-79");
  host.affinity[2] = *parse_cpu_list("80-119");
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, false, true, nullptr));
}

TEST(NpuCpuBindingTest, EmptyTopologyUsesGlobalPolicy) {
  const auto plan =
      make_cpu_binding_plan(topology(80, 2), 1, false, true, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->mode, "global_slice");
  EXPECT_EQ(all_cpus(*plan), *parse_cpu_list("40-79"));
}

TEST(NpuCpuBindingTest, EmptyAffinityIntersectionDoesNotWidenCpuset) {
  auto host = topology(80, 2);
  host.allowed_cpus = *parse_cpu_list("40-79");
  host.affinity[0] = *parse_cpu_list("0-39");
  host.affinity[1] = *parse_cpu_list("40-79");
  EXPECT_FALSE(make_cpu_binding_plan(host, 0, false, false, nullptr));
  const auto plan = make_cpu_binding_plan(host, 1, false, false, nullptr);
  ASSERT_TRUE(plan);
  EXPECT_EQ(all_cpus(*plan), host.allowed_cpus);
}

#if defined(__linux__)
TEST(NpuCpuBindingTest, AppliesWorkerAndRuntimeRolesOnlyInChildProcess) {
  cpu_set_t parent_mask;
  ASSERT_EQ(sched_getaffinity(0, sizeof(parent_mask), &parent_mask), 0);
  std::vector<int32_t> cpus;
  cpus.reserve(CPU_COUNT(&parent_mask));
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &parent_mask)) {
      cpus.emplace_back(cpu);
    }
  }
  if (cpus.size() < 3) {
    GTEST_SKIP() << "requires three allowed CPUs";
  }
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    std::atomic<int32_t> ready{0};
    std::atomic<bool> done{false};
    const auto runtime_thread = [&](const char* name) {
      pthread_setname_np(pthread_self(), name);
      ++ready;
      while (!done.load()) {
        std::this_thread::yield();
      }
    };
    std::thread acl(runtime_thread, "acl_thread");
    std::thread release(runtime_thread, "release_thread");
    while (ready.load() != 2) {
      std::this_thread::yield();
    }
    CpuBindingPlan plan;
    plan.worker_cpus = {cpus[0]};
    plan.acl_cpu = cpus[1];
    plan.release_cpu = cpus[2];
    bool success = apply_cpu_binding_plan(plan);
    cpu_set_t mask;
    success &= sched_getaffinity(0, sizeof(mask), &mask) == 0 &&
               CPU_COUNT(&mask) == 1 && CPU_ISSET(cpus[0], &mask);
    success &=
        pthread_getaffinity_np(acl.native_handle(), sizeof(mask), &mask) == 0 &&
        CPU_COUNT(&mask) == 1 && CPU_ISSET(cpus[1], &mask);
    success &= pthread_getaffinity_np(
                   release.native_handle(), sizeof(mask), &mask) == 0 &&
               CPU_COUNT(&mask) == 1 && CPU_ISSET(cpus[2], &mask);
    done = true;
    acl.join();
    release.join();
    _exit(success ? 0 : 1);
  }
  int child_status = 0;
  ASSERT_EQ(waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);
  cpu_set_t after;
  ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
  EXPECT_TRUE(CPU_EQUAL(&parent_mask, &after));
}

TEST(NpuCpuBindingTest, InvalidPlansLeaveAffinityUnchanged) {
  cpu_set_t before;
  ASSERT_EQ(sched_getaffinity(0, sizeof(before), &before), 0);
  CpuBindingPlan plan;
  EXPECT_FALSE(apply_cpu_binding_plan(plan));
  plan.worker_cpus = {CPU_SETSIZE};
  plan.acl_cpu = 0;
  plan.release_cpu = 1;
  EXPECT_FALSE(apply_cpu_binding_plan(plan));
  plan.worker_cpus = {0};
  EXPECT_FALSE(apply_cpu_binding_plan(plan));
  cpu_set_t after;
  ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
  EXPECT_TRUE(CPU_EQUAL(&before, &after));
}
#endif

}  // namespace
}  // namespace xllm::npu
