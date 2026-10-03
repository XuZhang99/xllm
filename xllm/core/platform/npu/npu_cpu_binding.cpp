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

#include <fcntl.h>
#include <glog/logging.h>
#include <linux/mempolicy.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

extern char** environ;

namespace xllm::npu {
namespace {

// No shell and a bounded timeout: optional placement discovery must not hang
// server startup when the driver management interface is unavailable.
std::optional<std::string> run_command(std::vector<std::string> arguments) {
  int pipe_fds[2];
  if (pipe2(pipe_fds, O_CLOEXEC) != 0) {
    return std::nullopt;
  }
  arguments.insert(arguments.begin(), {"env", "LC_ALL=C"});
  std::vector<char*> argv;
  argv.reserve(arguments.size() + 1);
  for (auto& argument : arguments) {
    argv.emplace_back(argument.data());
  }
  argv.emplace_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
  posix_spawn_file_actions_addopen(
      &actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
  pid_t child = -1;
  const int32_t status =
      posix_spawnp(&child, "env", &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(pipe_fds[1]);
  if (status != 0) {
    close(pipe_fds[0]);
    return std::nullopt;
  }
  if (fcntl(pipe_fds[0], F_SETFL, O_NONBLOCK) < 0) {
    kill(child, SIGKILL);
    while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
    }
    close(pipe_fds[0]);
    return std::nullopt;
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  std::string output;
  std::array<char, 4096> buffer;
  bool finished = false;
  int child_status = 0;
  while (std::chrono::steady_clock::now() < deadline &&
         output.size() < 1024 * 1024) {
    pollfd descriptor{pipe_fds[0], POLLIN, 0};
    poll(&descriptor, 1, 50);
    ssize_t count = 0;
    while ((count = read(pipe_fds[0], buffer.data(), buffer.size())) > 0) {
      output.append(buffer.data(), static_cast<size_t>(count));
    }
    if (waitpid(child, &child_status, WNOHANG) == child) {
      // The child may have written between the last read and waitpid.
      while ((count = read(pipe_fds[0], buffer.data(), buffer.size())) > 0) {
        output.append(buffer.data(), static_cast<size_t>(count));
      }
      finished = true;
      break;
    }
  }
  if (!finished) {
    kill(child, SIGKILL);
    while (waitpid(child, &child_status, 0) < 0 && errno == EINTR) {
    }
  }
  close(pipe_fds[0]);
  if (!finished || !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    return std::nullopt;
  }
  return output;
}

std::string cpu_list(const std::vector<int32_t>& cpus) {
  std::string result;
  for (int32_t cpu : cpus) {
    if (!result.empty()) {
      result += ',';
    }
    result += std::to_string(cpu);
  }
  return result;
}

cpu_set_t make_mask(const std::vector<int32_t>& cpus) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int32_t cpu : cpus) {
    CPU_SET(cpu, &mask);
  }
  return mask;
}

bool apply_threads(const CpuBindingPlan& plan) {
  struct ThreadAffinity {
    pid_t tid;
    cpu_set_t previous;
    cpu_set_t requested;
  };
  std::vector<ThreadAffinity> threads;
  std::error_code error;
  auto directory =
      std::filesystem::directory_iterator("/proc/self/task", error);
  if (error) {
    return false;
  }
  threads.reserve(256);
  const cpu_set_t worker_mask = make_mask(plan.worker_cpus);
  const cpu_set_t acl_mask = make_mask({plan.acl_cpu});
  const cpu_set_t release_mask = make_mask({plan.release_cpu});
  int32_t acl_threads = 0;
  int32_t release_threads = 0;
  const std::filesystem::directory_iterator end;
  for (; directory != end; directory.increment(error)) {
    if (error) {
      return false;
    }
    const auto& entry = *directory;
    const std::string name = entry.path().filename().string();
    pid_t tid = -1;
    const auto parsed =
        std::from_chars(name.data(), name.data() + name.size(), tid);
    if (parsed.ec != std::errc() || tid <= 0) {
      continue;
    }
    cpu_set_t old_mask;
    if (sched_getaffinity(tid, sizeof(old_mask), &old_mask) != 0) {
      if (errno == ESRCH) {
        continue;
      }
      return false;
    }
    std::string thread_name;
    std::ifstream comm(entry.path() / "comm");
    std::getline(comm, thread_name);
    cpu_set_t target = worker_mask;
    if (thread_name == "acl_thread") {
      target = acl_mask;
      ++acl_threads;
    } else if (thread_name == "release_thread") {
      target = release_mask;
      ++release_threads;
    }
    threads.emplace_back(ThreadAffinity{tid, old_mask, target});
  }
  if (error) {
    return false;
  }
  for (const auto& thread : threads) {
    int32_t status = sched_setaffinity(
        thread.tid, sizeof(thread.requested), &thread.requested);
    if (status != 0 && errno == ESRCH) {
      continue;
    }
    cpu_set_t actual;
    if (status == 0) {
      status = sched_getaffinity(thread.tid, sizeof(actual), &actual);
      if ((status == 0 && CPU_EQUAL(&actual, &thread.requested)) ||
          (status != 0 && errno == ESRCH)) {
        continue;
      }
    }
    LOG(WARNING) << "NPU CPU binding could not set/verify affinity for tid="
                 << thread.tid << "; restoring the previous affinity";
    for (const auto& restore : threads) {
      if (sched_setaffinity(
              restore.tid, sizeof(restore.previous), &restore.previous) != 0 &&
          errno != ESRCH) {
        LOG(WARNING) << "Failed to restore CPU affinity for tid="
                     << restore.tid;
      }
    }
    return false;
  }
  LOG(INFO) << "NPU CPU binding applied: logical_npu=" << plan.logical_id
            << " mode=" << plan.mode
            << " worker_cpus=" << cpu_list(plan.worker_cpus)
            << " acl_cpu=" << plan.acl_cpu
            << " release_cpu=" << plan.release_cpu
            << " threads=" << threads.size() << " acl_threads=" << acl_threads
            << " release_threads=" << release_threads;
  return true;
}

void apply_memory_policy(
    const CpuBindingPlan& plan,
    const std::unordered_map<int32_t, int32_t>& cpu_nodes) {
  if (plan.memory_node < 0) {
    LOG(INFO) << "NPU CPU binding: no NUMA node available for memory placement";
    return;
  }
  int32_t max_node = plan.memory_node;
  for (const auto& entry : cpu_nodes) {
    max_node = std::max(max_node, entry.second);
  }
  constexpr size_t kBits = sizeof(unsigned long) * 8;
  std::vector<unsigned long> source(static_cast<size_t>(max_node) / kBits + 1,
                                    0);
  std::vector<unsigned long> target(source.size(), 0);
  for (const auto& entry : cpu_nodes) {
    if (entry.second >= 0) {
      source[entry.second / kBits] |= 1UL << (entry.second % kBits);
    }
  }
  target[plan.memory_node / kBits] |= 1UL << (plan.memory_node % kBits);
  // Linux get_nodes() decrements maxnode before copying the bitmap. Pass the
  // allocated bit capacity plus one, as libnuma does, so the highest node is
  // not silently discarded (an empty MPOL_PREFERRED mask means local memory).
  const unsigned long maxnode = source.size() * kBits + 1;
  // Prefer local memory without imposing a strict, potentially OOM-inducing
  // membind. Done on the startup thread before model/pinned host allocations.
  if (syscall(SYS_set_mempolicy, MPOL_PREFERRED, target.data(), maxnode) != 0) {
    LOG(WARNING) << "NPU CPU binding: memory policy unavailable: "
                 << strerror(errno);
    return;
  }
  const long remaining = syscall(
      SYS_migrate_pages, getpid(), maxnode, source.data(), target.data());
  if (remaining != 0) {
    LOG(WARNING) << "NPU CPU binding: memory migration incomplete: result="
                 << remaining
                 << (remaining < 0 ? std::string(" error=") + strerror(errno)
                                   : "");
  }
  LOG(INFO) << "NPU CPU binding: preferred memory node=" << plan.memory_node;
}

void apply_irq_affinity(const CpuBindingPlan& plan, const NpuIdentity& device) {
  if (plan.irq_cpus.empty()) {
    return;
  }
  const auto board = run_command({"npu-smi",
                                  "info",
                                  "-t",
                                  "board",
                                  "-i",
                                  std::to_string(device.card_id),
                                  "-c",
                                  std::to_string(device.chip_id)});
  if (!board) {
    LOG(WARNING) << "NPU CPU binding: IRQ board discovery unavailable";
    return;
  }
  std::istringstream lines(*board);
  std::string line;
  std::string pci;
  while (std::getline(lines, line)) {
    if (line.find("PCIe Bus Info") == std::string::npos) {
      continue;
    }
    std::istringstream fields(line);
    while (fields >> pci) {
    }
    break;
  }
  std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char value) {
    return std::tolower(value);
  });
  if (pci.find_first_not_of("0123456789abcdef:.") != std::string::npos ||
      pci.empty()) {
    LOG(WARNING) << "NPU CPU binding: no usable PCI address for IRQ binding";
    return;
  }
  const std::filesystem::path irq_directory =
      "/sys/bus/pci/devices/" + pci + "/msi_irqs";
  std::ifstream interrupts("/proc/interrupts");
  std::array<std::string, 2> irqs;
  while (std::getline(interrupts, line)) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::istringstream number(line.substr(0, colon));
    std::string irq;
    number >> irq;
    std::error_code error;
    if (!std::filesystem::exists(irq_directory / irq, error) || error) {
      continue;
    }
    if (line.find("sq_send_trigger_irq") != std::string::npos) {
      irqs[0] = irq;
    } else if (line.find("cq_update_irq") != std::string::npos) {
      irqs[1] = irq;
    }
  }
  if (irqs[0].empty() || irqs[1].empty()) {
    LOG(WARNING) << "NPU CPU binding: SQ/CQ IRQs could not be resolved for PCI "
                 << pci;
    return;
  }
  std::array<std::string, 2> previous;
  for (size_t index = 0; index < irqs.size(); ++index) {
    const std::string path = "/proc/irq/" + irqs[index] + "/smp_affinity_list";
    std::ifstream input(path);
    if (access(path.c_str(), W_OK) != 0 ||
        !std::getline(input, previous[index])) {
      LOG(WARNING) << "NPU CPU binding: IRQ affinity is not writable";
      return;
    }
  }
  for (size_t index = 0; index < irqs.size(); ++index) {
    std::ofstream output("/proc/irq/" + irqs[index] + "/smp_affinity_list");
    output << plan.irq_cpus[index] << std::flush;
    if (!output) {
      for (size_t restore = 0; restore < index; ++restore) {
        std::ofstream rollback("/proc/irq/" + irqs[restore] +
                               "/smp_affinity_list");
        rollback << previous[restore];
      }
      LOG(WARNING) << "NPU CPU binding: IRQ affinity write failed";
      return;
    }
  }
  LOG(INFO) << "NPU CPU binding: IRQ CPUs=" << cpu_list(plan.irq_cpus)
            << " SQ=" << irqs[0] << " CQ=" << irqs[1];
}

}  // namespace

bool apply_cpu_binding_plan(const CpuBindingPlan& plan) {
  if (plan.worker_cpus.empty() || plan.acl_cpu < 0 || plan.release_cpu < 0 ||
      plan.acl_cpu >= CPU_SETSIZE || plan.release_cpu >= CPU_SETSIZE ||
      plan.acl_cpu == plan.release_cpu) {
    return false;
  }
  for (int32_t cpu : plan.worker_cpus) {
    if (cpu < 0 || cpu >= CPU_SETSIZE || cpu == plan.acl_cpu ||
        cpu == plan.release_cpu) {
      return false;
    }
  }
  return apply_threads(plan);
}

NpuCpuBinding& NpuCpuBinding::get_instance() {
  static NpuCpuBinding instance;
  return instance;
}

void NpuCpuBinding::initialize(int32_t device_index,
                               const std::string& soc_name,
                               bool bind_irq) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (plan_) {
    return;
  }
#if !defined(__aarch64__)
  LOG(WARNING) << "NPU CPU binding is supported on aarch64 hosts only";
  return;
#endif
  // Ascend 950 uses a different cluster/UVB policy; do not silently apply the
  // A2/A3 thread-role layout to it.
  if (soc_name.find("950") != std::string::npos) {
    LOG(WARNING)
        << "NPU CPU binding: Ascend 950 cluster placement is not supported";
    return;
  }
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    LOG(WARNING) << "NPU CPU binding: cannot read startup cpuset: "
                 << strerror(errno);
    return;
  }
  CpuBindingTopology topology;
  topology.allowed_cpus.reserve(CPU_COUNT(&allowed));
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      topology.allowed_cpus.emplace_back(cpu);
    }
  }
  const auto inventory = run_command({"npu-smi", "info", "-m"});
  if (!inventory) {
    LOG(WARNING) << "NPU CPU binding: NPU inventory query failed";
    return;
  }
  topology.devices = parse_npu_inventory(*inventory);
  const char* visible = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  const auto id = resolve_npu_logical_id(
      device_index, visible == nullptr ? "" : visible, topology.devices);
  if (!id) {
    LOG(WARNING) << "NPU CPU binding: could not resolve runtime device "
                 << device_index;
    return;
  }
  const bool global_slice = soc_name.find("910_93") != std::string::npos ||
                            soc_name.find("910C") != std::string::npos;
  if (!global_slice) {
    const auto affinity = run_command({"npu-smi", "info", "-t", "topo"});
    if (affinity) {
      topology.affinity = parse_npu_affinity(*affinity, topology.devices);
    }
    if (topology.affinity.empty()) {
      LOG(INFO) << "NPU CPU binding: topology affinity unavailable; using "
                   "global_slice";
    }
  }
  const auto numa = run_command({"lscpu", "-e=CPU,NODE"});
  if (numa) {
    std::istringstream lines(*numa);
    std::string line;
    while (std::getline(lines, line)) {
      std::istringstream fields(line);
      int32_t cpu = -1;
      int32_t node = -1;
      if (fields >> cpu >> node && cpu >= 0 && node >= 0) {
        topology.cpu_nodes.emplace(cpu, node);
      }
    }
  }
  std::string error;
  auto plan =
      make_cpu_binding_plan(topology, *id, global_slice, bind_irq, &error);
  if (!plan || !apply_cpu_binding_plan(*plan)) {
    LOG(WARNING) << "NPU CPU binding skipped: " << error;
    return;
  }
  apply_memory_policy(*plan, topology.cpu_nodes);
  const auto device =
      std::find_if(topology.devices.begin(),
                   topology.devices.end(),
                   [&id](const auto& item) { return item.logical_id == *id; });
  apply_irq_affinity(*plan, *device);
  plan_ = std::move(plan);
}

void NpuCpuBinding::refresh_threads() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (plan_ && !apply_cpu_binding_plan(*plan_)) {
    LOG(WARNING) << "NPU CPU binding: could not refresh thread placement";
  }
}

void NpuCpuBinding::refresh_after_first_forward() {
  std::call_once(first_forward_, [this]() { refresh_threads(); });
}

}  // namespace xllm::npu
