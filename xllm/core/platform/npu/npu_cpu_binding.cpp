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
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

#include "core/platform/numa_utils.h"

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
  if (numa::bind_process_to_cpus(plan.worker_cpus,
                                 {{"acl_thread", {plan.acl_cpu}},
                                  {"release_thread", {plan.release_cpu}}}) !=
      0) {
    return false;
  }
  LOG(INFO) << "NPU CPU binding applied: logical_npu=" << plan.logical_id
            << " mode=" << plan.mode
            << " worker_cpus=" << cpu_list(plan.worker_cpus)
            << " acl_cpu=" << plan.acl_cpu
            << " release_cpu=" << plan.release_cpu;
  return true;
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
  CpuBindingTopology topology;
  topology.allowed_cpus = numa::get_thread_cpus();
  if (topology.allowed_cpus.empty()) {
    LOG(WARNING) << "NPU CPU binding: cannot read startup cpuset";
    return;
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
  topology.cpu_nodes = numa::get_cpu_numa_nodes();
  std::string error;
  auto plan =
      make_cpu_binding_plan(topology, *id, global_slice, bind_irq, &error);
  if (!plan || !apply_cpu_binding_plan(*plan)) {
    LOG(WARNING) << "NPU CPU binding skipped: " << error;
    return;
  }
  if (numa::bind_memory_to_numa_node(plan->memory_node,
                                     numa::MemoryPolicy::PREFERRED) == 0) {
    LOG(INFO) << "NPU CPU binding: preferred memory node=" << plan->memory_node;
  }
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
