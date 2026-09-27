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
#include <memory>
#include <mutex>
#include <string>

namespace xllm {

// torch_npu owns a process-wide collector for the current NPU and installs
// thread-local CPU callbacks. One worker per process is required; use the
// native ascend backend for multiple devices in a single process.
class NpuTorchProfiler final {
 public:
  static NpuTorchProfiler& get_instance();

  // Both calls must run on the worker's compute thread, with device work
  // drained.
  bool start(const std::string& profile_dir, int32_t device_id, int32_t rank);
  bool stop(int32_t device_id);

 private:
  class Impl;

  NpuTorchProfiler();
  ~NpuTorchProfiler();
  NpuTorchProfiler(const NpuTorchProfiler&) = delete;
  NpuTorchProfiler& operator=(const NpuTorchProfiler&) = delete;

  std::mutex mutex_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm
