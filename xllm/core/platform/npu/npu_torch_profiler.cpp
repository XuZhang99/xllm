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

#include "core/platform/npu/npu_torch_profiler.h"

#include <glog/logging.h>
#include <pybind11/embed.h>

#include <filesystem>
#include <thread>

namespace py = pybind11;

namespace xllm {

class __attribute__((visibility("hidden"))) NpuTorchProfiler::Impl final {
 public:
  ~Impl() {
    // An unfinished capture cannot be stopped on the static-destructor thread:
    // its CPU callbacks belong to the compute thread. Normal stop releases the
    // object with the GIL held. Leave any unfinished object to process
    // teardown.
    static_cast<void>(profiler_.release());
  }

  py::object profiler_;
  std::thread::id owner_;
  int32_t device_id_ = -1;
  bool running_ = false;
  bool failed_ = false;
  std::string output_dir_;
};

NpuTorchProfiler::NpuTorchProfiler() : impl_(std::make_unique<Impl>()) {}
NpuTorchProfiler::~NpuTorchProfiler() = default;

NpuTorchProfiler& NpuTorchProfiler::get_instance() {
  static NpuTorchProfiler instance;
  return instance;
}

bool NpuTorchProfiler::start(const std::string& profile_dir,
                             int32_t device_id,
                             int32_t rank) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (impl_->profiler_) {
    if (impl_->device_id_ != device_id ||
        impl_->owner_ != std::this_thread::get_id()) {
      LOG(ERROR)
          << "NPU torch profiling requires one worker per process. "
             "Launch one process per NPU or use --profile_backend=ascend.";
      return false;
    }
    return impl_->running_ && !impl_->failed_;
  }
  if (device_id < 0 || !Py_IsInitialized()) {
    LOG(ERROR) << "NPU torch profiling requires a valid device and the Python "
                  "runtime initialized by the NPU server.";
    return false;
  }

  std::error_code error;
  const auto output_dir =
      std::filesystem::absolute(profile_dir.empty() ? "." : profile_dir, error);
  if (!error) {
    std::filesystem::create_directories(output_dir, error);
  }
  if (error) {
    LOG(ERROR) << "Cannot create NPU torch profiling directory: "
               << error.message();
    return false;
  }

  py::gil_scoped_acquire gil;
  try {
    py::module_ autograd = py::module_::import("torch.autograd");
    if (autograd.attr("_profiler_enabled")().cast<bool>()) {
      LOG(ERROR)
          << "A PyTorch profiler is already active on the compute thread.";
      return false;
    }
    py::module_ profiler_module = py::module_::import("torch_npu.profiler");
    py::object activity = profiler_module.attr("ProfilerActivity");
    py::object experimental = profiler_module.attr("_ExperimentalConfig")(
        py::arg("profiler_level") =
            profiler_module.attr("ProfilerLevel").attr("Level1"),
        py::arg("aic_metrics") =
            profiler_module.attr("AiCMetrics").attr("PipeUtilization"),
        py::arg("data_simplification") = false);
    py::object handler = profiler_module.attr("tensorboard_trace_handler")(
        output_dir.string(),
        py::arg("worker_name") = "xllm_rank" + std::to_string(rank));
    if (handler.is_none()) {
      LOG(ERROR) << "torch_npu could not configure the trace output directory: "
                 << output_dir;
      return false;
    }
    impl_->profiler_ = profiler_module.attr("profile")(
        py::arg("activities") =
            py::make_tuple(activity.attr("CPU"), activity.attr("NPU")),
        py::arg("record_shapes") = false,
        py::arg("profile_memory") = false,
        py::arg("with_stack") = false,
        py::arg("with_modules") = false,
        py::arg("experimental_config") = experimental,
        py::arg("on_trace_ready") = handler);
    impl_->device_id_ = device_id;
    impl_->owner_ = std::this_thread::get_id();
    impl_->failed_ = true;
    impl_->profiler_.attr("start")();
    // torch_npu's public API logs some errors instead of raising. Check that
    // the CPU callback was actually installed before acknowledging the start.
    if (!autograd.attr("_profiler_enabled")().cast<bool>()) {
      LOG(ERROR) << "torch_npu.profiler did not enable CPU collection.";
      return false;
    }
    impl_->output_dir_ =
        impl_->profiler_.attr("prof_if").attr("prof_path").cast<std::string>();
    impl_->running_ = true;
    impl_->failed_ = false;
    LOG(INFO) << "NPU torch profiling started: " << impl_->output_dir_;
    return true;
  } catch (const py::error_already_set& error) {
    LOG(ERROR) << "Failed to start torch_npu.profiler: " << error.what();
    return false;
  }
}

bool NpuTorchProfiler::stop(int32_t device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!impl_->profiler_ || impl_->device_id_ != device_id) {
    return true;
  }
  if (impl_->owner_ != std::this_thread::get_id()) {
    LOG(ERROR)
        << "NPU torch profiling must stop on the thread that started it.";
    return false;
  }
  py::gil_scoped_acquire gil;
  try {
    impl_->failed_ = true;
    impl_->profiler_.attr("stop")();
    if (!impl_->profiler_.attr("stopped").cast<bool>() ||
        py::module_::import("torch.autograd")
            .attr("_profiler_enabled")()
            .cast<bool>()) {
      LOG(ERROR) << "torch_npu.profiler did not finish stopping.";
      return false;
    }
    // The handler parses synchronously. Do not report successful export when
    // torch_npu swallowed an analysis error; retain the raw data for diagnosis.
    const auto trace_path = std::filesystem::path(impl_->output_dir_) /
                            "ASCEND_PROFILER_OUTPUT" / "trace_view.json";
    std::error_code error;
    const bool exported =
        impl_->running_ && std::filesystem::is_regular_file(trace_path, error);
    const bool was_running = impl_->running_;
    impl_->profiler_ = py::object();
    impl_->running_ = false;
    impl_->failed_ = false;
    if (!was_running) {
      return true;
    }
    if (!exported) {
      LOG(ERROR) << "NPU torch trace export failed. Raw data: "
                 << impl_->output_dir_;
      return false;
    }
    LOG(INFO) << "NPU torch profiling stopped. Trace: " << trace_path;
    return true;
  } catch (const py::error_already_set& error) {
    LOG(ERROR) << "Failed to stop torch_npu.profiler: " << error.what();
    return false;
  }
}

}  // namespace xllm
