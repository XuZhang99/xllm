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

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <future>
#include <string>

namespace py = pybind11;

namespace {

class NpuTorchProfilerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!Py_IsInitialized()) {
      py::initialize_interpreter(/*init_signal_handlers=*/false);
      PyEval_SaveThread();
    }
  }

  void SetUp() override {
    output_dir_ = (std::filesystem::temp_directory_path() /
                   ("xllm_torch_profiler_test_" + std::to_string(getpid())))
                      .string();
    std::filesystem::remove_all(output_dir_);
    py::gil_scoped_acquire gil;
    py::exec(R"PY(
import sys
import types
from pathlib import Path

original_modules = {name: sys.modules.get(name) for name in
                    ('torch', 'torch.autograd', 'torch_npu', 'torch_npu.profiler')}
state = types.SimpleNamespace(enabled=False, start_calls=0, stop_calls=0,
    handler_error=False, constructor_error=False, start_error=False, silent_start=False,
    stop_error=False, silent_stop=False, export_error=False, paths=[])
module = types.ModuleType('torch_npu.profiler')
autograd = types.ModuleType('torch.autograd')
autograd._profiler_enabled = lambda: state.enabled
module.ProfilerActivity = types.SimpleNamespace(CPU='CPU', NPU='NPU')
module.ProfilerLevel = types.SimpleNamespace(Level1='Level1')
module.AiCMetrics = types.SimpleNamespace(PipeUtilization='PipeUtilization')
module._ExperimentalConfig = lambda **kwargs: kwargs

def handler(directory: str, worker_name: str):
    if state.handler_error:
        return None
    state.directory = directory
    state.worker_name = worker_name
    return lambda profiler: None

class FakeProfile:
    def __init__(self, **kwargs) -> None:
        if state.constructor_error:
            raise RuntimeError('constructor failure')
        assert kwargs['activities'] == ('CPU', 'NPU')
        assert not kwargs['with_stack'] and not kwargs['profile_memory']
        self.stopped = False
        self.prof_if = types.SimpleNamespace(prof_path='')

    def start(self) -> None:
        state.start_calls += 1
        path = Path(state.directory) / (state.worker_name + str(state.start_calls))
        self.prof_if.prof_path = str(path)
        state.paths.append(str(path))
        state.enabled = not state.silent_start
        if state.start_error:
            raise RuntimeError('partial start failure')

    def stop(self) -> None:
        state.stop_calls += 1
        if state.stop_error:
            raise RuntimeError('stop failure')
        if state.silent_stop:
            return
        state.enabled = False
        self.stopped = True
        if not state.export_error:
            path = Path(self.prof_if.prof_path) / 'ASCEND_PROFILER_OUTPUT'
            path.mkdir(parents=True, exist_ok=True)
            (path / 'trace_view.json').write_text('{}')

module.profile = FakeProfile
module.tensorboard_trace_handler = handler
torch_module = types.ModuleType('torch')
torch_module.autograd = autograd
npu_module = types.ModuleType('torch_npu')
npu_module.profiler = module
sys.modules.update({'torch': torch_module, 'torch.autograd': autograd,
                    'torch_npu': npu_module, 'torch_npu.profiler': module})
)PY");
  }

  void TearDown() override {
    set_flag("stop_error", false);
    set_flag("silent_stop", false);
    set_flag("export_error", false);
    EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
    std::filesystem::remove_all(output_dir_);
    py::gil_scoped_acquire gil;
    py::exec(R"PY(
for name, original in original_modules.items():
    if original is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = original
)PY");
  }

  void set_flag(const std::string& name, bool value) {
    py::gil_scoped_acquire gil;
    py::globals()["state"].attr(name.c_str()) = value;
  }

  int32_t count(const std::string& name) {
    py::gil_scoped_acquire gil;
    return py::globals()["state"].attr(name.c_str()).cast<int32_t>();
  }

  xllm::NpuTorchProfiler& profiler_ = xllm::NpuTorchProfiler::get_instance();
  std::string output_dir_;
};

TEST_F(NpuTorchProfilerTest, RepeatedWindowsAndDuplicateCalls) {
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  for (int32_t window = 0; window < 2; ++window) {
    ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/7));
    EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/7));
    EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
    EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  }
  EXPECT_EQ(count("start_calls"), 2);
  EXPECT_EQ(count("stop_calls"), 2);
  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(output_dir_) / "xllm_rank71" /
      "ASCEND_PROFILER_OUTPUT/trace_view.json"));
  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(output_dir_) / "xllm_rank72" /
      "ASCEND_PROFILER_OUTPUT/trace_view.json"));
}

TEST_F(NpuTorchProfilerTest, RejectsAnotherWorkerWithoutStoppingTheOwner) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  auto other = std::async(std::launch::async, [this]() {
    EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/1, /*rank=*/1));
    EXPECT_TRUE(profiler_.stop(/*device_id=*/1));
    EXPECT_FALSE(profiler_.stop(/*device_id=*/0));
  });
  other.get();
  EXPECT_EQ(count("stop_calls"), 0);
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
}

TEST_F(NpuTorchProfilerTest, RejectsInvalidDirectoryAndDevice) {
  std::filesystem::create_directories(output_dir_);
  const auto file = std::filesystem::path(output_dir_) / "file";
  std::ofstream(file) << "not a directory";
  EXPECT_FALSE(profiler_.start(file.string(), /*device_id=*/0, /*rank=*/0));
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/-1, /*rank=*/0));
  EXPECT_EQ(count("start_calls"), 0);
}

TEST_F(NpuTorchProfilerTest, DoesNotTakeOverAnExistingProfiler) {
  set_flag("enabled", true);
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  EXPECT_EQ(count("stop_calls"), 0);
  set_flag("enabled", false);
}

TEST_F(NpuTorchProfilerTest, ConstructorFailureAllowsNewStart) {
  set_flag("constructor_error", true);
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  set_flag("constructor_error", false);
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
}

TEST_F(NpuTorchProfilerTest, RejectsSilentHandlerFailureBeforeStarting) {
  set_flag("handler_error", true);
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  EXPECT_EQ(count("start_calls"), 0);
  set_flag("handler_error", false);
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
}

TEST_F(NpuTorchProfilerTest, PartialStartFailureCanBeCleanedUp) {
  set_flag("start_error", true);
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  set_flag("start_error", false);
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
}

TEST_F(NpuTorchProfilerTest, SilentStartFailureIsNotAcknowledged) {
  set_flag("silent_start", true);
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  set_flag("silent_start", false);
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
}

TEST_F(NpuTorchProfilerTest, StopFailureRetainsCaptureForCleanup) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  set_flag("stop_error", true);
  EXPECT_FALSE(profiler_.stop(/*device_id=*/0));
  set_flag("stop_error", false);
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
}

TEST_F(NpuTorchProfilerTest, SilentStopFailureIsNotAcknowledged) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  set_flag("silent_stop", true);
  EXPECT_FALSE(profiler_.stop(/*device_id=*/0));
  set_flag("silent_stop", false);
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
}

TEST_F(NpuTorchProfilerTest, ReportsMissingExportAndAllowsNextWindow) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
  set_flag("export_error", true);
  EXPECT_FALSE(profiler_.stop(/*device_id=*/0));
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  set_flag("export_error", false);
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0, /*rank=*/0));
}

}  // namespace
