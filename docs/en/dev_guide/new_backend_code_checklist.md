# xLLM New Backend Code Modification Checklist

This document complements `docs/zh/dev_guide/adapt_new_chip.md`. Its purpose is not to explain the design, but to provide a more implementation-oriented checklist. Assume the target backend is named `XPU` for now, and replace it with the real device name when you actually implement it.

Recommended execution order:

- first make it compile, recognize the device, and generate on a single card
- then fill in key kernels, models, and performance work
- finally add multi-card, graph mode, KV, and other advanced features

---

## 0. Define the target

For the first version, the target should be only:

**single-machine, single-card, single-model, offline generation works**

More concretely:

- `xllm` can be built
- `--devices="xpu:0"` is recognized
- one decoder-only model such as `qwen3` can be loaded
- prefill + decode run successfully
- the system returns reasonable tokens

Do not set the initial target to:

- multi-node / multi-card
- MoE
- VLM
- DiT
- graph mode
- speculative / MTP

---

## 1. Build-layer changes

### 1.1 Root CMake switch

File:

- `CMakeLists.txt`

Tasks:

- add `option(USE_XPU "Enable XPU support" OFF)`
- wire in your compiler, toolchain, runtime, headers, and libraries around `project(...)`
- if a dedicated language frontend or special compiler flags are required, follow `USE_MUSA`
- if it is a standard C++ runtime integration, follow `USE_NPU` / `USE_MLU`

Acceptance criteria:

- `cmake -DUSE_XPU=ON ...` configures successfully
- existing `USE_NPU/USE_CUDA/USE_MLU/USE_MUSA/USE_ILU` flows are not broken

### 1.2 Submodule CMake files

Files:

- `xllm/core/kernels/CMakeLists.txt`
- `xllm/core/layers/CMakeLists.txt`
- `xllm/core/platform/CMakeLists.txt`

Tasks:

- in `core/kernels/CMakeLists.txt`, add `add_subdirectory(xpu)` under `USE_XPU`
- link `xpu_kernels` into the top-level `kernels` target
- in `core/layers/CMakeLists.txt`, add `add_subdirectory(xpu)` under `USE_XPU`
- link `xpu_layers` into the top-level `layers` target
- add XPU runtime dependencies to the `platform` target in `core/platform/CMakeLists.txt`

Acceptance criteria:

- `platform` links against the XPU runtime
- `kernels` links against the XPU kernel library
- `layers` links against the XPU layer library

### 1.3 Add backend directories

Recommended new directories:

```text
xllm/core/kernels/xpu/
xllm/core/layers/xpu/
```

```text
xllm/models/llm/xpu/
```

If platform-specific helpers are needed, you can also add:

```text
xllm/core/platform/xpu/
```

Acceptance criteria:

- the new directories are wired into CMake
- the codebase still compiles at skeleton level even before full implementation

---

## 2. Device and platform-layer changes

### 2.1 Device recognition

Files:

- `xllm/core/platform/device.h`
- `xllm/core/platform/device.cpp`
- `xllm/core/util/device_name_utils.cpp`

Tasks:

- add a `#elif defined(USE_XPU)` branch in `device.cpp`
- implement:
  - `Device::device_count()`
  - `Device::type_str()`
  - `Device::type_torch()`
  - `Device::set_device()`
  - `Device::init_device_context()`
  - `Device::empty_cache()`
  - `Device::synchronize_default_stream()`
  - `Device::current_stream()`
  - `Device::get_device_mem()`

Key decisions:

- whether the device string should be `xpu`
- whether PyTorch maps it to `torch::kPrivateUse1`, a dedicated type, or something else

Notes:

- `DeviceNameUtils::parse_devices(...)` checks that the device prefix equals `Device::type_str()`
- one build artifact usually supports only one primary device type

Acceptance criteria:

- `--devices="xpu:0"` parses correctly
- `Device::device_count()` returns the correct number of devices
- `torch::Tensor(...).to(torch::Device("xpu:0"))` works

### 2.2 Stream abstraction

Files:

- `xllm/core/platform/stream.h`
- `xllm/core/platform/stream.cpp`

Tasks:

- wire `Stream::Stream(...)` to the XPU stream pool
- wire `Stream::synchronize()` to XPU stream synchronization
- wire `set_stream_guard()` to your c10 stream wrapper
- verify whether `wait_stream(...)` event record/block behavior works correctly on XPU

Notes:

- `stream.cpp` currently wraps `CUDA/ILU/MUSA` differently from `NPU/MLU`
- if your backend does not provide a complete c10 stream wrapper, you may need to add one

Acceptance criteria:

- streams can be created
- streams can be switched
- streams can wait on each other

### 2.3 Memory and VMM capability

Files:

- `xllm/core/platform/vmm_api.h`
- `xllm/core/platform/vmm_api.cpp`
- `xllm/core/platform/shared_vmm_allocator.h`
- `xllm/core/platform/shared_vmm_allocator.cpp`

Recommendation for the first version:

- if the chip does not yet support VMM, do not force a full implementation
- first confirm which paths strongly depend on VMM, then turn those features off via feature flags

Acceptance criteria:

- the basic single-card path is not blocked by missing VMM support

---

## 3. Entry point and runtime-argument changes

### 3.1 Main entry startup path

File:

- `xllm/xllm.cpp`

What to check:

- whether `FLAGS_devices` accepts `xpu:*`
- whether device and backend information in `Options` flows correctly into model creation and runtime

Usually this file does not need major changes, but you should still check:

- whether some flags are hard-coded for NPU only
- whether some features should be force-disabled on XPU

Recommended features to disable explicitly in version one:

- `enable_graph`
- `enable_prefill_piecewise_graph`
- `enable_graph_vmm_pool`
- `enable_prefix_cache`
- `enable_multi_stream_parallel`
- `enable_xtensor`
- `enable_manual_loader`
- `enable_rolling_load`

Acceptance criteria:

- startup does not fail immediately due to feature validation

### 3.2 Global flags and conditional logic

Files:

- `xllm/core/common/global_flags.h`
- `xllm/core/common/global_flags.cpp`

Tasks:

- check whether there are device-specific default values hard-coded for other backends
- check whether XPU-specific flags are needed
- check whether help text should mention XPU

Acceptance criteria:

- help text matches actual backend capability

---

## 4. Kernel dispatch-layer changes

### 4.1 Add an XPU kernel API

Files:

- `xllm/core/kernels/ops_api.h`
- `xllm/core/kernels/ops_api.cpp`
- `xllm/core/kernels/xpu/xpu_ops_api.h`

Tasks:

- add `#elif defined(USE_XPU)` in `ops_api.cpp`
- dispatch unified APIs to `xpu::xxx(...)`

Typical form:

```cpp
#elif defined(USE_XPU)
#include "xpu/xpu_ops_api.h"
```

and:

```cpp
#elif defined(USE_XPU)
  xpu::fused_layernorm(...);
```

Acceptance criteria:

- `ops_api.cpp` builds successfully
- unimplemented interfaces may temporarily use `NOT_IMPLEMENTED()`, but the minimal model path must not hit them

### 4.2 First batch of required kernels

Start with a simple LLM path. Do not try to cover all of `ops_api.h`.

Recommended first batch:

- `apply_rotary`
- `fused_layernorm`
- `matmul`
- `reshape_paged_cache`
- attention-related core interfaces
- `apply_top_k_top_p`
- `random_sample`

If the first target model does not use some kernels, they can be postponed.

Acceptance criteria:

- the minimal model inference path does not fall into `NOT_IMPLEMENTED()`

### 4.3 Second batch of kernels

Add these only when you are ready for performance work and more complex models:

- `group_gemm`
- `scaled_quantize`
- `scaled_matmul`
- `rejection_sample`
- `gather_split`
- `fused_mla_q`
- `fused_mla_kv`
- `fused_indexer_q`
- `fused_indexer_k`

### 4.4 MoE and all2all family

Do not include these in phase one:

- `moe_active_topk`
- `moe_gen_idx`
- `moe_expand_input`
- `moe_combine_result`
- `moe_all2all_*`
- `moe_init_routing_v2`

Only implement them when you explicitly decide to support MoE models.

---

## 5. Model-layer changes

### 5.1 Choose the model integration strategy

File:

- `xllm/models/models.h`

You need to decide early:

- reuse common model implementations
- or add dedicated `models/llm/xpu/*`

Recommendation:

- if the chip behaves similarly to CUDA / MLU, try to reuse common models first
- if attention, cache layout, or rope processing differ substantially, create the `xpu` directory

### 5.2 Recommended first model

Recommended first target model:

- `qwen3`

Reference files:

- `xllm/models/llm/qwen3.h`
- `xllm/models/llm/musa/qwen3.h`
- `xllm/models/llm/npu/qwen3.h`

Why:

- the execution path is clear
- multiple backend implementations already exist in the repository
- the MUSA version is close to a "minimal new-backend template"

### 5.3 Model registration

Files:

- `xllm/models/models.h`
- `xllm/models/model_registry.h`
- `xllm/models/model_registry.cpp`

Tasks:

- add a `#elif defined(USE_XPU)` include branch in `models.h`
- add these in the model header:
  - `REGISTER_CAUSAL_MODEL(...)`
  - `REGISTER_MODEL_ARGS(...)`

Check these as needed:

- `ModelRegistry::get_model_backend(...)`
- `resolve_model_registration(...)`

Acceptance criteria:

- the model type is recognized
- runtime does not report `Model is not supported currently`

### 5.4 ModelContext and tensor options

Files:

- `xllm/core/framework/model_context.cpp`
- `xllm/core/framework/model/*.h`

Tasks:

- verify that dtype, tensor options, and device moves are correct on XPU
- check whether backend-specific branches need extension

Acceptance criteria:

- model parameters load correctly onto XPU
- intermediate tensors in forward do not accidentally land on CPU or another device

---

## 6. Layer and attention-path changes

### 6.1 Trace the real model call path first

Before implementation, do a call-chain review:

- choose `qwen3`
- find the decoder layer used by that model
- find the kernels and attention implementation used by that decoder layer

Recommended search:

```bash
rg "kernel::" xllm/core/layers xllm/models
```

### 6.2 Attention-related focus points

Important directories:

- `xllm/core/layers/`
- `xllm/core/kernels/`

What to verify:

- prefill path
- decode path
- paged KV cache layout
- rope input format
- dtype of position ids
- block size assumptions

This area is usually where a new backend fails most often.

Acceptance criteria:

- single-batch short-text output is normal
- multi-round decode does not crash
- long prompts do not show obvious cache corruption

---

## 7. Runtime and distributed-layer changes

### 7.1 Usually no major change is needed in the single-card phase

Files:

- `xllm/core/distributed_runtime/master.cpp`
- `xllm/core/distributed_runtime/master.h`

For the first backend version, you usually do not need to change the backend classification in `create_master(...)`, because it classifies task types:

- `llm`
- `vlm`
- `dit`
- `rec`

not device types.

### 7.2 Do communication later

After single-card stability, then review:

- `xllm/core/framework/parallel_state/collective_communicator.*`
- `xllm/core/distributed_runtime/worker_server.cpp`
- `xllm/core/distributed_runtime/dist_manager.cpp`

Tasks:

- evaluate whether to integrate an XPU communication library
- evaluate whether allreduce / allgather / all2all are supported
- evaluate rank table, topology discovery, and process-group initialization

Recommendation for version one:

- do not do it yet

---

## 8. Test and validation checklist

### 8.1 Build validation

- `USE_XPU=ON` builds end to end
- other backends remain unaffected

### 8.2 Device validation

- `device_count()` is correct
- `--devices="xpu:0"` is correct
- tensors can be constructed on XPU

### 8.3 Kernel unit tests

At minimum, validate:

- matmul
- layernorm
- rotary
- paged cache write/read
- sampling

### 8.4 End-to-end validation

Start with:

- `examples/generate.py`
- a simple prompt
- short-text decode

Then run:

- long prompts
- batch > 1
- multi-turn conversation

### 8.5 Stability validation

At minimum, verify:

- repeated load/unload cycles
- memory is reclaimed correctly
- stream synchronization has no intermittent errors
- no cache corruption during multi-round decode

---

## 9. Recommended PR split

To reduce regression risk, split the work in this order.

### PR 1: build skeleton

- root `CMakeLists.txt`
- `core/kernels/CMakeLists.txt`
- `core/platform/CMakeLists.txt`
- empty `xpu/` directories and basic headers

### PR 2: device and stream

- `device.cpp`
- `stream.cpp`
- device recognition and minimal runtime integration

### PR 3: minimal kernel path

- `ops_api.cpp`
- `xpu_ops_api.h`
- minimum required kernels

### PR 4: single-model `qwen3` integration

- `models.h`
- `models/llm/xpu/qwen3.h` or common-model adjustments

### PR 5: end-to-end validation and fixes

- fix bugs
- add minimal tests
- fix runtime argument and device-move issues

### PR 6: performance and advanced features

- attention optimization
- graph
- multi-card
- MoE

---

## 10. Pitfalls to avoid in version one

- do not implement the new backend and multi-card communication at the same time
- do not implement MoE / VLM / DiT in the first version
- do not assume the entire NPU path can be copied directly
- do not declare a large number of interfaces in `ops_api.h` without prioritizing them by the actual call path
- do not overlook consistency between `Device::type_str()` and the PyTorch device name
- do not overlook paged KV cache layout differences

---

## 11. A minimal checkbox list

- [ ] add `USE_XPU` in the root `CMakeLists.txt`
- [ ] wire `xpu_kernels` into `core/kernels/CMakeLists.txt`
- [ ] wire the XPU runtime into `core/platform/CMakeLists.txt`
- [ ] add `core/kernels/xpu/`
- [ ] add `USE_XPU` in `device.cpp`
- [ ] add `USE_XPU` in `stream.cpp`
- [ ] make `DeviceNameUtils` parse `xpu:0`
- [ ] wire `xpu::...` into `ops_api.cpp`
- [ ] implement `matmul`
- [ ] implement `fused_layernorm`
- [ ] implement `apply_rotary`
- [ ] implement `reshape_paged_cache`
- [ ] implement the minimal attention path
- [ ] implement `apply_top_k_top_p`
- [ ] implement `random_sample`
- [ ] integrate `qwen3`
- [ ] register the XPU model in `models.h`
- [ ] make single-card `generate.py` run
- [ ] make long prompts run
- [ ] make multi-round decode run

If all items in this checklist are completed, you effectively have a usable first version of a new xLLM backend.
