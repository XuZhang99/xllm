# Guide to Adapting xLLM to a New Domestic Accelerator

This document is based on a rough read-through of the `xllm` codebase. It is not meant to explain every implementation detail. The goal is to provide an executable adaptation path for bringing xLLM up on a new domestic accelerator.

Applicable if:

- You already have the chip vendor's compiler toolchain, runtime libraries, and a PyTorch/Libtorch backend.
- You want to get xLLM running on a single card first, then incrementally add performance and distributed support.

Not applicable if:

- You do not yet have a usable PyTorch backend, or the chip side still lacks basic Tensor/Stream/Allocator capabilities.
- You want to replicate the full NPU branch in one shot.

## 1. Bottom line first: where does xLLM need adaptation?

From the repository structure, "chip adaptation" in xLLM mainly falls into five layers:

1. Build layer: decides whether a device backend is enabled.
2. Device layer: decides what device xLLM recognizes, how it selects devices, how it synchronizes, and how it queries memory.
3. Kernel layer: dispatches unified `kernel::xxx(...)` APIs to your backend implementation.
4. Model layer: selects a model implementation for the target backend.
5. Communication / graph / advanced feature layer: adds multi-card, graph execution, KV management, and optimization features.

You can think of it as:

`xllm.cpp` -> `Options` -> `Device/Stream` -> `kernel::ops_api` -> `models/*/<backend>` -> runtime/scheduler/distributed

For a new accelerator, the most important thing is not full feature parity on day one. The first target should be this minimal path:

`build succeeds -> device is recognized -> model loads on a single card -> one prefill/decode round runs -> tokens are returned`

## 2. My rough conclusions after reading the code

### 2.1 Entry point

The entry point is `xllm/xllm.cpp`.

It is responsible for:

- parsing model path and `--backend`
- calling `DeviceNameUtils::parse_devices(...)` to parse `--devices`
- assembling runtime arguments into `Options`
- calling `create_master(...)` to start the inference flow

This means: if device parsing, platform capability, or model construction is not wired through, the service will not start.

### 2.2 Device abstraction

The device abstraction is mainly in:

- `xllm/core/util/device_name_utils.cpp`
- `xllm/core/platform/device.h`
- `xllm/core/platform/device.cpp`
- `xllm/core/platform/stream.h`
- `xllm/core/platform/stream.cpp`

These files unify:

- `device_count()`
- `type_str()`
- `type_torch()`
- `set_device()`
- `init_device_context()`
- `empty_cache()`
- `synchronize_default_stream()`
- `current_stream()`

This layer determines whether the xLLM binary treats your chip as `npu`, `mlu`, `musa`, or some other device name.

### 2.3 Unified kernel entry points

The main kernel entry points are:

- `xllm/core/kernels/ops_api.h`
- `xllm/core/kernels/ops_api.cpp`

These files are critical. Upper-level model and layer code mostly call unified interfaces such as `kernel::matmul(...)`, `kernel::apply_rotary(...)`, `kernel::fused_layernorm(...)`, and `kernel::group_gemm(...)`. Then `ops_api.cpp` dispatches them to concrete backends through compile-time macros:

- `npu::*`
- `mlu::*`
- `cuda::*`
- `ilu::*`
- `musa::*`

So the core work for a new chip backend is usually:

1. add a new `core/kernels/<your_backend>/`
2. wire dispatch in `ops_api.cpp`
3. incrementally implement the kernels actually used by the target model

### 2.4 How models are organized

Model registration and factories are in:

- `xllm/models/model_registry.h`
- `xllm/models/models.h`

`models.h` selects different model sets by compile-time macro. For example:

- under `USE_NPU`, it includes a large number of `llm/npu/*`
- under `USE_MUSA`, it currently includes only `llm/musa/qwen3.h`

This shows that xLLM does not force every device to share the exact same model implementation. It allows backend-specific variants.

If your chip behaves similarly to CUDA/MLU/MUSA, it is better to reuse common model code first.
If your chip needs a different attention path, cache layout, or operator order, it is better to create `models/llm/<your_backend>/`.

### 2.5 Backend maturity is uneven

Judging from the code volume:

- the `npu` branch is the most complete, including models, kernels, platform support, and distributed capabilities
- `cuda` / `mlu` / `ilu` also have relatively complete kernel coverage

This matters for adaptation:

- if your goal is "get it running first", start by learning from a smaller backend
- if your goal is "full-featured, high-performance support", use the `npu` branch as the main reference

## 3. Recommended adaptation strategy

I recommend four phases:

### Phase A: build the minimal backend skeleton

Target:

- it compiles
- it recognizes `--devices="<your_device>:0"`
- it can complete minimal single-card inference

### Phase B: complete the main LLM path

Target:

- at least one mainstream model runs, such as `qwen3`
- prefill + decode work
- basic KV cache read/write works

### Phase C: implement performance-critical kernels

Target:

- attention
- rotary
- norm
- matmul/group gemm
- sampling

### Phase D: add advanced features

Target:

- multi-card communication
- graph mode
- prefix cache / global kvcache
- MTP / speculative / VLM / MoE optimizations

## 4. Implementation steps

## 4.1 Add a build switch

First add a switch in the root `CMakeLists.txt`, for example:

```cmake
option(USE_XPU "Enable XPU support" OFF)
```

Then wire your toolchain, runtime libraries, and headers by following the patterns already used in the repository.

Reference files:

- `CMakeLists.txt`
- `xllm/core/kernels/CMakeLists.txt`
- `xllm/core/platform/CMakeLists.txt`

Suggested approach:

- if the chip uses a dedicated language frontend, follow the `USE_MUSA` path
- if it is a standard C++ extension backend, follow `USE_NPU` / `USE_MLU`

At the end of this step, at minimum:

- `cmake` recognizes the new switch
- the backend-specific directories can be added with `add_subdirectory(...)`
- `platform` and `kernels` can link against your runtime libraries

## 4.2 Integrate the device abstraction

Key files:

- `xllm/core/platform/device.cpp`
- `xllm/core/util/device_name_utils.cpp`

At minimum you need implementations for:

- `Device::device_count()`
- `Device::type_str()`
- `Device::type_torch()`
- `Device::set_device()`
- `Device::init_device_context()`
- `Device::empty_cache()`
- `Device::synchronize_default_stream()`
- `Device::current_stream()`
- `Device::get_device_mem()`

There are two key decisions here.

### Decision 1: what `torch::DeviceType` does your PyTorch backend map to?

In the current code:

- NPU / MLU use `torch::kPrivateUse1`
- CUDA / ILU use `torch::kCUDA`
- MUSA uses `torch::kMUSA`

If your chip backend also uses `PrivateUse1`, pay special attention to:

- whether the device string matches the name you registered in PyTorch
- whether Tensor `.to(device)` and `torch::Device(...)` construction behave correctly

### Decision 2: xLLM assumes one device type per build

`DeviceNameUtils::parse_devices(...)` checks:

- `parts[0] == Device::type_str()`

That means a given build artifact typically recognizes only one primary device type.
If you add `xpu`, you must ensure:

- `--devices="xpu:0"`
- `Device::type_str()` returns `"xpu"`

## 4.3 Integrate Stream and basic platform capability

Besides `device.cpp`, also review:

- `xllm/core/platform/stream.cpp`
- `xllm/core/platform/vmm_api.*`
- `xllm/core/platform/shared_vmm_allocator.*`

Minimum requirement:

- you can obtain the current stream
- you can synchronize the default stream

If your chip does not yet support VMM or unified virtual memory, the recommended approach is:

- in the first stage, make sure the code builds and disable related advanced features
- do not try to fully implement `shared_vmm_allocator` on day one

The reason is simple: the first milestone for xLLM does not depend on every advanced memory feature.

## 4.4 Add a new kernel backend directory

Recommended new directory:

```text
xllm/core/kernels/xpu/
```

At minimum it should contain:

- `CMakeLists.txt`
- `xpu_ops_api.h`
- several kernel implementation files

Then wire your backend into:

- `xllm/core/kernels/CMakeLists.txt`
- `xllm/core/kernels/ops_api.cpp`

Recommended pattern:

```cpp
#elif defined(USE_XPU)
#include "xpu/xpu_ops_api.h"
```

and:

```cpp
#elif defined(USE_XPU)
  xpu::apply_rotary(...);
```

## 4.5 The first batch of kernels you must implement

Do not try to implement everything in `ops_api.h` at once.
Start from a concrete model and implement only the kernels that are actually used in its execution path.

If you start with `qwen3`, the usual priority is:

1. `apply_rotary`
2. `fused_layernorm`
3. `matmul`
4. `reshape_paged_cache`
5. attention-related implementations
6. `apply_top_k_top_p`
7. `random_sample`

If you want MoE support later, add:

1. `group_gemm`
2. `moe_active_topk`
3. `moe_gen_idx`
4. `moe_expand_input`
5. `moe_combine_result`
6. all2all-related interfaces

A very practical method is:

- pick one target model first
- use `rg "kernel::"` to search the related layers / model code
- record the real call chain
- implement only the kernels required by that chain

## 4.6 Choose one minimal model as the template

Recommended first target:

- `qwen3`

Suggested strategy:

- if your chip's tensor semantics are close to the existing common implementation, reuse `models/llm/qwen3.h` first
- if cache layout, rope handling, or attention metadata differ significantly, create `models/llm/xpu/qwen3.h`

## 4.7 Register the model

After adding a model implementation, make sure it is registered and can actually be selected at runtime.

Key files:

- `xllm/models/models.h`
- `xllm/models/model_registry.h`

At minimum you need to do two things:

1. include your model header under the `#elif defined(USE_XPU)` branch in `models.h`
2. use the existing registration macros in the model header, for example:

```cpp
REGISTER_CAUSAL_MODEL(qwen3, QWen3ForCausalLM);
REGISTER_MODEL_ARGS(qwen3, [&] {
  ...
});
```

If you skip this step, the code may compile but runtime will still report that the model type is unsupported.

## 4.8 Decide whether to use backend-specific models or reuse common ones

This is the most important architectural choice in the adaptation.

### Option A: reuse common model code as much as possible

Suitable when:

- your chip's PyTorch backend behaves similarly to CUDA / MLU
- most layer code does not need special branching
- the main differences are in kernel implementations

Advantages:

- less code
- easier to keep up with upstream changes

Disadvantages:

- some performance optimizations are harder to push down

### Option B: maintain `models/llm/xpu/*`

Suitable when:

- attention / cache / rope / graph capture differ substantially
- you need backend-specific tensor layouts
- some layers must call fused vendor ops directly

Advantages:

- more room for performance work
- adaptation logic is more direct

Disadvantages:

- higher maintenance cost

If you do not have strong evidence that a model fork is required, start with Option A.

## 4.9 Disable advanced features first

For the first release of a new backend, I recommend disabling or postponing:

- graph mode
- prefix cache
- global kvcache
- disagg PD
- eplb
- multi-stream parallel
- speculative / mtp
- multi-node communication

The reason is straightforward: these features expand the problem from "are kernels and runtime correct" into "are scheduling, communication, memory, and graph capture all correct together".

The first milestone should be stable single-machine, single-card inference.

## 4.10 Build the communication backend separately

In `xllm/xllm.cpp`, you can see:

- `communication_backend`
- `rank_tablefile`
- `nnodes`
- `dp_size`
- `ep_size`

This indicates that xLLM treats multi-card capability as a separate layer.

Recommended order:

1. single card
2. single machine, multi-card
3. multi-machine

If your chip vendor provides a communication library similar to HCCL / NCCL / LCCL, then implement:

- collective service
- allreduce / all2all related kernels
- rank table / topology configuration

Do not debug "single-card kernel bugs" and "multi-card communication bugs" at the same time in version one.

## 5. A practical minimal implementation plan

If I were starting a new `XPU` backend from scratch, I would go in this order:

1. add `USE_XPU` in the root `CMakeLists.txt`
2. add a `USE_XPU` branch in `xllm/core/platform/device.cpp`
3. add a minimal kernel directory under `xllm/core/kernels/xpu/`
4. wire `USE_XPU` into `xllm/core/kernels/ops_api.cpp`
5. make `matmul`, `fused_layernorm`, `apply_rotary`, `reshape_paged_cache`, and sampling-related interfaces available first
6. add or reuse a `qwen3` model implementation
7. wire `USE_XPU` into `xllm/models/models.h`
8. run a minimal single-card generation example
9. fill in attention and other performance-critical paths
10. only then move on to multi-card, graph mode, and advanced optimizations

## 6. Validation order

I recommend validation in this order.

### 6.1 Build validation

Verify:

- `cmake` succeeds
- `platform` and `kernels` link correctly
- the main `xllm` binary is produced

### 6.2 Device validation

Verify:

- `--devices="xpu:0"` is recognized
- `Device::device_count()` returns a sane value
- tensors can be created successfully after `set_device()`

### 6.3 Minimal tensor validation

Write a few small tests for:

- `torch::randn(...).to("xpu:0")`
- matmul
- layernorm
- rotary
- cache write/read

### 6.4 Single-model validation

Start with the simplest offline generation path:

- `examples/generate.py`
- or launch `xllm/xllm.cpp` directly

The goal at this stage is not performance. It is:

- no crash
- tokens are produced
- output is not obviously broken

### 6.5 Regression validation

After single-card stability, then add:

- long prompts
- batched requests
- multi-round decode
- multi-card
- MoE
- VLM

## 7. Common pitfalls

### 7.1 Device type and PyTorch backend type do not match

If:

- `Device::type_str()` returns `xpu`
- but `torch::DeviceType` or the registered PyTorch device name is not `xpu`

then `.to(device)` or `torch::Device("xpu:0")` usually fails.

### 7.2 `PrivateUse1` backend device names are not aligned

NPU / MLU style backends often use `PrivateUse1`.
If you also use that route, you must make sure that:

- the Python-side device name
- the C++ `torch::Device(...)`
- xLLM's `Device::type_str()`

all match.

### 7.3 Not every model is a good phase-one target

Do not start with:

- DeepSeek MoE
- VLM
- MTP
- DiT

Prefer a regular decoder-only model such as `qwen3`.

### 7.4 Do not blindly copy the entire NPU path

The NPU branch is very complete, but it also relies on many specialized capabilities:

- `torch_npu`
- Ascend runtime
- ATB / custom operators
- communication and graph-mode support

If your chip does not provide these capabilities, copying that path mechanically will increase the adaptation cost significantly.

### 7.5 Do not chase graph mode too early

Graph mode in xLLM is an advanced optimization, not a startup prerequisite.
For a first backend release, make sure the eager path is correct before attempting graph capture.

## 8. Which files I recommend reading first

If you are about to start implementation, read these files first:

### Build and entry

- `CMakeLists.txt`
- `xllm/xllm.cpp`

### Device and platform

- `xllm/core/platform/device.h`
- `xllm/core/platform/device.cpp`
- `xllm/core/platform/stream.h`
- `xllm/core/platform/stream.cpp`
- `xllm/core/util/device_name_utils.cpp`

### Kernel dispatch

- `xllm/core/kernels/ops_api.h`
- `xllm/core/kernels/ops_api.cpp`
- `xllm/core/kernels/CMakeLists.txt`

### Model registration

- `xllm/models/model_registry.h`
- `xllm/models/models.h`

### Reference backends

- model reference: `xllm/models/llm/qwen3.h`
- kernel reference: `xllm/core/kernels/cuda/*`
- platform reference: `xllm/core/platform/device.cpp`

## 9. A realistic conclusion

If your goal is to adapt xLLM to a new domestic accelerator, the real workload is usually not "changing a few macros". It comes down to two questions:

1. is your chip's PyTorch backend complete enough?
2. does your chip already provide high-performance basic implementations for attention, norm, matmul, cache, and communication?

If both answers are yes, xLLM's code structure is suitable for integrating a new backend.
If not, adapting xLLM may effectively turn into "building a missing deep learning runtime along the way", which is a much larger effort.

## 10. Recommended project order

Finally, here is a more practical execution order:

1. decide the device name, PyTorch device type, and compile-time macro
2. wire through `platform/device.cpp`
3. add the kernel dispatch layer
4. choose `qwen3` as the first model
5. make single-card generation work
6. do correctness validation
7. improve attention / KV / sampling performance
8. then add multi-card communication
9. finally add graph mode, MoE, VLM, and speculative inference

If you are actually about to implement the code, I strongly recommend that the first milestone be only:

**"single-machine, single-card Qwen3 generation is stable"**

That is the most reasonable milestone.
