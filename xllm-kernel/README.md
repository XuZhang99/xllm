<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# xLLM Kernel

`xllm-kernel` contains explicit operator contracts, NPU backend adapters,
preparation, and independent numerical and performance tools. Runtime imports
are lazy: package and suite discovery do not initialize Torch or a device.
The current operator inventory follows the GLM-5.3 PyTorch graph implementation
in `glm5_2.py`, its shared `deepseek_v32.py` path, layers, and NPU wrappers.

This change is a **source-only draft**: its new backends, unit tests, numerical
cases and benchmarks have **not been compiled or run**. Registration is not
runtime support evidence. Existing model calls are unchanged; only the earlier
RMSNorm integration is automatically selected by host startup.

## Install and prepare

From the repository root, in the matching Torch/torch_npu environment:

```bash
python -m pip install --no-deps -e ./xllm-kernel
```

`xllm_native` and `tilelang` require the matching xLLM native extension and,
for TileLang, its compiled AOT variants. Build the host separately when needed
with `python setup.py build`. No CMake/CTest target runs these package tests.
`cann` uses torch_npu APIs; `torch` denotes an explicit PyTorch NPU composition,
including all of its temporary tensors and layout conversions.

Optional xlite bridges require installed GVirt revision
`4a6dc3102d5778928e4d21116bc4d4694d51e584`:

```bash
cd xllm-kernel
XLLM_KERNEL_BUILD_XLITE=1 MAX_JOBS=4 python setup.py build_ext --inplace
```

The bridge uses PyTorch's current NPU stream and torch_npu's submission queue.
It does not own a runtime, synchronize per call, or compile during preparation.
The build manifest pins Torch/torch_npu, GVirt, and the common/BF16/FP16 launcher
libraries. The package also supports an optional compiled wheel through
`XLLM_KERNEL_BUILD_XLITE=1 python -m pip wheel --no-build-isolation --no-deps ./xllm-kernel`.
These are commands for subsequent validation; they were not executed for this draft.

Prepare each operation before graph capture and retain its callable:

```python
from xllm_kernel.ops.silu_and_mul import prepare

plan = prepare("npu", "aclgraph", "npu.cann.silu_and_mul")
output = plan.function(gate_up)
```

Every implementation name is `npu.<backend>.<operation>`. Selection is explicit;
missing dependencies and unsupported modes fail without an execution fallback.
Each operation owns its signature, domain and mutation contract. `_out` and
in-place operations have distinct contracts and reset their mutated inputs
between validation and timing calls.

The existing package-level `initialize`, `prepare_rms_norm`, and
`XLLM_KERNEL_RMS_NORM_IMPL` remain the host's RMSNorm integration. New operation
modules expose their own `prepare`; they do not change model policy globally.

## Layout and responsibilities

```text
python/xllm_kernel/
  registry.py, operator.py       # preparation and immutable contracts
  ops/<operation>/<backend>.py  # operation-owned implementation loaders
  backends/                     # shared dependency and communicator loading
  fake/<operation>.py           # metadata for package-owned Torch schemas
  numerics/<operation>.py       # inputs, independent CPU reference, restrictions
  numerics/                     # common suite, comparison, reset and report lifecycle
  benchmark/                    # timing, statistics and reports
csrc/xlite/                     # optional current-stream launcher bridges
test/                          # CPU/package tests
  npu/                         # independent device accuracy and graph checks
  distributed/                 # explicit multi-rank tests under torchrun
```

The suite catalog lists modules only. Common runners and reports contain no
operator-name branches or operator-specific shapes, references or tolerances.
Numerical domain helpers are explicitly named; output and alias semantics
remain in each suite. There is no installed `testing/` package.

## Operator and backend coverage

| Operation | NPU implementations |
| --- | --- |
| `add` | `torch`, `xlite` |
| `all_gather` | `xllm_native`, `cann` |
| `all_reduce` | `xllm_native`, `cann` |
| `all_to_all_single` | `cann` |
| `atb_matmul_ein_sum` | `xllm_native`, `torch`, `xlite` |
| `atb_matmul_ein_sum_out` | `xllm_native` |
| `dequant_swiglu_quant` | `xllm_native`, `cann` |
| `dynamic_quant` | `xllm_native`, `cann`, `xlite` |
| `embedding` | `torch`, `xlite` |
| `fused_add_rms_norm` | `xllm_native`, `cann`, `xlite` |
| `fused_add_rms_norm_dynamic_quant` | `cann` |
| `grouped_matmul` | `cann` |
| `grouped_matmul_out` | `xllm_native` |
| `inplace_partial_rotary_mul` | `xllm_native`, `cann` |
| `interleaved_rotary_embedding` | `cann`, `xlite` |
| `layer_norm` | `torch`, `xlite` |
| `lightning_indexer` | `xlite`, `xllm_native`, `torch` |
| `lightning_indexer_out` | `xlite`, `xllm_native` |
| `linear` | `torch`, `xlite` |
| `moe_finalize_routing` | `cann` |
| `moe_gate_routing` | `xlite`, `cann` |
| `moe_gmm1` | `cann`, `xllm_native` |
| `moe_token_dispatch` | `cann`, `xllm_native` |
| `moe_token_unpermute` | `cann` |
| `quant_matmul` | `xllm_native`, `cann` |
| `quant_matmul_out` | `xllm_native` |
| `quantize_per_tensor` | `xllm_native` |
| `reduce_scatter` | `xllm_native`, `cann` |
| `reshape_paged_cache` | `xllm_native`, `torch` |
| `rms_norm` | `xllm_native`, `cann`, `xlite` |
| `rms_norm_dynamic_quant` | `xllm_native`, `cann` |
| `router_matmul` | `torch` |
| `scatter_nd_update` | `xllm_native`, `torch` |
| `sfa_dcp_remap` | `tilelang`, `torch` |
| `silu_and_mul` | `xllm_native`, `cann`, `xlite` |
| `sparse_flash_attention` | `xllm_native`, `torch` |
| `sparse_flash_attention_out` | `xlite`, `xllm_native` |

The inventory is for BF16/FP16 and W8A8 INT8 GLM paths. APIs expose the domains
used by their suites, not every overload of the underlying CANN library.
`all_reduce`, `all_gather`, and `reduce_scatter` offer eager CANN ProcessGroup
and eager/ACLGraph native current-stream implementations. `all_to_all_single`
is eager only. Other declared implementations expose eager and ACLGraph entry
points; actual device support still requires validation.

Key restrictions and differences:

- xlite dynamic quantization is BF16 only; zero-row repair is part of execution.
  Norm, add, activation, embedding, matmul and RoPE have explicit dtype, alignment
  and dimension limits in their bridges, fake implementations and suites.
- xlite router selection supports FP32 logits with 256 experts, top-k <=32,
  one group or eight groups retaining four. Bitmap conversion is timed.
- RMSNorm dynamic quantization uses the native fused operator or CANN's
  add-norm-quant with a zero residual; zero construction is timed. The xlite
  norm followed by BF16 quantization is not registered as the fused FP32 contract.
- MoE dispatch and GMM1 convert V2/V3 group metadata explicitly. Private NZ
  weight conversion happens during case preparation, before measurement.
- Paged indexer/attention suites cover one sequence, decode and multi-token
  causal queries, shuffled block tables, and attention index padding. The
  native profile uses index head 128/top-k 2048 and MLA latent width 512.
  Indexer accuracy checks bounds, causality, uniqueness, and selected CPU
  scores within 1.5%; it does not claim exact index equality at numerical ties.
- TileLang DCP remap uses the existing AOT specialization: <=256 rows,
  top-k 2048. The existing half-layout TileLang RoPE is not registered under
  the interleaved GLM contract.
- xlite's quantized matmul/grouped matmul FP16 intermediate and bitmap routing
  layouts are not substituted for GLM's BF16/INT32/NZ interfaces. Fused MLA
  preparation remains host-owned. xlite indexer adapters own and reset their
  scratch tensors, pack weights and include that work in timing. xlite sparse
  attention is available through `_out` for one-query BF16 decode only: sparse
  index compaction, KV gather and workspace allocation are timed; no LSE is produced.
- Compound MLA preprocessing and grouped MoE schedules remain host-owned and
  are covered here through their component operators. Optional MegaMoE EP,
  FP8 KV-cache attention, quantized index caches, MLAPO v2 fixed shapes, and
  MTP scheduler metadata are outside this standalone BF16/W8A8 operator inventory.
  Ordinary ATen view/split/concat/copy and request scheduling remain in PyTorch/xLLM.

## Independent accuracy and performance tools

Both commands discover the same operation-owned suites. Use
`--operation NAME --help` to inspect dimensions without a device SDK. Inputs
are seeded on CPU; references are independent CPU arithmetic. Validation
checks shapes, dtypes, finite values, read-only inputs, declared mutations and
aliasing. Integer metadata and cache writes are exact; quantization uses
explicit code/scale tolerances. XLite unit tests also compare fake metadata.

```bash
python -m xllm_kernel.numerics --operation silu_and_mul --device npu --mode eager \
  --implementation npu.cann.silu_and_mul --rows 1 16 128 --widths 4096 \
  --revision "$(git rev-parse HEAD)" --output silu-cann-numerics.json
python -m xllm_kernel.benchmark --operation silu_and_mul --device npu --mode aclgraph \
  --implementation npu.xlite.silu_and_mul --rows 1 16 128 --widths 4096 \
  --iterations 100 --repeats 11 --revision "$(git rev-parse HEAD)" --output silu-xlite.json
```

Run each backend in a fresh process on idle devices, with identical cases,
precision and timer settings. Accuracy and graph-output validation must pass
before timing. Reports include exclusions and failures; zero applicable cases
cannot pass. Preparation, warmup and capture are recorded separately. Pure
operators use 64 calls per captured graph; mutable operators use one call and
restore buffers outside the device/host intervals. Wall timing includes resets.
The cache policy is repeated hot inputs. No performance numbers are supplied
by this source-only change.

For multi-rank collectives, initialize explicitly through the CLI:

```bash
HCCL_OP_EXPANSION_MODE=AIV torchrun --nproc-per-node=2 --master-port=29653 \
  -m xllm_kernel.benchmark --distributed --operation all_reduce --device npu \
  --mode aclgraph --implementation npu.xllm_native.all_reduce \
  --iterations 100 --revision "$(git rev-parse HEAD)" --output all-reduce.json
```

Each rank writes its own report (`all-reduce.rank0.json`, etc.). Reports retain
local samples and per-repeat maxima across ranks. Every rank agrees on
validation success before entering timing. HCCL groups are created before
preparation and destroyed after report writing, never inside a captured call.

## Tests

Run package tests independently, without host CMake wiring:

```bash
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu \
  --ignore=xllm-kernel/test/distributed
python -m pytest -q xllm-kernel/test/npu/test_silu_and_mul_backends.py
HCCL_OP_EXPANSION_MODE=AIV torchrun --nproc-per-node=2 --master-port=29654 \
  -m pytest -q xllm-kernel/test/distributed
```

The NPU fixture loads the matching host for integration checks. Optional
providers must be installed to run their cases; missing implementations fail
instead of being silently treated as a successful comparison. The legacy
RMSNorm integration tests still require separate processes and their explicit
`XLLM_KERNEL_RMS_NORM_IMPL` selection. These tests do not establish model-level
correctness or end-to-end performance.

## Development

Add the operation contract and lazy loaders, its independent numerical suite,
fake metadata for new schemas, and tests under the package. Keep implementation
selection outside model execution. Add a suite catalog entry and document
input restrictions and conversions. Validate the actual model integration
separately when changing model call sites.

See the development guide ([English](../docs/src/content/docs/en/dev_guide/xllm_kernel_dev.md) |
[简体中文](../docs/src/content/docs/zh/dev_guide/xllm_kernel_dev.md)).
