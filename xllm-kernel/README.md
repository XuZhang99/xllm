<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# xLLM Kernel

Initial NPU RMSNorm integration for Python-defined models, including
GLM-5.3 (`glm_moe_dsa`). The native implementation, numerical behavior,
Torch schema, fake registration, and C++ callers retain their existing owners.

Install from a source checkout before starting Python model workers:

```bash
python -m pip install --no-deps -e ./xllm-kernel
python setup.py build
```

The main xLLM build also includes `xllm_kernel` in its wheel. The standalone
package can be imported without Torch or an NPU SDK; execution requires a
matching xLLM host, PyTorch, torch_npu, and CANN. Importing the package does
not load the native extension or initialize a device.

The existing host startup calls `xllm_kernel.initialize(device="npu")` after
native operators are loaded. It freezes a preparation-only registry and binds
`npu.xllm_native.rms_norm` before model import/capture. Repeating the same
initialization is safe; changing device or policy requires a worker restart.
A missing native operator, unsupported device, or unknown explicit
implementation fails during preparation. There is no execution-error fallback.

```python
from xllm_kernel import prepare_rms_norm
from xllm_kernel.ops import rms_norm

# Inside an initialized xLLM host:
plan = prepare_rms_norm()
output = rms_norm(input, weight, eps)
# A model may retain plan.function directly for its graph's entire lifetime.
```

`xllm.python.kernels.rms_norm` and the NPU normalization shim delegate to
this API. This covers the shared RMSNorm layers and MLA preprocessing used
by GLM-5.3. Residual/quantized/Gemma RMSNorm and other operator families
remain on their existing paths.

This first adapter uses the complete native tensor domain; it does not create
shape-specific plans or perform per-token selection. Native Torch dispatch
validates arguments. Output is independent of input/weight, and neither input
is modified. Registry diagnostics expose the selected contract, implementation,
execution modes, generation, and exclusion reasons. Runtime logs selection once.

Validation lives in `tests/scripts/test_kernel_package.py` (import/registry/
packaging) and `tests/python/test_kernel_rms_norm_npu.py` (native/fake/stream/
ACLGraph and GLM dimensions). Run NPU tests only on an idle device. A package
import or a passing fake test does not prove device or whole-model correctness.

CUDA adapters, shape-specialized selection, independent native libraries,
fused/quantized operators, and compiler ownership migration are later phases.
The existing CUDA-like platform paths remain unchanged.
