# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functional RMSNorm using a binding selected during package initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.registry import OpContract, PreparedKernel

if TYPE_CHECKING:
    import torch

RMS_NORM_CONTRACT = OpContract(
    operation="rms_norm",
    version=1,
    mutates=(),
    aliases_inputs=False,
    numerical_mode="xllm_native",
)
OPERATION = Operation(
    contract=RMS_NORM_CONTRACT,
    backends=(
        BackendModule("xllm_kernel.ops.rms_norm.npu", "npu", ("torch_npu",)),
        BackendModule("xllm_kernel.ops.rms_norm.xlite", "npu", ("torch_npu", "xlite", "xllm_kernel._xlite")),
    ),
)


def prepare_rms_norm() -> PreparedKernel:
    """Return the existing binding; this never searches for an implementation."""
    return OPERATION.get_prepared()


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the last dimension without modifying or aliasing the inputs.

    The default binding preserves torch_npu RMSNorm's complete tensor domain.
    Explicit xlite selection restricts inputs to the domain in its KernelSpec;
    its bridge validates tensors before launch and never falls back. Both have
    no residual, gamma offset, quantization, or out-buffer semantics. Numerical
    equivalence between implementations is tolerance-based, not bitwise.
    """
    prepared = OPERATION.prepared
    if prepared is None:
        raise RuntimeError("xllm_kernel runtime is not initialized")
    return prepared.function(input, weight, eps)
