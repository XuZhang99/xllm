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

"""Functional RMSNorm, preserving the host native Torch operator's contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.registry import KernelRegistry, OpContract, PreparedKernel

if TYPE_CHECKING:
    import torch

RMS_NORM_CONTRACT = OpContract(
    operation="rms_norm",
    version=1,
    mutates=(),
    aliases_inputs=False,
    numerical_mode="xllm_native",
)
_prepared: PreparedKernel | None = None
_configuration: tuple[str, str | None] | None = None


def initialize(*, device: str, implementation: str | None = None) -> PreparedKernel:
    """Bind the NPU host operator once, before model import or graph capture.

    The host must have registered ``torch.ops.xllm_ops`` first. It remains the
    owner of native schemas and fake implementations. No dependency discovery,
    device probing, compilation, or execution-error fallback happens here.
    Changing policy requires a fresh worker and fresh graphs.
    """
    global _prepared, _configuration
    configuration = (device, implementation)
    if _prepared is not None:
        if configuration != _configuration:
            raise RuntimeError("xllm_kernel is already initialized; restart the worker to change kernel policy")
        return _prepared
    if device != "npu":
        raise ValueError(f"xllm_kernel does not yet provide a {device!r} adapter")

    from xllm_kernel.ops.normalization.npu import register

    registry = KernelRegistry()
    register(registry)
    registry.freeze()
    prepared = registry.prepare(
        RMS_NORM_CONTRACT,
        device=device,
        execution_mode="aclgraph",
        implementation=implementation,
    )
    _configuration = configuration
    _prepared = prepared
    return prepared


def prepare_rms_norm() -> PreparedKernel:
    """Return the existing binding; this never searches for an implementation."""
    if _prepared is None:
        raise RuntimeError("call xllm_kernel.initialize(device='npu') after loading the host native Torch ops")
    return _prepared


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the last dimension without modifying or aliasing the inputs.

    This thin adapter preserves ``torch.ops.xllm_ops.rms_norm`` argument
    validation, strided views, dtype rules, allocation, and rounding. It has no
    residual, gamma offset, quantization, or out-buffer semantics. The existing
    host fake registration describes its abstract output; no schema is added.
    """
    if _prepared is None:
        raise RuntimeError("xllm_kernel runtime is not initialized")
    return _prepared.function(input, weight, eps)
