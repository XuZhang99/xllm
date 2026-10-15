# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned silu_and_mul contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("silu_and_mul", 1, (), False, "fp32_swiglu_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.silu_and_mul.xllm_native:load",
            "Contiguous NPU tensors; see operation module for the exact signature.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.silu_and_mul.cann:load",
            "Contiguous NPU tensors; see operation module for the exact signature.",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.silu_and_mul.xlite:load",
            "Contiguous NPU tensors; see operation module for the exact signature.",
        ),
    ),
)
prepare = OPERATOR.prepare
