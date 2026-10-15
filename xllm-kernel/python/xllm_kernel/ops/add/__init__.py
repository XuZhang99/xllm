# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned add contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("add", 1, (), False, "same_dtype_add")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "torch",
            "xllm_kernel.ops.add.torch:load",
            "Contiguous NPU tensors; see operation module for the exact signature.",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.add.xlite:load",
            "Contiguous NPU tensors; see operation module for the exact signature.",
        ),
    ),
)
prepare = OPERATOR.prepare
