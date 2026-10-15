# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned atb_matmul_ein_sum_out contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("atb_matmul_ein_sum_out", 1, ("output",), True, "fp32_matmul_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.atb_matmul_ein_sum_out.xllm_native:load",
            "BF16 [T,H,D] x [H,D,O] into caller-owned output[T,H,O]; stable output buffer for graph replay.",
        ),
    ),
)
prepare = OPERATOR.prepare
