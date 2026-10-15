# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned atb_matmul_ein_sum contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("atb_matmul_ein_sum", 1, (), False, "fp32_matmul_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.atb_matmul_ein_sum.xllm_native:load",
            "BF16 [T,H,D] x [H,D,O] -> [T,H,O]; xlite D/O multiples of 16.",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.atb_matmul_ein_sum.torch:load",
            "BF16 [T,H,D] x [H,D,O] -> [T,H,O]; xlite D/O multiples of 16.",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.atb_matmul_ein_sum.xlite:load",
            "BF16 [T,H,D] x [H,D,O] -> [T,H,O]; xlite D/O multiples of 16.",
        ),
    ),
)
prepare = OPERATOR.prepare
