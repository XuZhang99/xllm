# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned grouped_matmul_out contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("grouped_matmul_out", 1, ("output",), True, "int32_accumulate_fp32_dequant_bf16")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.grouped_matmul_out.xllm_native:load",
            "Caller-owned BF16 output of grouped INT8 down projection; private NZ weights and cumulative group list.",
        ),
    ),
)
prepare = OPERATOR.prepare
