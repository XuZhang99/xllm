# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned grouped_matmul contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("grouped_matmul", 1, (), False, "int32_accumulate_fp32_dequant_bf16")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "cann",
            "xllm_kernel.ops.grouped_matmul.cann:load",
            "INT8 grouped W8A8 down projection; private NZ weights[E,K,N], cumulative INT64 group offsets; BF16 output.",
        ),
    ),
)
prepare = OPERATOR.prepare
