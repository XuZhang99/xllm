# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned quant_matmul contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("quant_matmul", 1, (), False, "int32_accumulate_or_fp32_dequant_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.quant_matmul.xllm_native:load",
            "INT8 value[T,K], weight[N,K], FP32 channel[N]/token[T] scales -> BF16/FP16[T,N]; no bias or offset.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.quant_matmul.cann:load",
            "INT8 value[T,K], weight[N,K], FP32 channel[N]/token[T] scales -> BF16/FP16[T,N]; no bias or offset.",
        ),
    ),
)
prepare = OPERATOR.prepare
