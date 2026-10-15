# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned dequant_swiglu_quant contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("dequant_swiglu_quant", 1, (), False, "fp32_swiglu_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.dequant_swiglu_quant.xllm_native:load",
            "INT32[T,2H] with FP32 dequantization scales; explicit standard or interleaved SwiGLU; dynamic INT8 output.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.dequant_swiglu_quant.cann:load",
            "INT32[T,2H] with FP32 dequantization scales; explicit standard or interleaved SwiGLU; dynamic INT8 output.",
        ),
    ),
)
prepare = OPERATOR.prepare
