# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned rms_norm_dynamic_quant contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("rms_norm_dynamic_quant", 1, (), False, "fp32_norm_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.rms_norm_dynamic_quant.xllm_native:load",
            "FP16/BF16 [T,D] plus gamma[D] -> (INT8[T,D], FP32[T]); fused FP32 norm before quantization.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.rms_norm_dynamic_quant.cann:load",
            "FP16/BF16 [T,D] plus gamma[D] -> (INT8[T,D], FP32[T]); fused FP32 norm before quantization.",
        ),
    ),
)
prepare = OPERATOR.prepare
