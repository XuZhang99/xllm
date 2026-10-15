# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned quantize_per_tensor contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("quantize_per_tensor", 1, (), False, "static_saturated_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.quantize_per_tensor.xllm_native:load",
            "Static INT8 quantization with positive FP32 scalar scale and INT32 zero point; qint8 passed to native schema; saturated output.",
        ),
    ),
)
prepare = OPERATOR.prepare
