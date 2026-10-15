# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned dynamic_quant contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("dynamic_quant", 1, (), False, "per_token_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.dynamic_quant.xllm_native:load",
            "value[T,D] -> (int8[T,D], float32[T]); finite per-token symmetric quantization; no smoothing/grouping.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.dynamic_quant.cann:load",
            "value[T,D] -> (int8[T,D], float32[T]); finite per-token symmetric quantization; no smoothing/grouping.",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.dynamic_quant.xlite:load",
            "value[T,D] -> (int8[T,D], float32[T]); finite per-token symmetric quantization; no smoothing/grouping.",
        ),
    ),
)
prepare = OPERATOR.prepare
