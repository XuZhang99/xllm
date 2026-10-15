# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned moe_gmm1 contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("moe_gmm1", 1, (), False, "int32_accumulate_fp32_swiglu_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "cann",
            "xllm_kernel.ops.moe_gmm1.cann:load",
            "Grouped INT8 gate/up + standard SwiGLU + per-token INT8 quant; private NZ weights and FP32 scales; metadata conversion timed.",
        ),
        Backend(
            "xllm_native",
            "xllm_kernel.ops.moe_gmm1.xllm_native:load",
            "Grouped INT8 gate/up + standard SwiGLU + per-token INT8 quant; private NZ weights and FP32 scales; metadata conversion timed.",
        ),
    ),
)
prepare = OPERATOR.prepare
