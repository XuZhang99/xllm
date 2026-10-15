# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned fused_add_rms_norm_dynamic_quant contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("fused_add_rms_norm_dynamic_quant", 1, (), False, "fp32_add_norm_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "cann",
            "xllm_kernel.ops.fused_add_rms_norm_dynamic_quant.cann:load",
            "Functional residual add + RMSNorm + dynamic INT8; returns codes, scale, residual sum; inputs unchanged.",
        ),
    ),
)
prepare = OPERATOR.prepare
