# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned fused_add_rms_norm contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("fused_add_rms_norm", 1, (), False, "fp32_residual_norm_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.fused_add_rms_norm.xllm_native:load",
            "Functional add + RMSNorm returning (normalized,sum); neither input is mutated. xlite copies residual into its returned sum buffer.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.fused_add_rms_norm.cann:load",
            "Functional add + RMSNorm returning (normalized,sum); neither input is mutated. xlite copies residual into its returned sum buffer.",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.fused_add_rms_norm.xlite:load",
            "Functional add + RMSNorm returning (normalized,sum); neither input is mutated. xlite copies residual into its returned sum buffer.",
        ),
    ),
)
prepare = OPERATOR.prepare
