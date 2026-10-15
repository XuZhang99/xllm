# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned moe_token_unpermute contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("moe_token_unpermute", 1, (), False, "fp32_weighted_sum_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "cann",
            "xllm_kernel.ops.moe_token_unpermute.cann:load",
            "Full-load gather row indices[T*K], FP32 probabilities[T,K], routed tensor[T*K,D] -> weighted sum[T,D].",
        ),
    ),
)
prepare = OPERATOR.prepare
