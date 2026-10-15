# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned moe_token_dispatch contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("moe_token_dispatch", 1, (), False, "symmetric_int8_gather_cumulative_groups")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "cann",
            "xllm_kernel.ops.moe_token_dispatch.cann:load",
            "Full-load W8A8 dispatch; cumulative group offsets and gather row indices; native V3 metadata conversion is included.",
        ),
        Backend(
            "xllm_native",
            "xllm_kernel.ops.moe_token_dispatch.xllm_native:load",
            "Full-load W8A8 dispatch; cumulative group offsets and gather row indices; native V3 metadata conversion is included.",
        ),
    ),
)
prepare = OPERATOR.prepare
