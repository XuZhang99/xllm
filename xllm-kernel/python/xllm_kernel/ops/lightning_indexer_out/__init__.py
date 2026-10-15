# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned lightning_indexer_out contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("lightning_indexer_out", 1, ("output",), True, "relu_weighted_score_causal_topk")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xlite",
            "xllm_kernel.ops.lightning_indexer_out.xlite:load",
            "Single sequence, H=32,D=128,block=128,K=2048,T<=128; weight packing and workspace initialization timed",
        ),
        Backend(
            "xllm_native",
            "xllm_kernel.ops.lightning_indexer_out.xllm_native:load",
            "Caller-owned INT32 index output; auxiliary unrequested score buffer allocated internally and included in timing.",
        ),
    ),
)
prepare = OPERATOR.prepare
