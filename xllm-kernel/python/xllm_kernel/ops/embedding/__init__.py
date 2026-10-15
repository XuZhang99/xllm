# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned embedding contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("embedding", 1, (), False, "exact_gather")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "torch",
            "xllm_kernel.ops.embedding.torch:load",
            "NPU INT32 valid IDs [0,vocab), BF16/FP16 [vocab,width] weights; xlite width divisible by 64 and <=8192.",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.embedding.xlite:load",
            "NPU INT32 valid IDs [0,vocab), BF16/FP16 [vocab,width] weights; xlite width divisible by 64 and <=8192.",
        ),
    ),
)
prepare = OPERATOR.prepare
