# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned linear contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("linear", 1, (), False, "fp32_matmul_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "torch",
            "xllm_kernel.ops.linear.torch:load",
            "NPU FP16/BF16 [tokens,K] times [N,K]; no bias; xlite requires K and N divisible by 16.",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.linear.xlite:load",
            "NPU FP16/BF16 [tokens,K] times [N,K]; no bias; xlite requires K and N divisible by 16.",
        ),
    ),
)
prepare = OPERATOR.prepare
