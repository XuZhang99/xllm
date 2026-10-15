# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned router_matmul contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("router_matmul", 1, (), False, "bf16_matmul_fp32_output")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "torch",
            "xllm_kernel.ops.router_matmul.torch:load",
            "BF16 input and weights, FP32 router logits; preserves torch.addmm out dtype.",
        ),
    ),
)
prepare = OPERATOR.prepare
