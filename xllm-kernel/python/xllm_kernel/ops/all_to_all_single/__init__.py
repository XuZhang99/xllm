# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned all_to_all_single contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("all_to_all_single", 1, ("output",), True, "exact_equal_split_exchange")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "cann",
            "xllm_kernel.ops.all_to_all_single.cann:load",
            "Equal split rank-ordered HCCL all-to-all on initialized default group; contiguous ND input/output[world,rows,width]; eager only.",
            execution_modes=("eager",),
        ),
    ),
)
prepare = OPERATOR.prepare
