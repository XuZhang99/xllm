# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned all_gather contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("all_gather", 1, ("output",), True, "exact_rank_order_gather")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.all_gather.xllm_native:load",
            "Explicit initialized default HCCL group; contiguous ND tensors; SUM/rank-order semantics; communicator acquired before graph capture.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.all_gather.cann:load",
            "Explicit initialized default HCCL group; contiguous ND tensors; SUM/rank-order semantics; communicator acquired before graph capture.",
            execution_modes=("eager",),
        ),
    ),
)
prepare = OPERATOR.prepare
