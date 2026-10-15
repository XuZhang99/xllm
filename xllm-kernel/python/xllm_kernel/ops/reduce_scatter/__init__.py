# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned reduce_scatter contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("reduce_scatter", 1, ("output",), True, "same_dtype_sum_scatter")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.reduce_scatter.xllm_native:load",
            "Explicit initialized default HCCL group; contiguous ND tensors; SUM/rank-order semantics; communicator acquired before graph capture.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.reduce_scatter.cann:load",
            "Explicit initialized default HCCL group; contiguous ND tensors; SUM/rank-order semantics; communicator acquired before graph capture.",
            execution_modes=("eager",),
        ),
    ),
)
prepare = OPERATOR.prepare
