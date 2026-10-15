# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned all_reduce contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("all_reduce", 1, ("value",), True, "same_dtype_sum")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.all_reduce.xllm_native:load",
            "Explicit initialized default HCCL group; contiguous ND tensors; SUM/rank-order semantics; communicator acquired before graph capture.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.all_reduce.cann:load",
            "Explicit initialized default HCCL group; contiguous ND tensors; SUM/rank-order semantics; communicator acquired before graph capture.",
            execution_modes=("eager",),
        ),
    ),
)
prepare = OPERATOR.prepare
