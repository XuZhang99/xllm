# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned scatter_nd_update contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("scatter_nd_update", 1, ("value",), True, "exact_cache_write")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.scatter_nd_update.xllm_native:load",
            "Unique valid block/offset indices[T,2] write cache[B,S,1,D]; in-place with all untouched slots preserved.",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.scatter_nd_update.torch:load",
            "Unique valid block/offset indices[T,2] write cache[B,S,1,D]; in-place with all untouched slots preserved.",
        ),
    ),
)
prepare = OPERATOR.prepare
