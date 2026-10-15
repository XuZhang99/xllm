# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned reshape_paged_cache contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("reshape_paged_cache", 1, ("key_cache", "value_cache"), True, "exact_cache_write")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.reshape_paged_cache.xllm_native:load",
            "INT32 unique nonnegative slots[T]; key/value[T,1,D] -> paged cache[B,S,1,D]; updates both caches; no padding slots.",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.reshape_paged_cache.torch:load",
            "INT32 unique nonnegative slots[T]; key/value[T,1,D] -> paged cache[B,S,1,D]; updates both caches; no padding slots.",
        ),
    ),
)
prepare = OPERATOR.prepare
