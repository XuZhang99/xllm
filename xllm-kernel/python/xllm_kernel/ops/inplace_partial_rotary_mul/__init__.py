# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned inplace_partial_rotary_mul contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("inplace_partial_rotary_mul", 1, ("value",), True, "fp32_interleaved_rope_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.inplace_partial_rotary_mul.xllm_native:load",
            "In-place half-open [start,end) interleaved rotation on [T,H,D]; untouched tail must remain exact; CANN gather/rotate/copy timed.",
        ),
        Backend(
            "cann",
            "xllm_kernel.ops.inplace_partial_rotary_mul.cann:load",
            "In-place half-open [start,end) interleaved rotation on [T,H,D]; untouched tail must remain exact; CANN gather/rotate/copy timed.",
        ),
    ),
)
prepare = OPERATOR.prepare
