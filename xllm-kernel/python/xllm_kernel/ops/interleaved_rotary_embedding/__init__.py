# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned interleaved_rotary_embedding contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("interleaved_rotary_embedding", 1, (), False, "fp32_interleaved_rope_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "cann",
            "xllm_kernel.ops.interleaved_rotary_embedding.cann:load",
            "value[T,H,D], cosine/sine[T,1,1,D] with duplicated even/odd frequencies; functional interleaved RoPE; xlite table packing timed.",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.interleaved_rotary_embedding.xlite:load",
            "value[T,H,D], cosine/sine[T,1,1,D] with duplicated even/odd frequencies; functional interleaved RoPE; xlite table packing timed.",
        ),
    ),
)
prepare = OPERATOR.prepare
