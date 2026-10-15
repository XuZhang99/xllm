# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned sparse_flash_attention contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("sparse_flash_attention", 1, (), False, "fp32_sparse_softmax_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xllm_native",
            "xllm_kernel.ops.sparse_flash_attention.xllm_native:load",
            "Single-sequence causal MLA with paged latent KV and interleaved RoPE; sparse token indices (-1 padding); output plus FP32 softmax max/sum.",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.sparse_flash_attention.torch:load",
            "Single-sequence causal MLA with paged latent KV and interleaved RoPE; sparse token indices (-1 padding); output plus FP32 softmax max/sum.",
        ),
    ),
)
prepare = OPERATOR.prepare
