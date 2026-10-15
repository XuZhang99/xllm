# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned sparse_flash_attention_out contract and explicit backend preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("sparse_flash_attention_out", 1, ("output",), True, "fp32_sparse_softmax_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xlite",
            "xllm_kernel.ops.sparse_flash_attention_out.xlite:load",
            "BF16 single-query decode, H<=32,D=512,R=64,K=2048,block=128; compaction, gather and workspace timed",
        ),
        Backend(
            "xllm_native",
            "xllm_kernel.ops.sparse_flash_attention_out.xllm_native:load",
            "Paged causal MLA into caller-owned output[T,H,512]; no LSE output; separate mutation contract.",
        ),
    ),
)
prepare = OPERATOR.prepare
