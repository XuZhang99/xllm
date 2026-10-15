# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""sparse_flash_attention_out xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    from xllm_kernel.operator import native_operator

    op = native_operator("sparse_flash_attention_lse_out")

    def run(
        query: torch.Tensor,
        cache: torch.Tensor,
        indices: torch.Tensor,
        query_lengths: torch.Tensor,
        key_lengths: torch.Tensor,
        block_table: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        scale: float,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return op(
            query,
            cache,
            cache,
            indices,
            block_table,
            query_lengths,
            key_lengths,
            query_rope,
            key_rope,
            scale,
            1,
            "TND",
            "PA_BSND",
            3,
            9223372036854775807,
            9223372036854775807,
            2,
            False,
            output,
        )

    return run
