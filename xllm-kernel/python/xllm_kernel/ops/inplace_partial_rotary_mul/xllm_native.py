# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""inplace_partial_rotary_mul xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    from xllm_kernel.operator import native_operator

    op = native_operator("inplace_partial_rotary_mul")

    def run(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, start: int, end: int) -> None:
        op(value.unsqueeze(2), cosine, sine, "interleave", [start, end])

    return run
