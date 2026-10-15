# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""inplace_partial_rotary_mul cann provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch_npu

    def run(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, start: int, end: int) -> None:
        value[..., start:end].copy_(
            torch_npu.npu_interleave_rope(value[..., start:end].unsqueeze(2).contiguous(), cosine, sine).squeeze(2)
        )

    return run
