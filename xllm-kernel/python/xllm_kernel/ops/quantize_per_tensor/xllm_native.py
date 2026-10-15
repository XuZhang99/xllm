# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""quantize_per_tensor xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    from xllm_kernel.operator import native_operator

    op = native_operator("quantize_per_tensor")

    def run(value: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        return op(value, scale, zero_point, torch.qint8, -1)

    return run
