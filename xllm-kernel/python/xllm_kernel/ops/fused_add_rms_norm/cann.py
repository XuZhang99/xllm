# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""fused_add_rms_norm cann provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch_npu

    def run(
        value: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, _, summed = torch_npu.npu_add_rms_norm(value, residual, weight, eps)
        return output, summed

    return run
