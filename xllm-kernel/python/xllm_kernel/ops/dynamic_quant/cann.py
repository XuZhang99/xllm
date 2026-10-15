# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""dynamic_quant cann provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch_npu

    def run(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_npu.npu_dynamic_quant(value)

    return run
