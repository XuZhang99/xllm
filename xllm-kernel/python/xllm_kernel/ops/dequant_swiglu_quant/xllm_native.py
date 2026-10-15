# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""dequant_swiglu_quant xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    from xllm_kernel.operator import native_operator

    op = native_operator("dequant_swiglu_quant")

    def run(
        value: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        swiglu_mode: int = 0,
        clamp_limit: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return op(
            value, weight_scale, activation_scale, None, None, None, None, True, 1, swiglu_mode, clamp_limit, 1.0, 0.0
        )

    return run
