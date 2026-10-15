# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""dequant_swiglu_quant cann provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch_npu

    def run(
        value: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        swiglu_mode: int = 0,
        clamp_limit: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_npu.npu_dequant_swiglu_quant(
            value,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=swiglu_mode,
            clamp_limit=clamp_limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )

    return run
