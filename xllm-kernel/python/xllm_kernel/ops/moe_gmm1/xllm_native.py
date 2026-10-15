# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_gmm1 xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    from xllm_kernel.operator import native_operator

    op = native_operator("moe_grouped_matmul_swiglu_quant")

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        groups: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.cat((groups[:1], groups[1:] - groups[:-1]))
        pairs = torch.stack((torch.arange(groups.numel(), device=groups.device, dtype=torch.int64), counts), -1)
        return op(value, weight, weight_scale, activation_scale, pairs)

    return run
