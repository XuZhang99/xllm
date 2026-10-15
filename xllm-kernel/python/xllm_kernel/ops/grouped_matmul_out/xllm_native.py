# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""grouped_matmul_out xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    from xllm_kernel.operator import native_operator

    op = native_operator("grouped_matmul_out")

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        groups: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return op(value, weight, weight_scale, activation_scale, groups, 2, 0, 0, output)

    return run
