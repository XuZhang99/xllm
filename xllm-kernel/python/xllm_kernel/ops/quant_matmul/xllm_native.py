# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""quant_matmul xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    from xllm_kernel.operator import native_operator

    op = native_operator("quant_matmul")

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return op(value, weight, True, weight_scale, None, activation_scale, None, output_dtype)

    return run
