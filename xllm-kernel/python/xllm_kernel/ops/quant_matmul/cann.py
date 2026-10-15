# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""quant_matmul cann provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch_npu

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch_npu.npu_quant_matmul(
            value, weight.T, weight_scale, pertoken_scale=activation_scale, output_dtype=output_dtype
        )

    return run
