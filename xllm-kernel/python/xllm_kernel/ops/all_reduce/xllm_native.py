# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""all_reduce xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    from xllm_kernel.backends.collective import prepare_hccl
    from xllm_kernel.operator import native_operator

    op = native_operator("npu_all_reduce")
    group, comm = prepare_hccl()

    def run(value: torch.Tensor) -> torch.Tensor:
        op(value, comm)
        return value

    return run
