# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""reduce_scatter cann provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch.distributed as dist

    from xllm_kernel.backends.collective import prepare_hccl

    group, _ = prepare_hccl()

    def run(value: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        dist.reduce_scatter_tensor(output.view(-1), value.view(-1), group=group)
        return output

    return run
