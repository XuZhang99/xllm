# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""all_reduce cann provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch.distributed as dist

    from xllm_kernel.backends.collective import prepare_hccl

    group, _ = prepare_hccl()

    def run(value: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(value, group=group)
        return value

    return run
