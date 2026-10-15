# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_token_dispatch xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch

    from xllm_kernel.operator import native_operator

    op = native_operator("moe_init_routing_v3")

    def run(
        value: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, idx, pairs, scale = op(value, expert_ids, expert_ids.numel(), num_experts, [0, num_experts], 1)
        counts = torch.zeros(num_experts, dtype=torch.int64, device=value.device)
        counts.scatter_add_(0, pairs[:, 0], pairs[:, 1])
        return q, idx, counts.cumsum(0), scale.reshape(-1)

    return run
