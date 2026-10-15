# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_token_dispatch cann provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    import torch_npu

    def run(
        value: torch.Tensor, expert_ids: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, idx, groups, scale = torch_npu.npu_moe_init_routing_v2(
            value,
            expert_ids,
            active_num=expert_ids.numel(),
            expert_num=num_experts,
            expert_tokens_num_type=0,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=1,
        )
        return q, idx, groups, scale.reshape(-1)

    return run
