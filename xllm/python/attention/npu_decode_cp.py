# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor transformations for NPU decode context parallelism.

The scheduler allocates a logical KV block of ``block_size * dcp_size``
tokens. Each DCP rank stores one contiguous ``block_size``-token sub-block,
while the Lightning Indexer cache is replicated in global token order.
"""

from __future__ import annotations

import torch


def _validate_layout(block_size: int, dcp_size: int, dcp_rank: int) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if dcp_size <= 1:
        raise ValueError(f"dcp_size must be greater than one, got {dcp_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank must be in [0, {dcp_size}), got {dcp_rank}")


def localize_cache_slots(
    logical_slots: torch.Tensor,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Map global logical slots to this rank's physical cache slots."""
    _validate_layout(block_size, dcp_size, dcp_rank)
    safe_slots = logical_slots.clamp_min(0)
    logical_block_size = block_size * dcp_size
    logical_offsets = torch.remainder(safe_slots, logical_block_size)
    owner_ranks = torch.div(logical_offsets, block_size, rounding_mode="floor")
    local_slots = (
        torch.div(safe_slots, logical_block_size, rounding_mode="floor") * block_size
        + torch.remainder(logical_offsets, block_size)
    )
    owned = (logical_slots >= 0) & (owner_ranks == dcp_rank)
    return torch.where(owned, local_slots, torch.full_like(local_slots, -1))


def expand_indexer_block_table(
    logical_block_table: torch.Tensor,
    dcp_size: int,
) -> torch.Tensor:
    """Expand each logical cache block into every replicated indexer block."""
    if logical_block_table.dim() != 2:
        raise ValueError("logical_block_table must be two-dimensional")
    if dcp_size <= 1:
        raise ValueError(f"dcp_size must be greater than one, got {dcp_size}")
    shard_offsets = torch.arange(
        dcp_size,
        dtype=logical_block_table.dtype,
        device=logical_block_table.device,
    )
    expanded = logical_block_table.unsqueeze(-1) * dcp_size + shard_offsets
    expanded = torch.where(
        logical_block_table.unsqueeze(-1) >= 0,
        expanded,
        torch.full_like(expanded, -1),
    )
    return expanded.flatten(start_dim=1).contiguous()


def local_sequence_lengths(
    global_lengths: torch.Tensor,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Return compact per-request KV lengths stored by one DCP rank."""
    _validate_layout(block_size, dcp_size, dcp_rank)
    logical_block_size = block_size * dcp_size
    safe_lengths = global_lengths.clamp_min(0)
    complete_blocks = torch.div(safe_lengths, logical_block_size, rounding_mode="floor")
    remainder = torch.remainder(safe_lengths, logical_block_size)
    local_remainder = torch.clamp(remainder - dcp_rank * block_size, min=0, max=block_size)
    return complete_blocks * block_size + local_remainder


def remap_sparse_indices(
    global_indices: torch.Tensor,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Map global top-k indices to local positions and pack valid ones first."""
    _validate_layout(block_size, dcp_size, dcp_rank)
    topk_count = global_indices.shape[-1]
    # Float32 division is materially faster than integer floor-divide on
    # Ascend. GLM-5.2 positions are far below float32's exact integer range.
    indices_fp32 = global_indices.to(torch.float32)
    safe_indices = indices_fp32.clamp_min(0)
    global_sub_blocks = torch.floor(safe_indices / block_size)
    owner_ranks = torch.remainder(global_sub_blocks, dcp_size)
    owned = (indices_fp32 >= 0) & (owner_ranks == dcp_rank)
    local_indices = (
        torch.floor(safe_indices / (block_size * dcp_size)) * block_size
        + torch.remainder(safe_indices, block_size)
    )
    local_indices = torch.where(owned, local_indices, torch.full_like(local_indices, -1)).to(
        global_indices.dtype
    )

    original_order = torch.arange(
        topk_count,
        dtype=torch.float32,
        device=global_indices.device,
    ).expand_as(global_indices)
    pack_keys = original_order + (~owned).to(torch.float32) * topk_count
    _, pack_order = torch.sort(pack_keys, dim=-1)
    if global_indices.device.type == "npu":
        pack_order = pack_order.to(torch.int32)
    return torch.gather(local_indices, dim=-1, index=pack_order).contiguous()


def local_attention_scale(
    local_lse: torch.Tensor,
    gathered_lse: torch.Tensor,
) -> torch.Tensor:
    """Compute this rank's stable softmax correction from all ranks' LSEs."""
    if gathered_lse.dim() != local_lse.dim() + 1:
        raise ValueError("gathered_lse must have a leading DCP-rank dimension")
    if tuple(gathered_lse.shape[1:]) != tuple(local_lse.shape):
        raise ValueError("gathered_lse trailing dimensions must match local_lse")
    finite_lse = gathered_lse.masked_fill(~torch.isfinite(gathered_lse), float("-inf"))
    global_lse = torch.logsumexp(finite_lse, dim=0)
    scale = torch.exp(local_lse - global_lse)
    return torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)


def sparse_attention_lse(
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    num_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """Convert Ascend TND SFA max/sum outputs to natural-log ``[T, H]``."""
    if softmax_max.shape != softmax_sum.shape or softmax_max.dim() != 3:
        raise ValueError("NPU sparse attention max/sum tensors must have matching three-dimensional shapes")
    lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
    lse = lse.permute(1, 0, 2).reshape(num_tokens, -1)
    if tuple(lse.shape) != (num_tokens, num_heads):
        raise ValueError(
            "NPU sparse attention LSE shape does not match its query: "
            f"got {tuple(lse.shape)}, expected {(num_tokens, num_heads)}"
        )
    return lse


__all__ = [
    "expand_indexer_block_table",
    "local_attention_scale",
    "local_sequence_lengths",
    "localize_cache_slots",
    "remap_sparse_indices",
    "sparse_attention_lse",
]
