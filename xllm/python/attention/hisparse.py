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

"""Bounded HBM working set for Host-backed GLM sparse MLA caches.

The C++ cache allocator supplies NPU-addressable Host tensors. Logical paging
and full Index Cache remain unchanged. Cache tags validate stale reverse-map
entries when a hot slot is reused, including reordered requests and graph buckets.
"""

from __future__ import annotations

import torch

from xllm.python.attention.backend import LayerCache


def physical_topk_slots(
    topk: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Translate logical token positions without host reads or dynamic shapes."""
    logical = topk.reshape(topk.shape[0], -1).to(torch.int64)
    valid = (logical >= 0) & (logical < seq_lens.reshape(-1, 1))
    valid = valid & (slot_mapping.reshape(-1, 1) >= 0)
    page = (logical.clamp_min(0) // page_size).clamp_max(block_table.shape[1] - 1)
    physical_page = block_table.gather(1, page).to(torch.int64)
    physical = physical_page * page_size + logical.remainder(page_size)
    return torch.where(valid & (physical_page >= 0), physical, -1).to(torch.int32).reshape(-1)


class HiSparseWorkspace:
    """One selected-KV buffer shared by sequential layers and decode graphs."""

    def __init__(self, max_selected_tokens: int, device: torch.device) -> None:
        if not 0 < max_selected_tokens <= 1048576:
            raise ValueError("HiSparse workspace requires 1 to 1048576 selected tokens")
        self.key = torch.empty((max_selected_tokens, 512), dtype=torch.bfloat16, device=device)
        self.rope = torch.empty((max_selected_tokens, 64), dtype=torch.bfloat16, device=device)


class HiSparseCache:
    """Per-layer physical-slot cache; only unique decode Top-K rows are refilled."""

    def __init__(self, cache: LayerCache, hot_tokens: int, workspace: HiSparseWorkspace) -> None:
        if cache.key is None or cache.value is None:
            raise ValueError("HiSparse needs MLA latent and RoPE caches")
        self._host_key = cache.key.view(-1, cache.key.shape[-1])
        self._host_rope = cache.value.view(-1, cache.value.shape[-1])
        if (
            self._host_key.dtype != torch.bfloat16
            or self._host_rope.dtype != torch.bfloat16
            or self._host_key.shape[1] != 512
            or self._host_rope.shape[1] != 64
        ):
            raise ValueError("HiSparse supports BF16 MLA latent=512, RoPE=64")
        if hot_tokens <= 0 or not 0 < self._host_key.shape[0] <= 1048576:
            raise ValueError("HiSparse requires positive hot capacity and at most 1048576 logical slots")
        if self._host_key.shape[0] != self._host_rope.shape[0]:
            raise ValueError("HiSparse latent and RoPE logical capacities must match")
        self._workspace = workspace
        self._hot_tokens = min(hot_tokens, self._host_key.shape[0])
        self._hot_key = self._host_key.new_empty((self._hot_tokens, 512))
        self._hot_rope = self._host_rope.new_empty((self._hot_tokens, 64))
        self._tags = torch.full((self._hot_tokens,), -1, dtype=torch.int32, device=cache.key.device)
        # The final entry absorbs graph padding updates. Its value is never read.
        self._slot_map = torch.full((self._host_key.shape[0] + 1,), -1, dtype=torch.int32, device=cache.key.device)

    def store(self, key: torch.Tensor, rope: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        torch.ops.xllm_ops.hisparse_store(key.reshape(-1, 512).contiguous(), slot_mapping, self._host_key)
        torch.ops.xllm_ops.hisparse_store(rope.reshape(-1, 64).contiguous(), slot_mapping, self._host_rope)

    def invalidate(self, slot_mapping: torch.Tensor) -> None:
        # Recycled physical slots and newly written tokens must never hit old KV.
        safe_slots = torch.where(slot_mapping >= 0, slot_mapping, self._host_key.shape[0]).to(torch.int64)
        self._slot_map.index_copy_(0, safe_slots, torch.full_like(safe_slots, -1, dtype=torch.int32))

    def materialize(
        self,
        topk: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, selected = topk.shape[0], topk.shape[-1]
        if selected % page_size:
            raise ValueError("HiSparse Top-K must be divisible by the cache block size")
        slots = physical_topk_slots(topk, block_table, seq_lens, slot_mapping, page_size)
        rows = batch * selected
        if rows > self._workspace.key.shape[0]:
            raise ValueError("HiSparse selection exceeds the configured workspace")
        # Schedule overlap is disabled: every consumer finishes on the current
        # stream before another layer or graph can overwrite these addresses.
        key = self._workspace.key[:rows]
        rope = self._workspace.rope[:rows]
        torch.ops.xllm_ops.hisparse_gather_out(self._host_key, self._hot_key, slots, self._slot_map, self._tags, key)
        torch.ops.xllm_ops.hisparse_gather_out(self._host_rope, self._hot_rope, slots, self._slot_map, self._tags, rope)
        # Both gathers must finish before any hot slot is overwritten. No prefix
        # sharing or speculative tokens: valid physical selections are unique.
        count = min(slots.numel(), self._hot_tokens)
        self._hot_key[:count].copy_(key[:count])
        self._hot_rope[:count].copy_(rope[:count])
        self._tags[:count].copy_(slots[:count])
        locations = torch.arange(count, dtype=torch.int32, device=slots.device)
        safe_slots = torch.where(slots[:count] >= 0, slots[:count], self._host_key.shape[0]).to(torch.int64)
        self._slot_map.index_copy_(0, safe_slots, locations)
        indices = torch.arange(selected, dtype=torch.int32, device=slots.device).expand(batch, selected)
        indices = torch.where(slots.view(batch, selected) >= 0, indices, -1).unsqueeze(1)
        pages = torch.arange(batch * selected // page_size, dtype=torch.int32, device=slots.device).view(batch, -1)
        lengths = torch.full((batch,), selected, dtype=torch.int32, device=slots.device)
        return key.view(-1, page_size, 1, 512), rope.view(-1, page_size, 1, 64), indices, pages, lengths
