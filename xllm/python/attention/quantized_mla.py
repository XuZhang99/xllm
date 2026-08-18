# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch fallback for sparse MLA over an INT8 latent cache."""

from __future__ import annotations

import torch

_QUERY_CHUNK_SIZE = 64


def _query_batch_indices(
    num_queries: int,
    actual_seq_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_positions = torch.arange(
        num_queries,
        dtype=actual_seq_q.dtype,
        device=actual_seq_q.device,
    )
    batch_indices = torch.searchsorted(actual_seq_q, query_positions, right=True)
    query_starts = torch.cat([actual_seq_q.new_zeros(1), actual_seq_q[:-1]])
    query_lengths = actual_seq_q - query_starts
    return batch_indices, query_starts, query_lengths


def _gather_quantized_cache(
    cache: torch.Tensor,
    cache_scale: torch.Tensor,
    logical_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    block_table: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = cache.size(1)
    logical_blocks = torch.div(logical_indices, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(logical_indices, block_size)
    valid_blocks = logical_blocks < block_table.size(1)
    safe_logical_blocks = logical_blocks.clamp(0, block_table.size(1) - 1)
    physical_blocks = block_table[batch_indices.unsqueeze(1), safe_logical_blocks]
    valid_blocks = valid_blocks & (physical_blocks >= 0) & (physical_blocks < cache.size(0))
    safe_physical_blocks = physical_blocks.clamp(0, cache.size(0) - 1)
    selected_cache = cache[safe_physical_blocks, block_offsets, 0]
    selected_scale = cache_scale[safe_physical_blocks, block_offsets, 0]
    dequantized = selected_cache.to(dtype) * selected_scale.to(dtype).unsqueeze(-1)
    return dequantized, valid_blocks


def quantized_sparse_mla_attention(
    q_latent: torch.Tensor,
    q_pe: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    nope_cache_scale: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_q: torch.Tensor,
    actual_seq_kv: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run sparse absorbed MLA while keeping the persistent latent cache INT8.

    The current NPU SparseFlashAttention custom operator accepts only floating
    point caches. This fallback gathers only the selected paged-cache rows,
    dequantizes them with their per-token scales, and computes attention with
    regular PyTorch operators. Query chunking bounds the temporary gather and
    score buffers independently of the prompt length.
    """
    if topk.dim() == 3:
        if topk.size(1) != 1:
            raise ValueError("quantized sparse MLA requires one KV head")
        logical_indices = topk[:, 0]
    elif topk.dim() == 2:
        logical_indices = topk
    else:
        raise ValueError("topk must have shape [T, 1, K] or [T, K]")

    num_queries = q_latent.size(0)
    if num_queries == 0:
        return torch.empty_like(q_latent)
    batch_indices, query_starts, query_lengths = _query_batch_indices(
        num_queries,
        actual_seq_q,
    )
    outputs: list[torch.Tensor] = []
    for start in range(0, num_queries, _QUERY_CHUNK_SIZE):
        end = min(start + _QUERY_CHUNK_SIZE, num_queries)
        chunk_batches = batch_indices[start:end]
        chunk_indices = logical_indices[start:end]
        selected_nope, valid_blocks = _gather_quantized_cache(
            nope_cache,
            nope_cache_scale,
            chunk_indices.clamp_min(0),
            chunk_batches,
            block_table,
            q_latent.dtype,
        )
        block_size = rope_cache.size(1)
        logical_blocks = torch.div(chunk_indices.clamp_min(0), block_size, rounding_mode="floor")
        block_offsets = torch.remainder(chunk_indices.clamp_min(0), block_size)
        safe_logical_blocks = logical_blocks.clamp(0, block_table.size(1) - 1)
        physical_blocks = block_table[chunk_batches.unsqueeze(1), safe_logical_blocks]
        safe_physical_blocks = physical_blocks.clamp(0, rope_cache.size(0) - 1)
        selected_rope = rope_cache[safe_physical_blocks, block_offsets, 0].to(q_pe.dtype)

        chunk_query_positions = torch.arange(
            start,
            end,
            dtype=actual_seq_q.dtype,
            device=actual_seq_q.device,
        )
        local_query_positions = chunk_query_positions - query_starts[chunk_batches]
        last_visible_positions = actual_seq_kv[chunk_batches] - query_lengths[chunk_batches] + local_query_positions
        valid = (
            valid_blocks
            & (chunk_indices >= 0)
            & (chunk_indices < actual_seq_kv[chunk_batches].unsqueeze(1))
            & (chunk_indices <= last_visible_positions.unsqueeze(1))
        )

        scores = torch.matmul(q_latent[start:end], selected_nope.transpose(1, 2))
        scores = scores + torch.matmul(q_pe[start:end], selected_rope.transpose(1, 2))
        scores = scores * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.to(torch.float32), dim=-1).to(q_latent.dtype)
        probabilities = probabilities * valid.unsqueeze(1).to(probabilities.dtype)
        outputs.append(torch.matmul(probabilities, selected_nope))
    return torch.cat(outputs, dim=0)


__all__ = ["quantized_sparse_mla_attention"]
