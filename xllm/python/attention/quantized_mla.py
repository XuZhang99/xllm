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

"""PyTorch fallback for sparse MLA over quantized latent caches."""

from __future__ import annotations

import torch

from xllm.python.attention.fp8_cache import dequantize_e4m3

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


def _gather_cache_rows(
    cache: torch.Tensor,
    logical_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    block_table: torch.Tensor,
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
    return selected_cache, valid_blocks


def _dequantize_nope_cache(
    selected_cache: torch.Tensor,
    cache_scale: torch.Tensor | None,
    logical_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    block_table: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if selected_cache.dtype == torch.uint8:
        return dequantize_e4m3(selected_cache, torch.bfloat16)
    if selected_cache.dtype != torch.int8:
        return selected_cache.to(dtype)
    if cache_scale is None:
        raise ValueError("INT8 MLA latent cache requires a scale cache")
    selected_scale, _ = _gather_cache_rows(cache_scale, logical_indices, batch_indices, block_table)
    return selected_cache.to(dtype) * selected_scale.to(dtype).unsqueeze(-1)


def quantized_sparse_mla_attention(
    q_latent: torch.Tensor,
    q_pe: torch.Tensor,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    nope_cache_scale: torch.Tensor | None,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_q: torch.Tensor,
    actual_seq_kv: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run sparse absorbed MLA over persistent INT8 or E4M3 caches.

    The current NPU SparseFlashAttention custom operator accepts only floating
    point caches. This fallback gathers only the selected paged-cache rows,
    dequantizes FP8 rows to BF16, and computes attention with regular PyTorch
    operators. Query chunking bounds the temporary gather and score buffers
    independently of the prompt length.
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
        safe_chunk_indices = chunk_indices.clamp_min(0)
        selected_nope, valid_blocks = _gather_cache_rows(
            nope_cache,
            safe_chunk_indices,
            chunk_batches,
            block_table,
        )
        selected_nope = _dequantize_nope_cache(
            selected_nope,
            nope_cache_scale,
            safe_chunk_indices,
            chunk_batches,
            block_table,
            q_latent.dtype,
        )
        selected_rope, valid_rope_blocks = _gather_cache_rows(
            rope_cache,
            safe_chunk_indices,
            chunk_batches,
            block_table,
        )
        valid_blocks = valid_blocks & valid_rope_blocks
        if selected_rope.dtype == torch.uint8:
            selected_rope = dequantize_e4m3(selected_rope, torch.bfloat16)
        else:
            selected_rope = selected_rope.to(q_pe.dtype)

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

        q_latent_chunk = q_latent[start:end].to(selected_nope.dtype)
        q_pe_chunk = q_pe[start:end].to(selected_rope.dtype)
        scores = torch.matmul(q_latent_chunk, selected_nope.transpose(1, 2))
        scores = scores + torch.matmul(q_pe_chunk, selected_rope.transpose(1, 2))
        scores = scores * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.to(torch.float32), dim=-1).to(selected_nope.dtype)
        probabilities = probabilities * valid.unsqueeze(1).to(probabilities.dtype)
        outputs.append(torch.matmul(probabilities, selected_nope))
    return torch.cat(outputs, dim=0)


__all__ = ["quantized_sparse_mla_attention"]
