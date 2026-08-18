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

"""Unit tests for NPU decode-context-parallel tensor transformations."""

from __future__ import annotations

import torch

from xllm.python.attention.npu_decode_cp import (
    expand_indexer_block_table,
    local_attention_scale,
    local_sequence_lengths,
    localize_cache_slots,
    remap_sparse_indices,
    sparse_attention_lse,
)


def test_localize_cache_slots_routes_logical_sub_blocks_to_owner() -> None:
    logical_slots = torch.tensor([-1, 0, 3, 4, 7, 8, 11, 12, 15, 16], dtype=torch.int32)

    rank_zero = localize_cache_slots(logical_slots, block_size=4, dcp_size=2, dcp_rank=0)
    rank_one = localize_cache_slots(logical_slots, block_size=4, dcp_size=2, dcp_rank=1)

    assert rank_zero.tolist() == [-1, 0, 3, -1, -1, 4, 7, -1, -1, 8]
    assert rank_one.tolist() == [-1, -1, -1, 0, 3, -1, -1, 4, 7, -1]


def test_expand_indexer_block_table_builds_replicated_global_view() -> None:
    block_table = torch.tensor([[3, 7, -1], [2, -1, -1]], dtype=torch.int32)

    expanded = expand_indexer_block_table(block_table, dcp_size=2)

    assert expanded.tolist() == [[6, 7, 14, 15, -1, -1], [4, 5, -1, -1, -1, -1]]


def test_local_sequence_lengths_partition_every_global_token() -> None:
    global_lengths = torch.tensor([0, 1, 4, 5, 8, 9, 12, 16, 17], dtype=torch.int32)

    local_lengths = torch.stack(
        [
            local_sequence_lengths(global_lengths, block_size=4, dcp_size=2, dcp_rank=rank)
            for rank in range(2)
        ]
    )

    torch.testing.assert_close(local_lengths.sum(dim=0), global_lengths)
    assert local_lengths[0].tolist() == [0, 1, 4, 4, 4, 5, 8, 8, 9]
    assert local_lengths[1].tolist() == [0, 0, 0, 1, 4, 4, 4, 8, 8]


def test_remap_sparse_indices_preserves_topk_order_and_packs_invalid_tail() -> None:
    global_indices = torch.tensor(
        [[[0, 4, 1, 7, 8, 12, -1, 15]]],
        dtype=torch.int32,
    )

    rank_zero = remap_sparse_indices(global_indices, block_size=4, dcp_size=2, dcp_rank=0)
    rank_one = remap_sparse_indices(global_indices, block_size=4, dcp_size=2, dcp_rank=1)

    assert rank_zero.tolist() == [[[0, 1, 4, -1, -1, -1, -1, -1]]]
    assert rank_one.tolist() == [[[0, 3, 4, 7, -1, -1, -1, -1]]]


def test_lse_scale_merges_partitioned_attention_like_global_softmax() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    values = torch.randn(2, 3, 4, 5, 6, dtype=torch.float64)
    local_lse = torch.logsumexp(logits, dim=-1)
    local_output = (torch.softmax(logits, dim=-1).unsqueeze(-1) * values).sum(dim=-2)

    merged = torch.zeros_like(local_output[0])
    for rank in range(logits.shape[0]):
        scale = local_attention_scale(local_lse[rank], local_lse)
        merged += local_output[rank] * scale.unsqueeze(-1)

    global_logits = logits.movedim(0, -2).reshape(3, 4, -1)
    global_values = values.movedim(0, -3).reshape(3, 4, -1, 6)
    expected = (torch.softmax(global_logits, dim=-1).unsqueeze(-1) * global_values).sum(dim=-2)
    torch.testing.assert_close(merged, expected)


def test_sparse_attention_lse_flattens_tnd_kv_head_groups() -> None:
    softmax_max = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    softmax_sum = torch.tensor([[[2.0, 4.0, 8.0], [1.0, 2.0, 4.0]]])

    lse = sparse_attention_lse(softmax_max, softmax_sum, num_tokens=2, num_heads=3)

    expected = softmax_max.squeeze(0) + torch.log(softmax_sum.squeeze(0))
    torch.testing.assert_close(lse, expected)


def test_lse_scale_zeroes_empty_rank_and_all_empty_rows() -> None:
    gathered_lse = torch.tensor(
        [
            [[0.0, float("-inf")]],
            [[float("-inf"), float("-inf")]],
        ]
    )

    rank_zero = local_attention_scale(gathered_lse[0], gathered_lse)
    rank_one = local_attention_scale(gathered_lse[1], gathered_lse)

    torch.testing.assert_close(rank_zero, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(rank_one, torch.zeros(1, 2))
