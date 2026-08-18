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

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.quantized_mla import quantized_sparse_mla_attention


def test_quantized_sparse_mla_dequantizes_selected_cache_rows() -> None:
    q_latent = torch.zeros(2, 1, 2, dtype=torch.float32)
    q_pe = torch.zeros(2, 1, 1, dtype=torch.float32)
    nope_cache = torch.tensor(
        [[[[1, 0]], [[0, 1]], [[1, 1]], [[-1, 1]]]],
        dtype=torch.int8,
    )
    nope_scale = torch.tensor([[[1.0], [2.0], [0.5], [1.0]]])
    rope_cache = torch.zeros(1, 4, 1, 1)
    topk = torch.tensor([[[0, 2, 3]], [[1, 2, 3]]], dtype=torch.int32)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    actual_seq_q = torch.tensor([2], dtype=torch.int32)
    actual_seq_kv = torch.tensor([4], dtype=torch.int32)

    output = quantized_sparse_mla_attention(
        q_latent,
        q_pe,
        nope_cache,
        rope_cache,
        nope_scale,
        topk,
        block_table,
        actual_seq_q,
        actual_seq_kv,
        1.0,
    )

    expected = torch.tensor([[[0.75, 0.25]], [[-1.0 / 6.0, 7.0 / 6.0]]])
    torch.testing.assert_close(output, expected)


def test_quantized_sparse_mla_rejects_multiple_kv_heads() -> None:
    with pytest.raises(ValueError, match="requires one KV head"):
        quantized_sparse_mla_attention(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1, 2, dtype=torch.int8),
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1, 1),
            torch.zeros(1, 2, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            1.0,
        )


def test_quantized_sparse_mla_accepts_empty_query_batch() -> None:
    output = quantized_sparse_mla_attention(
        torch.zeros(0, 1, 2),
        torch.zeros(0, 1, 1),
        torch.zeros(1, 1, 1, 2, dtype=torch.int8),
        torch.zeros(1, 1, 1, 1),
        torch.ones(1, 1, 1),
        torch.zeros(0, 1, 1, dtype=torch.int32),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        1.0,
    )

    assert output.shape == (0, 1, 2)
