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

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention import npu_paged_attention
from xllm.python.attention.fp8_cache import (
    create_e4m3_decode_table,
    dequantize_e4m3,
    quantize_e4m3,
)
from xllm.python.attention.quantized_mla import quantized_sparse_mla_attention


def test_e4m3_cache_encoding_matches_pytorch_float8_bits() -> None:
    values = torch.tensor(
        [
            -448.0,
            -16.0,
            -1.5,
            -(2.0**-6),
            -(2.0**-9),
            0.0,
            2.0**-9,
            2.0**-6,
            1.5,
            16.0,
            448.0,
        ],
        dtype=torch.bfloat16,
    )

    encoded = quantize_e4m3(values)
    expected = values.to(torch.float8_e4m3fn).view(torch.uint8)

    assert torch.equal(encoded, expected)
    assert torch.equal(dequantize_e4m3(encoded), expected.view(torch.float8_e4m3fn).to(torch.bfloat16))


def test_e4m3_cache_encoding_saturates_out_of_range_values() -> None:
    values = torch.tensor([-float("inf"), -500.0, 500.0, float("inf"), float("nan")])

    decoded = dequantize_e4m3(quantize_e4m3(values), torch.float32)

    torch.testing.assert_close(decoded[:4], torch.tensor([-448.0, -448.0, 448.0, 448.0]))
    assert decoded[4].item() == 0.0


def test_e4m3_decode_table_covers_every_raw_byte() -> None:
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8)

    table = create_e4m3_decode_table(torch.device("cpu"))

    assert table.dtype == torch.float32
    torch.testing.assert_close(table, dequantize_e4m3(raw, torch.float32))


def test_e4m3_paged_cache_update_preserves_raw_high_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = torch.zeros(1, 2, 1, 3, dtype=torch.uint8)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
    values = torch.tensor([[[254, 216, 188]], [[129, 1, 126]]], dtype=torch.uint8)

    def scatter_nd_update(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("uint8 cache updates must not use ScatterNdUpdateV2")

    monkeypatch.setattr(
        npu_paged_attention.kernels,
        "scatter_nd_update",
        scatter_nd_update,
        raising=False,
    )

    npu_paged_attention.NpuPagedAttentionBackend._update_paged_cache(cache, slot_mapping, values)

    assert torch.equal(cache.view(2, 3), values.view(2, 3))


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


def test_fp8_mla_dequantizes_caches_before_sparse_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q_latent = torch.zeros(2, 1, 2, dtype=torch.bfloat16)
    q_pe = torch.zeros(2, 1, 1, dtype=torch.bfloat16)
    nope_values = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 2.0]], [[0.5, 0.5]], [[-1.0, 1.0]]]],
        dtype=torch.bfloat16,
    )
    rope_values = torch.zeros(1, 4, 1, 1, dtype=torch.bfloat16)
    topk = torch.tensor([[[0, 2, 3]], [[1, 2, 3]]], dtype=torch.int32)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    actual_seq_q = torch.tensor([2], dtype=torch.int32)
    actual_seq_kv = torch.tensor([4], dtype=torch.int32)
    backend = object.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    backend._mla_actual_seq_q = actual_seq_q
    backend._mla_actual_seq_kv = actual_seq_kv
    backend.scale = 1.0
    captured: dict[str, torch.Tensor] = {}

    def sparse_flash_attention_out(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *_args: object,
    ) -> torch.Tensor:
        captured["query"] = query
        captured["key"] = key
        captured["value"] = value
        captured["rope"] = _args[5]
        return _args[-1]

    monkeypatch.setattr(
        npu_paged_attention,
        "get_execution_buffer",
        lambda _key, factory: factory(),
    )
    monkeypatch.setattr(
        npu_paged_attention.kernels,
        "sparse_flash_attention_out",
        sparse_flash_attention_out,
        raising=False,
    )

    output = backend._mla_sparse(
        q_latent,
        q_pe,
        quantize_e4m3(nope_values),
        quantize_e4m3(rope_values),
        None,
        topk,
        block_table,
        0,
    )

    assert output.dtype == torch.bfloat16
    assert captured["query"] is q_latent
    torch.testing.assert_close(captured["key"], nope_values)
    torch.testing.assert_close(captured["value"], nope_values)
    torch.testing.assert_close(captured["rope"], rope_values)


def test_fp8_mla_decode_uses_tilelang_sparse_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_queries = 2
    num_heads = 4
    q_latent = torch.zeros(num_heads, num_queries, 512, dtype=torch.bfloat16).transpose(0, 1)
    q_pe = torch.zeros(num_heads, num_queries, 64, dtype=torch.bfloat16).transpose(0, 1)
    nope_cache = torch.zeros(1, 128, 1, 512, dtype=torch.uint8)
    rope_cache = torch.zeros(1, 128, 1, 64, dtype=torch.uint8)
    topk = torch.zeros(num_queries, 1, 2048, dtype=torch.int32)
    block_table = torch.zeros(num_queries, 1, dtype=torch.int32)
    actual_seq_kv = torch.full((num_queries,), 128, dtype=torch.int32)
    backend = object.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
    backend._mla_actual_seq_q = torch.arange(1, num_queries + 1, dtype=torch.int32)
    backend._mla_actual_seq_kv = actual_seq_kv
    backend._fp8_e4m3_decode_table = None
    backend._fp8_mla_workspaces = None
    backend.scale = 0.0625
    captured: dict[str, object] = {}

    def glm52_fp8_sparse_mla_attention_out(*args: object) -> torch.Tensor:
        captured["args"] = args
        return args[8]  # type: ignore[return-value]

    def sparse_flash_attention_out(*_args: object) -> torch.Tensor:
        raise AssertionError("eligible FP8 decode must use the fused TileLang kernel")

    monkeypatch.setattr(
        npu_paged_attention,
        "get_execution_buffer",
        lambda _key, factory: factory(),
    )
    monkeypatch.setattr(
        npu_paged_attention.kernels,
        "glm52_fp8_sparse_mla_attention_out",
        glm52_fp8_sparse_mla_attention_out,
        raising=False,
    )
    monkeypatch.setattr(
        npu_paged_attention.kernels,
        "sparse_flash_attention_out",
        sparse_flash_attention_out,
        raising=False,
    )

    output = backend._mla_sparse(
        q_latent,
        q_pe,
        nope_cache,
        rope_cache,
        None,
        topk,
        block_table,
        0,
    )

    assert output.shape == q_latent.shape
    assert output.is_contiguous()
    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[0] is q_latent
    assert args[1] is q_pe
    assert args[2] is nope_cache
    assert args[3] is rope_cache
    assert args[6] is actual_seq_kv
    assert args[7].shape == (256,)
    assert args[-1] == backend.scale


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
