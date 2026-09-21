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

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention import npu_paged_attention
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.fp8_cache import (
    create_e4m3_decode_table,
    dequantize_e4m3,
    quantize_e4m3,
)


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


@pytest.mark.parametrize("slots", [[-1, 0, -1, 2], [-1, -1, -1, -1]])
def test_e4m3_cache_padding_does_not_overwrite_valid_slots(slots: list[int]) -> None:
    cache = torch.full((1, 3, 1, 4), 42, dtype=torch.uint8)
    values = torch.arange(16, dtype=torch.uint8).reshape(4, 1, 4) + 128
    expected = cache.clone().view(3, 4)
    for row, slot in enumerate(slots):
        if slot >= 0:
            expected[slot] = values[row, 0]

    npu_paged_attention.NpuPagedAttentionBackend._update_paged_cache(cache, torch.tensor(slots), values)

    assert torch.equal(cache.view(3, 4), expected)


def test_e4m3_mla_cache_writer_quantizes_latent_and_rope() -> None:
    latent = torch.tensor([[[1.1, -2.2]], [[3.3, -4.4]]], dtype=torch.bfloat16)
    rope = torch.tensor([[[0.2]], [[-0.3]]], dtype=torch.bfloat16)
    nope_cache = torch.zeros((1, 3, 1, 2), dtype=torch.uint8)
    rope_cache = torch.zeros((1, 3, 1, 1), dtype=torch.uint8)

    npu_paged_attention.NpuPagedAttentionBackend._update_mla_cache(
        torch.tensor([2, 0]), latent, rope, nope_cache, rope_cache
    )

    assert torch.equal(nope_cache[0, [2, 0]], quantize_e4m3(latent))
    assert torch.equal(rope_cache[0, [2, 0]], quantize_e4m3(rope))
    assert torch.count_nonzero(nope_cache[0, 1]) == 0


@pytest.mark.parametrize("cache_dtype", [torch.uint8, torch.bfloat16])
def test_mla_preprocess_context_requires_unquantized_cache(cache_dtype: torch.dtype) -> None:
    backend = object.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    slots = torch.tensor([0])
    backend._metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False, slot_mapping=slots)
    key = torch.zeros((1, 128, 1, 512), dtype=cache_dtype)
    value = torch.zeros((1, 128, 1, 64), dtype=cache_dtype)
    backend._kv_caches = [LayerCache(key=key, value=value)]

    context = backend.mla_preprocess_context(SimpleNamespace(layer_id=0))

    if cache_dtype == torch.uint8:
        assert context is None
    else:
        assert context is not None
        assert context.kv_cache is key
        assert context.rope_cache is value
        assert context.slot_mapping is slots


@pytest.mark.parametrize("num_tokens", (3, 129))
@pytest.mark.parametrize("slot_dtype", (torch.int32, torch.int64))
def test_fp8_preprocess_stages_only_current_tokens_and_commits_live_slots(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, slot_dtype: torch.dtype
) -> None:
    buffers: dict[tuple, torch.Tensor] = {}

    def get_buffer(key: tuple, create: Callable[[], torch.Tensor]) -> torch.Tensor:
        if key not in buffers:
            buffers[key] = create()
        return buffers[key]

    monkeypatch.setattr(npu_paged_attention, "get_execution_buffer", get_buffer)
    backend = object.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    backend.dtype = torch.bfloat16
    slots = torch.arange(num_tokens + 2, dtype=slot_dtype) + 259
    slots[1] = -1
    backend._metadata = SimpleNamespace(
        is_prefill=False, is_chunked_prefill=False, has_kv_shard=False, slot_mapping=slots
    )
    key = torch.full((16, 128, 1, 512), 42, dtype=torch.uint8)
    rope = torch.full((16, 128, 1, 64), 42, dtype=torch.uint8)
    backend._kv_caches = [LayerCache(key=key, value=rope)]
    layer = SimpleNamespace(layer_id=0)
    context = backend.mla_preprocess_context(layer, num_tokens=num_tokens)
    assert context is not None and context.commit_cache is not None
    assert context.kv_cache.shape == ((num_tokens + 127) // 128, 128, 1, 512)
    assert context.rope_cache.shape == ((num_tokens + 127) // 128, 128, 1, 64)
    torch.testing.assert_close(context.slot_mapping, torch.arange(num_tokens, dtype=torch.int32))
    assert slots.dtype == slot_dtype
    expected_key, expected_rope = key.clone(), rope.clone()
    torch.manual_seed(20260921)
    for offset in (0, 512):
        live_slots = slots[:num_tokens] + offset
        live_slots[1] = -1
        slots[:num_tokens].copy_(live_slots)
        latent = torch.randn(num_tokens, 512, dtype=torch.bfloat16)
        positional = torch.randn(num_tokens, 64, dtype=torch.bfloat16)
        context.kv_cache.view(-1, 512)[:num_tokens].copy_(latent)
        context.rope_cache.view(-1, 64)[:num_tokens].copy_(positional)
        valid = live_slots >= 0
        expected_key.view(-1, 512)[live_slots[valid]] = quantize_e4m3(latent[valid])
        expected_rope.view(-1, 64)[live_slots[valid]] = quantize_e4m3(positional[valid])
        context.commit_cache()
        torch.testing.assert_close(key, expected_key, atol=0, rtol=0)
        torch.testing.assert_close(rope, expected_rope, atol=0, rtol=0)
    reused = backend.mla_preprocess_context(layer, num_tokens=num_tokens)
    assert reused.kv_cache is context.kv_cache
    assert reused.rope_cache is context.rope_cache
    assert reused.slot_mapping is context.slot_mapping


@pytest.mark.parametrize("unsupported", ("prefill", "chunked", "sharded", "dtype", "shape", "zero", "oversized"))
def test_fp8_preprocess_staging_preserves_fallbacks(unsupported: str) -> None:
    backend = object.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    backend.dtype = torch.float16 if unsupported == "dtype" else torch.bfloat16
    backend._metadata = SimpleNamespace(
        is_prefill=unsupported == "prefill",
        is_chunked_prefill=unsupported == "chunked",
        has_kv_shard=unsupported == "sharded",
        slot_mapping=torch.tensor([0]),
    )
    key = torch.empty((1, 128, 1, 256 if unsupported == "shape" else 512), dtype=torch.uint8)
    rope = torch.empty((1, 128, 1, 64), dtype=torch.uint8)
    backend._kv_caches = [LayerCache(key=key, value=rope)]
    num_tokens = 0 if unsupported == "zero" else 2 if unsupported == "oversized" else 1
    assert backend.mla_preprocess_context(SimpleNamespace(layer_id=0), num_tokens=num_tokens) is None


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
        topk,
        block_table,
        backend._mla_actual_seq_q,
        actual_seq_kv,
        0,
    )

    assert output.dtype == torch.bfloat16
    assert captured["query"] is q_latent
    torch.testing.assert_close(captured["key"], nope_values)
    torch.testing.assert_close(captured["value"], nope_values)
    torch.testing.assert_close(captured["rope"], rope_values)


@pytest.mark.parametrize("num_queries,num_splits", [(1, 16), (2, 8), (3, 8), (6, 4), (12, 2), (16, 1), (25, 1)])
def test_fp8_mla_decode_uses_tilelang_sparse_attention(
    monkeypatch: pytest.MonkeyPatch,
    num_queries: int,
    num_splits: int,
) -> None:
    num_heads = 4
    q_latent = torch.zeros(num_heads, num_queries, 512, dtype=torch.bfloat16).transpose(0, 1)
    q_pe = torch.zeros(num_heads, num_queries, 64, dtype=torch.bfloat16).transpose(0, 1)
    nope_cache = torch.zeros(1, 128, 1, 512, dtype=torch.uint8)
    rope_cache = torch.zeros(1, 128, 1, 64, dtype=torch.uint8)
    topk = torch.zeros(num_queries, 1, 2048, dtype=torch.int32)
    block_table = torch.zeros(num_queries, 1, dtype=torch.int32)
    actual_seq_kv = torch.full((num_queries,), 128, dtype=torch.int32)
    backend = object.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        is_prefill=False, is_chunked_prefill=False, is_mixed=False, is_spec_verify=False
    )
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
        topk,
        block_table,
        backend._mla_actual_seq_q,
        actual_seq_kv,
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
    # Pipeline lowering adds a second GM slot for these five intermediates.
    assert [workspace.size(0) for workspace in args[9:16]] == [48, 48, 48, 48, 48, 24, 24]
    assert args[-2] == num_splits
    if num_splits > 1:
        assert args[16].shape == (24, num_heads, 512)
        assert args[17].shape == (24, 2, 16)
        assert args[16].dtype == args[17].dtype == torch.float32
    else:
        assert args[16] is args[13]
        assert args[17] is args[11]
