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

"""Compare compact FP8 prefill pages with native SFA and a CPU oracle."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("query_lengths,kv_lengths", (([8], [8]), ([3], [259]), ([2, 3], [129, 257])))
def test_compact_fp8_prefill_matches_native_and_reference(
    monkeypatch: pytest.MonkeyPatch, query_lengths: list[int], kv_lengths: list[int]
) -> None:
    import xllm

    _ = xllm.xllm_export
    from xllm.python.attention import npu_paged_attention
    from xllm.python.attention.fp8_cache import dequantize_e4m3, quantize_e4m3
    from xllm.python.kernels_npu.sparse_attention import sparse_flash_attention_out

    monkeypatch.setattr(npu_paged_attention, "get_execution_buffer", lambda _key, create: create())
    monkeypatch.setattr(
        npu_paged_attention.kernels, "sparse_flash_attention_out", sparse_flash_attention_out, raising=False
    )
    torch.manual_seed(20260921)
    query_count = sum(query_lengths)
    width = (max(kv_lengths) + 127) // 128
    pages = torch.full((len(kv_lengths), width), -1, dtype=torch.int32)
    for row, length in enumerate(kv_lengths):
        count = (length + 127) // 128
        page_ids = torch.arange(count, dtype=torch.int32) + 8 + row * 4
        if count > 1:
            page_ids[:2] = page_ids[:2].flip(0)
        pages[row, :count] = page_ids
    raw_key = quantize_e4m3(torch.randn(32, 128, 1, 512).bfloat16() * 0.25)
    raw_rope = quantize_e4m3(torch.randn(32, 128, 1, 64).bfloat16() * 0.25)
    decoded_key = dequantize_e4m3(raw_key, torch.float32).view(-1, 512)
    decoded_rope = dequantize_e4m3(raw_rope, torch.float32).view(-1, 64)
    q = torch.randn(query_count, 4, 512).bfloat16() * 0.25
    q_rope = torch.randn(query_count, 4, 64).bfloat16() * 0.25
    topk = torch.full((query_count, 1, 2048), -1, dtype=torch.int32)
    expected = []
    query_start = 0
    for row, (query_length, kv_length) in enumerate(zip(query_lengths, kv_lengths)):
        for local_query in range(query_length):
            query = query_start + local_query
            selected = torch.arange(kv_length - query_length + local_query + 1)
            topk[query, 0, : selected.numel()] = selected.to(torch.int32)
            physical = pages[row, selected // 128].long() * 128 + selected % 128
            keys, rope_keys = decoded_key[physical], decoded_rope[physical]
            scores = (q[query].float() @ keys.T + q_rope[query].float() @ rope_keys.T) * 0.0625
            expected.append(torch.softmax(scores, dim=-1) @ keys)
        query_start += query_length

    block_table = pages.npu()
    backend = object.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        is_prefill=query_lengths == kv_lengths,
        is_chunked_prefill=query_lengths != kv_lengths,
        is_mixed=len(query_lengths) > 1,
    )
    backend._block_table_i32 = block_table
    backend._fp8_sfa_pages = None
    backend.scale = 0.0625
    args = [q.npu(), q_rope.npu(), raw_key.npu(), raw_rope.npu(), topk.npu()]
    actual_q = torch.tensor(query_lengths, dtype=torch.int32).cumsum(0).to(torch.int32).npu()
    actual_kv = torch.tensor(kv_lengths, dtype=torch.int32).npu()
    compact = backend._mla_sparse(*args, block_table, actual_q, actual_kv, 0)
    assert backend._fp8_sfa_pages[0].numel() == pages.numel()
    full = backend._mla_sparse(*args, block_table.clone(), actual_q, actual_kv, 0)
    torch.testing.assert_close(compact, full, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(compact.cpu().float(), torch.stack(expected), atol=2e-3, rtol=2e-2)
