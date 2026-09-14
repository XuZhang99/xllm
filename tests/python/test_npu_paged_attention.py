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

"""Tests for the NPU paged-attention backend."""

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import LayerCache  # noqa: E402
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
)


def test_uses_first_nonempty_key_cache() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    linear_cache = LayerCache(
        key=None,
        value=None,
        conv=torch.empty(8, 3, 64),
        ssm=torch.empty(8, 2, 4, 4),
    )
    key_cache = torch.empty(17, 128, 2, 64)
    value_cache = torch.empty_like(key_cache)

    backend.bind_kv_caches(
        [
            linear_cache,
            LayerCache(key=key_cache, value=value_cache),
        ]
    )

    assert backend.num_kv_blocks == 17
    assert backend.page_size == 128


def test_indexer_metadata_is_generated_inside_capture_and_retained(monkeypatch) -> None:
    from xllm.python.attention import npu_paged_attention
    from xllm.python.model_executor.forward_context import (
        AclGraphCaptureContext,
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._mla_actual_seq_q = torch.arange(1, 5, dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.full((4,), 16, dtype=torch.int32)
    backend._mla_max_seqlen_q = 1
    backend._mla_max_seqlen_k = 16384
    backend._mla_quant_indexer_metadata = {}
    generated = []

    def generate(*args):
        metadata = torch.full((1024,), len(generated), dtype=torch.int32)
        generated.append(metadata)
        return metadata

    monkeypatch.setattr(npu_paged_attention.kernels, "quant_lightning_indexer_metadata", generate, raising=False)
    state = AclGraphExecutionState({})
    context = ForwardContext(backend, torch.device("cpu"), None, [], execution_state=state)
    with forward_context(context):
        warmup = backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1)
        assert backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1) is warmup
    context = ForwardContext(
        backend,
        torch.device("cpu"),
        None,
        [],
        acl_graph=AclGraphCaptureContext(None, []),
        execution_state=state,
    )
    with forward_context(context):
        captured = backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1)
        assert backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1) is captured
    assert len(generated) == 2
    assert captured is not warmup
    backend._mla_quant_indexer_metadata.clear()
    assert any(value is captured for value in state.persistent_buffers.values())
