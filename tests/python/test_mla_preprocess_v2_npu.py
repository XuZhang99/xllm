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

"""NPU smoke and integration tests for the fused MLA preprocess path."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

_DTYPE = torch.bfloat16
_TOKEN_NUM = 2
_HIDDEN_SIZE = 7168
_Q_LORA_RANK = 1536
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_NUM_HEADS = 2
_BLOCK_NUM = 2
_BLOCK_SIZE = 128
_CACHE_MODE = 1  # krope_ctkv
_QUANT_MODE = 0  # per_tensor_quant_asymm


def _require_mla_preprocess_v2() -> ModuleType:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    try:
        import xllm

        # Loading the built export module registers torch.ops.xllm_ops.
        _ = xllm.xllm_export
        from xllm.python.kernels_npu import mla
    except (ImportError, OSError) as exc:
        pytest.skip(f"xLLM native extension is not built: {exc}")
    if not mla.has_mla_preprocess_v2():
        pytest.skip("aclnnMlaPreprocessV2 is not available in the loaded runtime")
    return mla


def _make_inputs(mla: ModuleType) -> dict[str, torch.Tensor | int | float | bool]:
    device = torch.device("npu")
    torch.manual_seed(0)

    qkv_weight = torch.randint(
        0,
        7,
        (_KV_LORA_RANK + _QK_ROPE_HEAD_DIM + _Q_LORA_RANK, _HIDDEN_SIZE),
        dtype=torch.int8,
        device=device,
    )
    qkv_deq_scale = torch.ones(
        _KV_LORA_RANK + _QK_ROPE_HEAD_DIM + _Q_LORA_RANK,
        dtype=torch.float32,
        device=device,
    )
    qkv_quant_bias = torch.zeros(
        _KV_LORA_RANK + _QK_ROPE_HEAD_DIM + _Q_LORA_RANK,
        dtype=torch.int32,
        device=device,
    )
    qkv_weight, qkv_deq_scale, qkv_quant_bias = mla.prepare_mla_preprocess_v2_qkv(
        qkv_weight,
        qkv_deq_scale,
        qkv_quant_bias,
        _KV_LORA_RANK,
        _QK_ROPE_HEAD_DIM,
    )

    q_b_weight = torch.randint(
        0,
        7,
        (_NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM), _Q_LORA_RANK),
        dtype=torch.int8,
        device=device,
    )
    q_b_deq_scale = torch.ones(
        _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    q_b_quant_bias = torch.zeros(
        _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
        dtype=torch.int32,
        device=device,
    )
    q_b_weight, q_b_deq_scale, q_b_quant_bias = mla.prepare_mla_preprocess_v2_q_b(
        q_b_weight,
        q_b_deq_scale,
        q_b_quant_bias,
        _NUM_HEADS,
        _QK_NOPE_HEAD_DIM,
        _QK_ROPE_HEAD_DIM,
    )

    w_uk = torch.randn(
        (_NUM_HEADS, _QK_NOPE_HEAD_DIM, _KV_LORA_RANK),
        dtype=_DTYPE,
        device=device,
    )
    w_uk = torch_npu.npu_format_cast(w_uk, 29)

    inputs: dict[str, torch.Tensor | int | float | bool] = {
        "hidden": torch.randn((_TOKEN_NUM, _HIDDEN_SIZE), dtype=_DTYPE, device=device),
        "input_norm_weight": torch.ones(_HIDDEN_SIZE, dtype=_DTYPE, device=device),
        "input_norm_bias": torch.zeros(_HIDDEN_SIZE, dtype=_DTYPE, device=device),
        "qkv_input_scale": torch.tensor([0.25], dtype=_DTYPE, device=device),
        "qkv_input_offset": torch.zeros(1, dtype=torch.int8, device=device),
        "qkv_weight": qkv_weight,
        "qkv_deq_scale": qkv_deq_scale,
        "qkv_quant_bias": qkv_quant_bias,
        "q_norm_weight": torch.ones(_Q_LORA_RANK, dtype=_DTYPE, device=device),
        "q_norm_bias": torch.zeros(_Q_LORA_RANK, dtype=_DTYPE, device=device),
        "q_b_input_scale": torch.tensor([0.25], dtype=_DTYPE, device=device),
        "q_b_input_offset": torch.zeros(1, dtype=torch.int8, device=device),
        "q_b_weight": q_b_weight,
        "q_b_deq_scale": q_b_deq_scale,
        "q_b_quant_bias": q_b_quant_bias,
        "kv_norm_weight": torch.ones(_KV_LORA_RANK, dtype=_DTYPE, device=device),
        "rope_cos": torch.randn((_TOKEN_NUM, _QK_ROPE_HEAD_DIM), dtype=_DTYPE, device=device),
        "rope_sin": torch.randn((_TOKEN_NUM, _QK_ROPE_HEAD_DIM), dtype=_DTYPE, device=device),
        "w_uk": w_uk,
        "kv_cache": torch.full(
            (_BLOCK_NUM, _BLOCK_SIZE, _KV_LORA_RANK),
            float("nan"),
            dtype=_DTYPE,
            device=device,
        ),
        "rope_cache": torch.full(
            (_BLOCK_NUM, _BLOCK_SIZE, _QK_ROPE_HEAD_DIM),
            float("nan"),
            dtype=_DTYPE,
            device=device,
        ),
        "slot_mapping": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "kv_lora_rank": _KV_LORA_RANK,
        "q_lora_rank": _Q_LORA_RANK,
        "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
        "norm_epsilon": 1e-5,
    }
    return inputs


def _run_low_level_op(inputs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    num_tokens = inputs["hidden"].shape[0]
    return torch.ops.xllm_ops.mla_preprocess_v2(
        inputs["hidden"],
        inputs["input_norm_weight"],
        inputs["input_norm_bias"],
        inputs["qkv_input_scale"],
        inputs["qkv_input_offset"],
        inputs["qkv_weight"],
        inputs["qkv_deq_scale"],
        inputs["qkv_quant_bias"],
        inputs["q_norm_weight"],
        inputs["q_norm_bias"],
        inputs["q_b_input_scale"],
        inputs["q_b_input_offset"],
        inputs["q_b_weight"],
        inputs["q_b_deq_scale"],
        inputs["q_b_quant_bias"],
        inputs["kv_norm_weight"],
        inputs["rope_cos"].view(num_tokens, -1),
        inputs["rope_sin"].view(num_tokens, -1),
        inputs["w_uk"],
        inputs["kv_cache"],
        inputs["rope_cache"],
        inputs["slot_mapping"],
        inputs["qkv_input_scale"],
        inputs["q_b_input_scale"],
        inputs["q_lora_rank"],
        inputs["qk_rope_head_dim"],
        inputs["qk_rope_head_dim"],
        inputs["norm_epsilon"],
        2,
        2,
        True,
        True,
        True,
        _CACHE_MODE,
        _QUANT_MODE,
        False,
        1,
        True,
    )


def test_mla_preprocess_v2_writes_outputs_and_selected_cache_slots() -> None:
    mla = _require_mla_preprocess_v2()
    inputs = _make_inputs(mla)

    q_latent, kv_cache, q_pe, rope_cache, q_c = _run_low_level_op(inputs)
    torch.npu.synchronize()

    assert q_latent.shape == (_TOKEN_NUM, _NUM_HEADS, _KV_LORA_RANK)
    assert q_pe.shape == (_TOKEN_NUM, _NUM_HEADS, _QK_ROPE_HEAD_DIM)
    assert q_c.shape == (_TOKEN_NUM, _Q_LORA_RANK)
    assert torch.isfinite(q_latent).all()
    assert torch.isfinite(q_pe).all()
    assert torch.isfinite(q_c).all()
    assert torch.isfinite(kv_cache[0, :_TOKEN_NUM]).all()
    assert torch.isfinite(rope_cache[0, :_TOKEN_NUM]).all()
    assert torch.isnan(kv_cache[1]).all()
    assert torch.isnan(rope_cache[1]).all()


def test_mla_preprocess_v2_python_wrapper_matches_low_level_op() -> None:
    mla = _require_mla_preprocess_v2()
    low_level_inputs = _make_inputs(mla)
    wrapper_inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in low_level_inputs.items()
    }

    low_level_outputs = _run_low_level_op(low_level_inputs)
    wrapper_outputs = mla.deepseek_mla_preprocess_decode_v2(
        wrapper_inputs["hidden"],
        wrapper_inputs["input_norm_weight"],
        wrapper_inputs["input_norm_bias"],
        wrapper_inputs["qkv_input_scale"],
        wrapper_inputs["qkv_input_offset"],
        wrapper_inputs["qkv_weight"],
        wrapper_inputs["qkv_deq_scale"],
        wrapper_inputs["qkv_quant_bias"],
        wrapper_inputs["q_norm_weight"],
        wrapper_inputs["q_norm_bias"],
        wrapper_inputs["q_b_input_scale"],
        wrapper_inputs["q_b_input_offset"],
        wrapper_inputs["q_b_weight"],
        wrapper_inputs["q_b_deq_scale"],
        wrapper_inputs["q_b_quant_bias"],
        wrapper_inputs["kv_norm_weight"],
        wrapper_inputs["rope_cos"],
        wrapper_inputs["rope_sin"],
        wrapper_inputs["w_uk"],
        wrapper_inputs["kv_cache"],
        wrapper_inputs["rope_cache"],
        wrapper_inputs["slot_mapping"],
        _KV_LORA_RANK,
        _Q_LORA_RANK,
        _QK_ROPE_HEAD_DIM,
        1e-5,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(wrapper_outputs[0], low_level_outputs[4], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(wrapper_outputs[1], low_level_outputs[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(wrapper_outputs[2], low_level_outputs[2], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("slot_dtype", (torch.int32, torch.int64))
def test_fp8_staged_preprocess_replays_live_slots(monkeypatch: pytest.MonkeyPatch, slot_dtype: torch.dtype) -> None:
    mla = _require_mla_preprocess_v2()
    from xllm.python.attention import npu_paged_attention
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.fp8_cache import quantize_e4m3
    from xllm.python.kernels_npu.sparse_attention import fp8_mla_cache_write

    # conftest stubs the registry; bind the real native wrapper for this test.
    monkeypatch.setattr(npu_paged_attention.kernels, "fp8_mla_cache_write", fp8_mla_cache_write, raising=False)
    monkeypatch.setattr(npu_paged_attention, "get_execution_buffer", lambda _key, create: create())
    inputs = _make_inputs(mla)
    raw_key = torch.full((_BLOCK_NUM, 128, 1, 512), 42, dtype=torch.uint8, device="npu")
    raw_rope = torch.full((_BLOCK_NUM, 128, 1, 64), 42, dtype=torch.uint8, device="npu")
    slots = torch.full((_TOKEN_NUM,), -1, dtype=slot_dtype, device="npu")
    backend = object.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    backend.dtype = _DTYPE
    backend._kv_caches = [LayerCache(key=raw_key, value=raw_rope)]
    backend._metadata = SimpleNamespace(
        is_prefill=False, is_chunked_prefill=False, has_kv_shard=False, slot_mapping=slots
    )
    context = backend.mla_preprocess_context(SimpleNamespace(layer_id=0), num_tokens=_TOKEN_NUM)
    assert context is not None and context.commit_cache is not None
    assert context.slot_mapping.dtype == torch.int32
    staged_inputs = dict(
        inputs, kv_cache=context.kv_cache, rope_cache=context.rope_cache, slot_mapping=context.slot_mapping
    )
    _run_low_level_op(staged_inputs)
    context.commit_cache()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        staged_outputs = _run_low_level_op(staged_inputs)
        context.commit_cache()

    expected_key, expected_rope = raw_key.cpu(), raw_rope.cpu()
    for live_slots in ([127, 128], [-1, 253]):
        inputs["hidden"].normal_()
        slots.copy_(torch.tensor(live_slots, dtype=slot_dtype, device="npu"))
        inputs["slot_mapping"].copy_(
            torch.tensor([max(slot, 0) for slot in live_slots], dtype=torch.int32, device="npu")
        )
        reference = _run_low_level_op(inputs)
        graph.replay()
        for output_index in (0, 2, 4):
            torch.testing.assert_close(staged_outputs[output_index], reference[output_index], rtol=1e-2, atol=1e-2)
        key_values = inputs["kv_cache"].cpu().view(-1, 512)
        rope_values = inputs["rope_cache"].cpu().view(-1, 64)
        for slot in live_slots:
            if slot >= 0:
                expected_key.view(-1, 512)[slot] = quantize_e4m3(key_values[slot])
                expected_rope.view(-1, 64)[slot] = quantize_e4m3(rope_values[slot])
        torch.testing.assert_close(raw_key.cpu(), expected_key, atol=0, rtol=0)
        torch.testing.assert_close(raw_rope.cpu(), expected_rope, atol=0, rtol=0)
