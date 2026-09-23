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

from types import ModuleType
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


@pytest.mark.parametrize(
    "q_lora_rank,qk_nope_head_dim,expected",
    [(1536, 128, True), (2048, 128, False), (1536, 192, False), (2048, 192, False)],
)
def test_mla_preprocess_v2_rejects_unsupported_projection_dimensions(
    monkeypatch: pytest.MonkeyPatch, q_lora_rank: int, qk_nope_head_dim: int, expected: bool
) -> None:
    mla = _require_mla_preprocess_v2()
    monkeypatch.setattr(mla, "has_mla_preprocess_v2", lambda: True)
    assert (
        mla.supports_mla_preprocess_v2(512, 64, q_lora_rank=q_lora_rank, qk_nope_head_dim=qk_nope_head_dim) is expected
    )


@pytest.mark.parametrize("tokens", (1, 4, 16))
@pytest.mark.parametrize("graph_mode", (False, True))
@torch.inference_mode()
def test_glm_decode_preprocess_matches_unfused_projections(
    monkeypatch: pytest.MonkeyPatch, tokens: int, graph_mode: bool
) -> None:
    mla = _require_mla_preprocess_v2()
    from xllm.python.kernels_npu import attention as attention_kernels
    from xllm.python.kernels_npu import linear, normalization, quantization, rotary_embedding
    from xllm.python.models import glm5_2

    for name, value in {
        "supports_mla_preprocess_v2": mla.supports_mla_preprocess_v2,
        "prepare_quant_weight": linear.prepare_quant_weight,
        "rms_norm": normalization.rms_norm,
        "quant_matmul": quantization.quant_matmul,
        "quantize_per_tensor": quantization.quantize_per_tensor,
    }.items():
        monkeypatch.setattr(glm5_2.kernels, name, value, raising=False)
    cfg = glm5_2.Glm52Config(hidden_size=256, n_heads=4, n_layers=1, indexer_types=["shared"])
    layer = glm5_2.Glm52MLAAttention(cfg, 0, _DTYPE, torch.device("npu"))
    torch.manual_seed(20260922)
    for projection in (layer.q_a_proj, layer.kv_a_proj_with_mqa, layer.q_b_proj, layer.o_proj):
        projection._set_dynamic_activation(False)
        projection.weight.data.random_(-4, 5)
        projection.deq_scale.fill_(0.002)
        projection.quant_bias.zero_()
        projection.input_scale.fill_(0.05)
        projection.input_offset.zero_()
    layer.q_a_layernorm.weight.data.fill_(1)
    layer.kv_a_layernorm.weight.data.fill_(1)
    layer.kv_b_proj.weight.data.normal_(std=0.01)
    layer.process_weights_after_loading()
    assert not layer._use_mlapo_v2
    assert layer._fused_mla_ready
    hidden = torch.randn(tokens, cfg.hidden_size, dtype=_DTYPE, device="npu")
    cos = torch.ones(tokens, 1, 1, 64, dtype=_DTYPE, device="npu") * 0.8
    sin = torch.ones_like(cos) * 0.6
    key = torch.full((1, 128, 1, 512), float("nan"), dtype=_DTYPE, device="npu")
    rope = torch.full((1, 128, 1, 64), float("nan"), dtype=_DTYPE, device="npu")
    slots = torch.arange(tokens, dtype=torch.int32, device="npu")

    def preprocess() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mla.deepseek_mla_preprocess_decode(
            hidden,
            layer.kv_a_proj_with_mqa.input_scale,
            layer.kv_a_proj_with_mqa.input_offset,
            layer._decode_qkv_weight,
            layer._decode_qkv_deq_scale,
            layer._decode_qkv_quant_bias,
            layer.q_a_layernorm.weight,
            layer.q_b_proj.input_scale,
            layer.q_b_proj.input_offset,
            layer.q_b_proj.weight,
            layer.q_b_proj.deq_scale,
            layer.q_b_proj.quant_bias,
            layer.W_UK,
            layer.kv_a_layernorm.weight,
            cos,
            sin,
            slots,
            key,
            rope,
            512,
            2048,
            4,
            192,
            64,
            cfg.rms_norm_eps,
            cfg.rms_norm_eps,
        )

    graph = None
    if graph_mode:
        preprocess()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = preprocess()
    for _ in range(2):
        hidden.normal_()
        q_c = layer.q_a_layernorm(layer.q_a_proj(hidden))
        q = layer.q_b_proj(q_c).view(tokens, 4, 256)
        q_nope, q_rope = q.split((192, 64), dim=-1)
        q_latent = attention_kernels.batch_matmul_transpose(q_nope, layer.W_UK)
        q_pe = rotary_embedding.interleaved_rotary_embedding(q_rope, cos, sin)
        kv, kr = layer.kv_a_proj_with_mqa(hidden).split((512, 64), dim=-1)
        expected_key = layer.kv_a_layernorm(kv)
        expected_rope = rotary_embedding.interleaved_rotary_embedding(kr.view(tokens, 1, 64), cos, sin)
        if graph is None:
            actual = preprocess()
        else:
            graph.replay()
        torch.npu.synchronize()
        for got, expected in zip(actual, (q_c, q_latent, q_pe)):
            torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(key.view(-1, 512)[:tokens], expected_key, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(rope.view(-1, 1, 64)[:tokens], expected_rope, rtol=2e-2, atol=2e-2)


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
