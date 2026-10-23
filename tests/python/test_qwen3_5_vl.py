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

"""Multimodal composition and position contracts for the Qwen3.5 family."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.python.qwen3_5_test_utils import StateDict
from xllm.python.layers.qwen3_5.common import PartialRotaryEmbedding
from xllm.python.model_loader import ScopedWeightLoader
from xllm.python.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5Model
from xllm.python.models.qwen3_5_vl import Qwen3_5ForConditionalGeneration
from xllm.python.models.qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionTransformer
from xllm.python.platform import current_platform
from xllm.python.registry import get_model_class


class _IdentityNorm(nn.Module):
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, None]:
        return hidden, None


def _model() -> Qwen3_5ForConditionalGeneration:
    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(hidden_size=4)
    model.image_token_id = 10
    model.video_token_id = 11
    decoder = Qwen3_5Model.__new__(Qwen3_5Model)
    nn.Module.__init__(decoder)
    decoder.embed_tokens = nn.Embedding.from_pretrained(torch.arange(64).view(16, 4).float())
    decoder.layers = nn.ModuleList()
    decoder.norm = _IdentityNorm()
    decoder._inputs_embeds = None
    model.model = decoder
    return model


@pytest.mark.parametrize("name", ["qwen3_5", "qwen3_5_moe", "Qwen3_5ForConditionalGeneration"])
def test_multimodal_registry_selects_vision_composer(name: str) -> None:
    if current_platform.device_type() != "npu":
        pytest.skip("Qwen3.5 vision tower currently supports NPU only")
    assert get_model_class(name) is Qwen3_5ForConditionalGeneration
    assert get_model_class("qwen3_5_text") is Qwen3_5ForCausalLM


def test_mixed_image_video_rows_are_consumed_once() -> None:
    model = _model()
    tokens = torch.tensor([1, 10, 10, 2, 11, 3])
    image = torch.full((2, 4), 100.0)
    video = torch.full((1, 4), 200.0)
    expected = model.model.embed_tokens(tokens).clone()
    expected[1:3] = image
    expected[4] = video[0]
    merged = model.get_input_embeddings(tokens, image, video)
    torch.testing.assert_close(merged, expected)
    torch.testing.assert_close(model.model(tokens, torch.zeros(3, 6, dtype=torch.long)), expected)
    assert model.model._inputs_embeds is None
    decode_tokens = torch.tensor([4, 5])
    torch.testing.assert_close(
        model.model(decode_tokens, torch.tensor([6, 7])), model.model.embed_tokens(decode_tokens)
    )


def test_chunk_uses_only_current_visual_rows() -> None:
    model = _model()
    for values in (torch.full((2, 4), 100.0), torch.full((1, 4), 200.0)):
        tokens = torch.tensor([1] + [10] * values.shape[0])
        merged = model.get_input_embeddings(tokens, values)
        torch.testing.assert_close(merged[1:], values)
        model.model(tokens, torch.zeros(3, len(tokens), dtype=torch.long))
    text = torch.tensor([1, 2])
    torch.testing.assert_close(model.get_input_embeddings(text), model.model.embed_tokens(text))
    assert model.model._inputs_embeds is None


def test_empty_visual_rows_keep_text_only_execution_state() -> None:
    model = _model()
    tokens = torch.tensor([1, 2])
    merged = model.get_input_embeddings(tokens, torch.empty(0, 4), torch.empty(0, 4))
    torch.testing.assert_close(merged, model.model.embed_tokens(tokens))
    assert model.model._inputs_embeds is None


@pytest.mark.parametrize("shape", [(1, 4), (3, 4), (2, 8)])
def test_visual_shape_mismatch_is_rejected_before_storing_state(shape: tuple[int, int]) -> None:
    model = _model()
    with pytest.raises(ValueError, match="visual embedding shape"):
        model.get_input_embeddings(torch.tensor([10, 10]), torch.zeros(shape))
    assert model.model._inputs_embeds is None


def test_qwen35_rejects_deepstack_before_constructing_decoder() -> None:
    with pytest.raises(ValueError, match="DeepStack"):
        Qwen3_5ForConditionalGeneration({"mm_deepstack_visual_indexes": [1]})


@pytest.mark.parametrize("deepstack_indexes", [[], [0]])
def test_shared_vision_weight_loading(deepstack_indexes: list[int]) -> None:
    cfg = Qwen3VLVisionConfig(
        deepstack_visual_indexes=deepstack_indexes,
        depth=1,
        hidden_size=16,
        num_heads=2,
        patch_size=2,
        intermediate_size=32,
        out_hidden_size=4,
        num_position_embeddings=16,
    )
    vision = Qwen3VLVisionTransformer(cfg, dtype=torch.float32, device=torch.device("cpu"))
    checkpoint = {}
    for index, (name, parameter) in enumerate(vision.named_parameters()):
        key = name.replace("patch_embed.", "patch_embed.proj.")
        checkpoint[f"model.visual.{key}"] = torch.full_like(parameter, index + 1)
    vision.load_weights(ScopedWeightLoader([StateDict(checkpoint)], src_prefixes=("model.visual.",)))
    for index, parameter in enumerate(vision.parameters()):
        torch.testing.assert_close(parameter, torch.full_like(parameter, index + 1))


@pytest.mark.parametrize("interleaved,axes", [(True, [0, 1, 2, 0, 1, 2]), (False, [0, 0, 1, 1, 2, 2])])
def test_partial_mrope_matches_axis_frequency_reference(interleaved: bool, axes: list[int]) -> None:
    rotary = PartialRotaryEmbedding(
        16,
        12,
        32,
        10000.0,
        torch.float32,
        torch.device("cpu"),
        mrope_section=[2, 2, 2],
        mrope_interleaved=interleaved,
    )
    positions = torch.tensor([[1, 2], [3, 4], [5, 6]])
    values = torch.arange(64, dtype=torch.float32).view(2, 2, 16) / 32
    frequencies = 10000.0 ** (-torch.arange(6, dtype=torch.float32) / 6)
    angles = positions[axes].T * frequencies
    cos = angles.cos().repeat(1, 2).unsqueeze(1)
    sin = angles.sin().repeat(1, 2).unsqueeze(1)
    rotated_half = torch.cat((-values[..., 6:12], values[..., :6]), dim=-1)
    expected = torch.cat((values[..., :12] * cos + rotated_half * sin, values[..., 12:]), dim=-1)
    torch.testing.assert_close(rotary(positions, values), expected)
    decode_positions = torch.tensor([7, 8])
    torch.testing.assert_close(rotary(decode_positions, values), rotary(decode_positions.expand(3, -1), values))
