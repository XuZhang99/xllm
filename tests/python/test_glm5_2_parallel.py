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

"""Parallel-layout tests for the GLM-5.2 Python NPU model."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.models import glm5_2
from xllm.python.models.glm5_2 import Glm52Config, Glm52ForCausalLM


def _config(**overrides) -> dict:
    values = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "first_k_dense_replace": 0,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "tp_size": 2,
        "tp_rank": 0,
        "dp_size": 2,
        "dp_rank": 0,
        "cp_size": 1,
        "cp_rank": 0,
        "world_size": 4,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "ep_size": 4,
        "ep_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


def test_full_world_ep_partitions_glm_experts() -> None:
    cfg = Glm52Config.from_dict(_config(ep_rank=3))
    cfg.validate()

    model = Glm52ForCausalLM(_config(ep_rank=3))
    moe = model.model.layers[0].mlp

    assert moe.local_expert_start == 6
    assert moe.local_expert_end == 8
    # experts_w13 / experts_w2 are created empty and materialized lazily during
    # weight loading (peak-memory optimization); allocate them the same way the
    # loader does before checking the per-rank expert-parallel shapes.
    moe.allocate_experts_w13_for_loading()
    moe.allocate_experts_w2_for_loading()
    assert moe.experts_w13.shape == (2, 16, 16)
    assert moe.experts_w2.shape == (2, 16, 8)


def test_glm_parallel_world_size_defaults_to_tp_dp_product() -> None:
    values = _config()
    values.pop("world_size")

    cfg = Glm52Config.from_dict(values)

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 4


def test_glm_parallel_world_size_includes_context_parallel() -> None:
    cfg = Glm52Config.from_dict(_config(cp_size=2, cp_rank=1, world_size=8, ep_size=8))

    cfg.validate()

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 8
    assert cfg.cp_rank == 1


def test_glm_dsa_multi_stream_config_is_opt_in() -> None:
    assert not Glm52Config.from_dict(_config()).enable_dsa_multi_stream
    assert Glm52Config.from_dict(_config(enable_dsa_multi_stream=True)).enable_dsa_multi_stream


def test_glm_dsa_indexer_stream_forks_and_joins_before_attention(monkeypatch) -> None:
    cfg_values = _config(
        tp_size=1,
        dp_size=1,
        world_size=1,
        ep_size=1,
        num_attention_heads=2,
    )
    model = Glm52ForCausalLM(cfg_values)
    attention = model.model.layers[0].self_attn
    call_order: list[str] = []

    class _Projection(torch.nn.Module):
        def __init__(self, output: torch.Tensor, name: str) -> None:
            super().__init__()
            self._output = output
            self._name = name

        def forward(self, _input: torch.Tensor) -> torch.Tensor:
            call_order.append(self._name)
            return self._output

    class _Indexer(torch.nn.Module):
        def select_qli(self, *_args: object) -> torch.Tensor:
            call_order.append("indexer")
            return torch.zeros(2, 1, cfg_values["index_topk"], dtype=torch.int32)

    hidden = torch.zeros(2, cfg_values["hidden_size"])
    attention.q_a_proj = _Projection(torch.zeros(2, cfg_values["q_lora_rank"]), "q_a")
    attention.q_a_layernorm = torch.nn.Identity()
    attention.q_b_proj = _Projection(
        torch.zeros(
            2,
            cfg_values["num_attention_heads"]
            * (cfg_values["qk_nope_head_dim"] + cfg_values["qk_rope_head_dim"]),
        ),
        "q_b",
    )
    attention.kv_a_proj_with_mqa = _Projection(
        torch.zeros(2, cfg_values["kv_lora_rank"] + cfg_values["qk_rope_head_dim"]),
        "kv_a",
    )
    attention.kv_a_layernorm = torch.nn.Identity()
    attention.o_proj = torch.nn.Identity()
    attention.indexer = _Indexer()

    indexer_stream = MagicMock()
    main_stream = MagicMock()
    main_stream.wait_stream.side_effect = lambda _stream: call_order.append("join")
    attention._indexer_stream = indexer_stream
    fake_npu = SimpleNamespace(
        current_stream=MagicMock(return_value=main_stream),
        stream=MagicMock(return_value=nullcontext()),
    )

    backend = MagicMock()
    backend.mla_index_context.return_value = object()

    def _execute_mla(*_args, **_kwargs) -> torch.Tensor:
        call_order.append("attention")
        return torch.zeros(2, cfg_values["num_attention_heads"], cfg_values["kv_lora_rank"])

    backend.execute_mla.side_effect = _execute_mla
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(
        glm5_2,
        "get_forward_context",
        lambda: SimpleNamespace(attention_backend=backend),
    )
    monkeypatch.setattr(
        glm5_2,
        "_gather_interleave_cos_sin",
        lambda cache, _positions: (cache, cache),
    )
    monkeypatch.setattr(glm5_2, "_interleave_rope_with", lambda tensor, _cos, _sin: tensor)

    attention(hidden, torch.arange(2), model.model.rotary.cos_sin_cache)

    indexer_stream.wait_stream.assert_called_once_with(main_stream)
    main_stream.wait_stream.assert_called_once_with(indexer_stream)
    assert call_order.index("indexer") < call_order.index("q_b")
    assert call_order.index("q_b") < call_order.index("join")
    assert call_order.index("kv_a") < call_order.index("join")
    assert call_order.index("join") < call_order.index("attention")


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an available NPU",
)
def test_glm_dsa_multi_stream_matches_single_stream_on_npu(monkeypatch) -> None:
    cfg_values = _config(
        tp_size=1,
        dp_size=1,
        world_size=1,
        ep_size=1,
        num_attention_heads=2,
    )
    model = Glm52ForCausalLM(cfg_values)
    attention = model.model.layers[0].self_attn
    device = torch.device("npu")

    class _NpuIndexer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "weight",
                torch.randn(cfg_values["q_lora_rank"], cfg_values["q_lora_rank"], device=device),
            )

        def select_qli(
            self,
            _hidden: torch.Tensor,
            q_c: torch.Tensor,
            *_args: object,
        ) -> torch.Tensor:
            scores = torch.matmul(q_c, self.weight)
            return torch.topk(scores, cfg_values["index_topk"], dim=-1).indices.to(torch.int32).unsqueeze(1)

    class _Backend:
        @staticmethod
        def mla_index_context(_attention) -> object:
            return object()

        @staticmethod
        def execute_mla(
            q_latent: torch.Tensor,
            _q_pe: torch.Tensor,
            k_latent: torch.Tensor,
            _k_pe: torch.Tensor,
            _attention,
            *,
            topk: torch.Tensor,
        ) -> torch.Tensor:
            topk_sum = topk.to(q_latent.dtype).sum(dim=-1, keepdim=True)
            return q_latent + k_latent + topk_sum

    torch.manual_seed(0)
    attention.q_a_proj = torch.nn.Linear(
        cfg_values["hidden_size"], cfg_values["q_lora_rank"], bias=False, device=device
    )
    attention.q_a_layernorm = torch.nn.Identity()
    attention.q_b_proj = torch.nn.Linear(
        cfg_values["q_lora_rank"],
        cfg_values["num_attention_heads"]
        * (cfg_values["qk_nope_head_dim"] + cfg_values["qk_rope_head_dim"]),
        bias=False,
        device=device,
    )
    attention.kv_a_proj_with_mqa = torch.nn.Linear(
        cfg_values["hidden_size"],
        cfg_values["kv_lora_rank"] + cfg_values["qk_rope_head_dim"],
        bias=False,
        device=device,
    )
    attention.kv_a_layernorm = torch.nn.Identity()
    attention.o_proj = torch.nn.Linear(
        cfg_values["num_attention_heads"] * cfg_values["v_head_dim"],
        cfg_values["hidden_size"],
        bias=False,
        device=device,
    )
    attention.indexer = _NpuIndexer()
    attention.W_UK = torch.randn(
        cfg_values["num_attention_heads"],
        cfg_values["qk_nope_head_dim"],
        cfg_values["kv_lora_rank"],
        device=device,
    )
    attention.W_UV = torch.randn(
        cfg_values["num_attention_heads"],
        cfg_values["kv_lora_rank"],
        cfg_values["v_head_dim"],
        device=device,
    )

    monkeypatch.setattr(
        glm5_2,
        "get_forward_context",
        lambda: SimpleNamespace(attention_backend=_Backend()),
    )
    monkeypatch.setattr(
        glm5_2,
        "_gather_interleave_cos_sin",
        lambda cache, _positions: (cache, cache),
    )
    monkeypatch.setattr(glm5_2, "_interleave_rope_with", lambda tensor, _cos, _sin: tensor)

    hidden = torch.randn(2, cfg_values["hidden_size"], device=device)
    positions = torch.arange(2, device=device)
    cos_sin_cache = torch.zeros(2, cfg_values["qk_rope_head_dim"], device=device)

    attention._indexer_stream = None
    expected_output, expected_topk = attention(hidden, positions, cos_sin_cache)
    torch.npu.synchronize()

    attention._indexer_stream = torch.npu.Stream(device=device)
    actual_output, actual_topk = attention(hidden, positions, cos_sin_cache)
    torch.npu.synchronize()

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_topk, expected_topk)

    capture_stream = torch.npu.Stream(device=device)
    capture_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(capture_stream):
        for _ in range(2):
            attention(hidden, positions, cos_sin_cache)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=capture_stream):
        graph_output, graph_topk = attention(hidden, positions, cos_sin_cache)
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(graph_output, expected_output)
    torch.testing.assert_close(graph_topk, expected_topk)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ep_size": 2}, "ep_size must be 1 or world_size"),
        ({"world_size": 8}, r"world_size must equal tp_size \* dp_size \* cp_size"),
        ({"cp_rank": 2}, "cp_rank must be in"),
        ({"n_routed_experts": 10}, "n_routed_experts must be divisible by ep_size"),
        ({"moe_tp_size": 2}, r"moe_tp_size \* ep_size"),
        ({"ep_rank": 4}, "ep_rank must be in"),
    ],
)
def test_invalid_glm_parallel_topology_is_rejected(overrides: dict, message: str) -> None:
    cfg = Glm52Config.from_dict(_config(**overrides))

    with pytest.raises(ValueError, match=message):
        cfg.validate()


class _RecordingLoader:
    latest: _RecordingLoader | None = None

    def __init__(self, _model, _state_dicts, tp_size: int, tp_rank: int) -> None:
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.loaded: list[str] = []
        self.shared_shards: list[tuple[str, int, int]] = []
        type(self).latest = self

    def load_tensor(self, name: str) -> torch.Tensor:
        self.loaded.append(name)
        if ".mlp.experts." not in name:
            return torch.zeros(32, 32)
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            return torch.zeros(8, 16, dtype=torch.int8)
        if name.endswith(("gate_proj.weight_scale", "up_proj.weight_scale")):
            return torch.zeros(8, 1)
        if name.endswith(("gate_proj.weight_offset", "up_proj.weight_offset")):
            return torch.zeros(8, 1)
        if name.endswith("down_proj.weight"):
            return torch.zeros(16, 8, dtype=torch.int8)
        if name.endswith(("down_proj.weight_scale", "down_proj.weight_offset")):
            return torch.zeros(16, 1)
        raise AssertionError(f"unexpected expert tensor: {name}")

    def shard(
        self,
        tensor: torch.Tensor,
        dim: int,
        world: int | None = None,
        rank: int | None = None,
    ) -> torch.Tensor:
        world = self.tp_size if world is None else world
        rank = self.tp_rank if rank is None else rank
        if world <= 1:
            return tensor
        size = tensor.size(dim) // world
        return tensor.narrow(dim, rank * size, size).contiguous()

    def copy_in(self, name: str, tensor: torch.Tensor) -> None:
        self.loaded.append(name)
        assert tensor.is_contiguous()

    def load_w8a8_a(self, prefix: str, proj: str, _shard_dims: dict | None = None) -> None:
        self.loaded.append(prefix + proj)

    def load_w8a8_b(self, prefix: str) -> None:
        self.loaded.append(prefix)
        if ".shared_experts." in prefix:
            self.shared_shards.append((prefix, self.tp_size, self.tp_rank))


def test_glm_weight_loader_reads_only_local_ep_experts(monkeypatch) -> None:
    model = Glm52ForCausalLM(_config(ep_rank=2))
    model.model.layers[0].self_attn.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    loader = _RecordingLoader.latest
    assert loader is not None
    expert_names = [name for name in loader.loaded if ".mlp.experts." in name]
    assert expert_names
    assert all(".experts.4." in name or ".experts.5." in name for name in expert_names)
    assert loader.tp_size == 2
    assert loader.tp_rank == 0
    assert loader.shared_shards == [("model.layers.0.mlp.shared_experts.", 1, 0)]
