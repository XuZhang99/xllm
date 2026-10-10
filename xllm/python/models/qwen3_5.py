# Copyright 2025-2026 The xLLM Authors.
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

"""Qwen3.5 hybrid-attention causal LM for the Python executor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, GemmaRMSNorm, HiddenParallelEmbedding
from xllm.python.layers.qwen3_5.common import PartialRotaryEmbedding
from xllm.python.layers.qwen3_5.decoder_layer import Qwen3_5DecoderLayer, get_qwen3_5_decoder_layer_class
from xllm.python.model_executor.forward_context import ExecutionMetadataBuilder, record_layer_event
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    gqa_head_split,
    load_causal_lm_weights,
)
from xllm.python.models.base import PyModelBase
from xllm.python.models.model_config import ModelContext


class Qwen35Context(ModelContext):
    """Execution view over the Transformers architecture config."""

    config_module = "qwen3_5"
    config_name = "Qwen3_5TextConfig"

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0 or self.n_layers <= 0:
            raise ValueError("invalid Qwen3.5 model dimensions")
        if min(self.tp_size, self.dp_size, self.moe_tp_size, self.ep_size) <= 0:
            raise ValueError("parallel sizes must be positive")
        if self.tp_size * self.dp_size != self.world_size:
            raise ValueError("world_size must equal tp_size * dp_size")
        if self.moe_tp_size * self.ep_size != self.world_size:
            raise ValueError("world_size must equal moe_tp_size * ep_size")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if not 0 <= self.moe_tp_rank < self.moe_tp_size:
            raise ValueError("moe_tp_rank must be in [0, moe_tp_size)")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("ep_rank must be in [0, ep_size)")
        for name, count in (
            ("hidden size", self.hidden_size),
            ("attention heads", self.n_heads),
            ("linear key heads", self.linear_num_key_heads),
            ("linear value heads", self.linear_num_value_heads),
            ("dense intermediate size", self.intermediate_size),
            ("vocabulary size", self.vocab_size),
        ):
            if count % self.tp_size:
                raise ValueError(f"{name} must be divisible by tp_size")
        if self.n_kv_heads >= self.tp_size:
            if self.n_kv_heads % self.tp_size:
                raise ValueError("KV heads must be divisible by tp_size when KV heads are sharded")
        elif self.tp_size % self.n_kv_heads:
            raise ValueError("tp_size must be divisible by KV heads when KV heads are replicated")
        if self.decoder_sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be positive")
        if self.num_experts:
            if self.num_experts_per_tok <= 0:
                raise ValueError("num_experts_per_tok must be positive for MoE")
            if self.num_experts % self.ep_size:
                raise ValueError("num_experts must be divisible by ep_size")
            if self.moe_intermediate_size <= 0:
                raise ValueError("moe_intermediate_size must be positive for MoE")
            if self.moe_intermediate_size % self.moe_tp_size:
                raise ValueError("moe_intermediate_size must be divisible by moe_tp_size")
            if self.shared_expert_intermediate_size <= 0:
                raise ValueError("shared_expert_intermediate_size must be positive for Qwen3.5 MoE")
            if self.shared_expert_intermediate_size % self.tp_size:
                raise ValueError("shared_expert_intermediate_size must be divisible by tp_size")
        if self.enable_mega_moe:
            if self.num_experts <= 0 or self.tp_size <= 1 or self.dp_size <= 1:
                raise ValueError("MegaMoe token ownership requires MoE, TP > 1 and DP > 1")
            if self.moe_tp_size != 1 or self.ep_size != self.world_size:
                raise ValueError("MegaMoe token ownership requires MoE-TP = 1 and EP = world_size")
            if self.mega_moe_context is None:
                raise ValueError("MegaMoe token ownership requires a communication context")
            if self.mega_moe_ccl_buffer_size <= 0 or self.mega_moe_num_max_tokens_per_rank <= 0:
                raise ValueError("MegaMoe token ownership requires positive communication capacities")

    def is_moe_layer(self, layer_id: int) -> bool:
        return (
            self.num_experts > 0
            and (layer_id + 1) % self.decoder_sparse_step == 0
            and layer_id not in self.mlp_only_layers
        )

    def head_split(self) -> tuple[int, int]:
        return gqa_head_split(self.n_heads, self.n_kv_heads, self.tp_size)


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg: Qwen35Context, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        if device.type in ("npu", "privateuseone"):
            builders: list[ExecutionMetadataBuilder] = []
            if "linear_attention" in cfg.layer_types:
                from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder

                if dtype != torch.bfloat16:
                    raise NotImplementedError("Qwen3.5 MegaGdn supports BF16 model weights only")
                builders.append(Qwen3_5GdnMetadataBuilder(cfg))
            if cfg.enable_mega_moe:
                from xllm.python.layers.npu.mega_moe_metadata_builder import TokenOwnerMegaMoeMetadataBuilder

                builders.append(TokenOwnerMegaMoeMetadataBuilder(cfg))
            if builders:
                self.execution_metadata_builders = tuple(builders)
        if cfg.hidden_size % cfg.tp_size:
            raise ValueError("hidden_size must be divisible by tp_size")
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // cfg.tp_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        self.rotary = PartialRotaryEmbedding(
            cfg.head_dim,
            rotary_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype,
            device,
        )
        decoder_layer_cls = get_qwen3_5_decoder_layer_class(device)
        self.layers = cast(
            Sequence[Qwen3_5DecoderLayer],
            nn.ModuleList(decoder_layer_cls(cfg, i, dtype, device, self.rotary) for i in range(cfg.n_layers)),
        )
        self.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, positions)
            record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        return hidden


class Qwen3_5ForCausalLM(PyModelBase):
    model: Qwen3_5Model
    lm_head: ColumnParallelLinear

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Qwen35Context.from_dict(config)
        self.cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device
        self.model = Qwen3_5Model(self.cfg, dtype, device)  # pyright: ignore[reportIncompatibleVariableOverride]
        self.lm_head = ColumnParallelLinear(  # pyright: ignore[reportIncompatibleVariableOverride]
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        all_weights = ScopedWeightLoader(
            state_dicts,
            src_prefixes=("model.language_model.", "model.", ""),
        )
        context = ParallelLoadContext.from_config(self.cfg, tp_rank, tp_size)
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            all_weights,
            context,
            tie_word_embeddings=self.cfg.tie_word_embeddings,
            embed_fallback=True,
        )
