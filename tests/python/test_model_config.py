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

"""Architecture ownership and the native-to-Transformers configuration boundary."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
import transformers
from huggingface_hub.errors import StrictDataclassClassValidationError
from transformers import (
    DeepseekV4Config,
    DeepseekV32Config,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
    GlmMoeDsaConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from xllm.python.models.deepseek_v4 import DeepseekV4Context
from xllm.python.models.deepseek_v32 import DeepseekV32Context
from xllm.python.models.glm5_2 import Glm52Context
from xllm.python.models.glm5_next import Glm5NextContext
from xllm.python.models.glm5_next_vl import Glm5NextVisionContext, Glm5NextVisionModel
from xllm.python.models.model_config import ModelContext, ModelRuntime
from xllm.python.models.qwen3 import Qwen3Context
from xllm.python.models.qwen3_5 import Qwen35Context
from xllm.python.models.qwen3_dflash import DFlashQwen3Context
from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3Context
from xllm.python.models.qwen3_dspark import Qwen3DSparkContext
from xllm.python.models.qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLTextContext,
    Qwen3VLVisionContext,
    Qwen3VLVisionTransformer,
)


@pytest.mark.parametrize(
    ("context_class", "config_class", "values"),
    [
        (Qwen3Context, Qwen3Config, {}),
        (Qwen35Context, Qwen3_5TextConfig, {}),
        (Qwen35Context, Qwen3_5MoeTextConfig, {"model_type": "qwen3_5_moe"}),
        (Qwen3VLTextContext, Qwen3VLTextConfig, {}),
        (Qwen3VLVisionContext, Qwen3VLVisionConfig, {}),
        (DeepseekV32Context, DeepseekV32Config, {}),
        (DeepseekV4Context, DeepseekV4Config, {}),
        (Glm52Context, GlmMoeDsaConfig, {}),
        (Glm5NextContext, Glm5NextTextConfig, {}),
        (Glm5NextVisionContext, Glm5NextVisionConfig, {}),
        (DFlashQwen3Context, Qwen3Config, {}),
        (DFlash2Qwen3Context, Qwen3Config, {}),
        (Qwen3DSparkContext, Qwen3Config, {}),
    ],
)
def test_architecture_and_defaults_come_from_transformers(
    context_class: type[ModelContext], config_class: type, values: dict[str, Any]
) -> None:
    context = context_class.from_dict(values)
    assert type(context.hf_config) is config_class
    assert context.hf_config.to_dict() == config_class().to_dict()


def test_reflected_fields_override_nested_text_without_mutating_input() -> None:
    source = {"text_config": {"num_hidden_layers": 4, "hidden_size": 128}, "n_layers": 2, "n_heads": 8, "n_kv_heads": 2}
    context = Qwen3Context.from_dict(source)
    assert context.n_layers == context.hf_config.num_hidden_layers == 2
    assert context.n_heads == context.hf_config.num_attention_heads == 8
    assert context.n_kv_heads == context.hf_config.num_key_value_heads == 2
    assert context.hidden_size == 128
    assert source["text_config"]["num_hidden_layers"] == 4
    assert "n_layers" not in context.hf_config.to_dict()


@pytest.mark.parametrize("context_class", [DeepseekV32Context, Glm52Context])
def test_unrelated_model_args_defaults_do_not_override_mla(context_class: type[ModelContext]) -> None:
    context = context_class.from_dict(
        {
            "n_layers": 2,
            "n_heads": 8,
            "qk_rope_head_dim": 16,
            "num_experts_per_tok": 2,
            "rope_scaling": -1,
            "rope_head_dim": 0,
            "n_activated_experts": 0,
            "window_size": 0,
            "partial_rotary_factor": 0.0,
            "index_topk_pattern": "",
            "full_attention_interval": 0,
        }
    )
    assert context.qk_rope_head_dim == 16
    assert context.num_experts_per_tok == 2
    assert context.hf_config.head_dim == 16


def test_mtp_registration_does_not_override_hf_model_type() -> None:
    context = Glm52Context.from_dict({"model_type": "glm_moe_dsa_mtp"})
    assert context.model_type == "glm_moe_dsa_mtp"
    assert context.hf_config.model_type == "glm_moe_dsa"


def test_runtime_tensor_and_parallelism_stay_out_of_hf_serialization() -> None:
    context = Qwen35Context.from_dict(
        {
            "num_experts": 8,
            "tp_size": 2,
            "dp_size": 2,
            "ep_size": 4,
            "mega_moe_context": torch.ones(1),
            "layers_to_capture": [1, 2],
        }
    )
    serialized = context.hf_config.to_dict()
    json.dumps(serialized)
    for key in ("tp_size", "dp_size", "ep_size", "world_size", "mega_moe_context", "layers_to_capture"):
        assert key not in serialized
    assert context.world_size == 4
    assert context.moe_tp_size == 1
    assert context.layers_to_capture == (1, 2)
    other = context.with_runtime(tp_rank=1)
    assert other.hf_config is context.hf_config
    assert context.tp_rank == 0 and other.tp_rank == 1
    assert other.runtime.mega_moe_context is context.runtime.mega_moe_context


def test_existing_hf_config_keeps_identity_and_runtime_is_separate() -> None:
    architecture = Qwen3Config(num_hidden_layers=2)
    runtime = ModelRuntime(tp_size=2, world_size=2)
    context = Qwen3Context(hf_config=architecture, runtime=runtime)
    assert context.hf_config is architecture and context.runtime is runtime
    context.tp_rank = 1
    assert runtime.tp_rank == 1
    assert not hasattr(architecture, "tp_rank")


def test_missing_model_config_reports_installed_version_without_blocking_other_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transformers, "Glm5NextTextConfig", None, raising=False)
    with pytest.raises(ImportError, match="Glm5NextTextConfig") as error:
        Glm5NextContext.from_dict({})
    assert transformers.__version__ in str(error.value)
    assert isinstance(Qwen3Context.from_dict({}).hf_config, Qwen3Config)


@pytest.mark.parametrize("num_experts", [0, 8])
def test_qwen35_selects_dense_or_moe_and_uses_upstream_layer_schedule(num_experts: int) -> None:
    context = Qwen35Context.from_dict({"n_layers": 5, "num_experts": num_experts, "full_attention_interval": 3})
    expected = Qwen3_5MoeTextConfig if num_experts else Qwen3_5TextConfig
    assert type(context.hf_config) is expected
    assert context.layer_types == [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
    ]
    assert context.partial_rotary_factor == 0.25


def test_qwen35_rejects_wrong_layer_schedule() -> None:
    with pytest.raises(StrictDataclassClassValidationError, match="number of `layer_types`"):
        Qwen35Context.from_dict({"n_layers": 2, "layer_types": ["full_attention"]})


def test_glm_next_maps_kda_and_legacy_attention_schedule() -> None:
    context = Glm5NextContext.from_dict(
        {
            "n_layers": 4,
            "first_k_dense_replace": 1,
            "linear_attn_config": {
                "num_heads": 8,
                "head_dim": 16,
                "short_conv_kernel_size": 3,
                "full_attn_layers": [1, 3],
                "safe_gate": True,
                "gate_lower_bound": None,
            },
        }
    )
    assert context.kda_num_heads == context.hf_config.linear_num_heads == 8
    assert context.kda_head_dim == context.hf_config.linear_head_dim == 16
    assert context.short_conv_kernel_size == 3
    assert context.linear_lower_bound == -5.0
    assert context.layer_types == ["linear_attention", "indexed_attention", "linear_attention", "indexed_attention"]
    assert context.is_dsa(1) and not context.is_dsa(0)
    assert context.mlp_layer_types == ["dense", "sparse", "sparse", "sparse"]


@pytest.mark.parametrize("context_class", [DeepseekV32Context, Glm52Context])
def test_mla_head_dim_keeps_transformers_rope_semantics(context_class: type[ModelContext]) -> None:
    context = context_class.from_dict({"qk_nope_head_dim": 32, "qk_rope_head_dim": 16, "head_dim": 999})
    assert context.hf_config.head_dim == 16
    assert context.hf_config.qk_head_dim == 48


def test_indexer_schedule_and_rope_normalization() -> None:
    context = Glm52Context.from_dict(
        {
            "n_layers": 4,
            "first_k_dense_replace": 1,
            "index_topk_pattern": "FSFS",
            "rope_theta": 1e6,
            "rope_scaling_factor": 4.0,
            "rope_scaling_original_max_position_embeddings": 8192,
            "rope_scaling_beta_fast": 0.0,
            "rope_scaling_beta_slow": -1.0,
        }
    )
    assert context.indexer_types == ["full", "shared", "full", "shared"]
    assert context.mlp_layer_types == ["dense", "sparse", "sparse", "sparse"]
    assert context.rope_scaling_factor == 4.0
    assert context.original_max_position_embeddings == 8192
    assert context.rope_beta_fast == 32 and context.rope_beta_slow == 1


def test_vision_does_not_inherit_flat_text_dimensions_or_bias() -> None:
    context = Glm5NextVisionContext.from_dict(
        {
            "hidden_size": 4096,
            "attention_bias": False,
            "mm_hidden_size": 128,
            "mm_num_attention_heads": 4,
            "mm_swiglu_limit": 0.0,
        }
    )
    assert context.hidden_size == 128
    assert context.attention_bias is True
    assert context.swiglu_limit == 10.0
    canonical = Glm5NextVisionContext.from_dict(
        {"vision_config": {"hidden_size": 64, "attention_bias": False}, "hidden_size": 4096}
    )
    assert canonical.hidden_size == 64 and canonical.attention_bias is False


def test_qwen_vl_reads_nested_mrope() -> None:
    context = Qwen3VLTextContext.from_dict(
        {"text_config": {"rope_parameters": {"rope_type": "default", "rope_theta": 1e6, "mrope_section": [24, 20, 20]}}}
    )
    assert context.mrope_section == [24, 20, 20]
    assert context.rope_theta == 1e6


def test_qwen_vl_text_builds_with_its_own_transformers_config() -> None:
    context = Qwen3VLTextContext.from_dict(
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 32,
            "vocab_size": 32,
            "max_position_embeddings": 16,
        }
    )
    model = Qwen3VLModel(context, torch.float32, torch.device("cpu"))
    assert len(model.layers) == 1
    assert model.layers[0].self_attn.head_dim == 8
    assert context.sliding_window == 0


@pytest.mark.parametrize(
    ("context_class", "model_class"),
    [
        (Qwen3VLVisionContext, Qwen3VLVisionTransformer),
        (Glm5NextVisionContext, Glm5NextVisionModel),
    ],
)
def test_vision_towers_build_from_nested_transformers_dimensions(
    context_class: type[ModelContext], model_class: type[torch.nn.Module]
) -> None:
    context = context_class.from_dict(
        {
            "hidden_size": 4096,
            "vision_config": {
                "depth": 1,
                "hidden_size": 16,
                "num_heads": 2,
                "intermediate_size": 32,
                "out_hidden_size": 16,
                "projection_intermediate_size": 16,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "spatial_merge_size": 2,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [0],
            },
        }
    )
    model = model_class(context, torch.float32, torch.device("cpu"))
    assert len(model.blocks) == 1
    assert model.patch_size == 2
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_deepseek_v4_legacy_fields_match_canonical_layer_types() -> None:
    context = DeepseekV4Context.from_dict(
        {
            "n_layers": 4,
            "compress_ratios": [0, 4, 128],
            "n_hash_layers": 1,
            "n_activated_experts": 3,
            "window_size": 128,
            "factor": 16.0,
            "rope_scaling_original_max_position_embeddings": 65536,
            "rope_scaling_attn_factor": 1.0,
        }
    )
    assert context.compress_ratios == [1, 4, 128, 1]
    assert context.hf_config.layer_types == [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "sliding_attention",
    ]
    assert context.hf_config.mlp_layer_types == ["hash_moe", "moe", "moe", "moe"]
    assert context.n_hash_layers == 1
    assert context.num_experts_per_tok == 3
    assert context.rope_scaling_factor == 16.0
    assert context.rope_mscale == 1.0
    assert context.hf_config.rope_parameters["compress"]["factor"] == 16.0


def test_draft_extensions_do_not_change_backbone_serialization() -> None:
    context = DFlash2Qwen3Context.from_dict(
        {
            "vocab_size": 128,
            "sliding_window": 8,
            "use_sliding_window": False,
            "dflash2_block_size": None,
            "dflash_config": {"block_size": 4, "selector_rank": 16},
        }
    )
    assert context.block_size == 4 and context.selector_rank == 16
    assert context.draft_vocab_size == 128
    assert context.hf_config.use_sliding_window is True
    assert context.sliding_window == 8
    serialized = context.hf_config.to_dict()
    assert "block_size" not in serialized and "draft_vocab_size" not in serialized


def test_deepseek_v4_nested_rope_parameters_drive_the_execution_view() -> None:
    context = DeepseekV4Context.from_dict(
        {
            "rope_parameters": {
                "main": {"rope_type": "default", "rope_theta": 20000.0, "partial_rotary_factor": 0.125},
                "compress": {
                    "rope_type": "yarn",
                    "rope_theta": 320000.0,
                    "partial_rotary_factor": 0.125,
                    "factor": 8.0,
                    "original_max_position_embeddings": 8192,
                    "attention_factor": 1.0,
                },
            }
        }
    )
    assert context.rope_theta == 20000.0
    assert context.compress_rope_theta == 320000.0
    assert context.rope_scaling_factor == 8.0
    assert context.original_max_position_embeddings == 8192
