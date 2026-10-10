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

"""Transformers architecture configs and the xLLM execution context.

The native executor currently sends a flat ModelArgs dictionary. Normalize that
transport at this boundary; never put ranks, communication tensors or execution
switches into a Transformers config. Layers can keep their short field names
through this view while ``hf_config`` remains the sole architecture owner.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import Self

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedConfig


@dataclass
class ModelRuntime:
    """Execution parameters, deliberately excluded from HF serialization."""

    # Executor registrations may add a suffix such as _mtp to the HF model type.
    model_type: str = ""
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    world_size: int = 1
    layerwise_split_size: int = 1
    layerwise_split_rank: int = 0
    layers_to_capture: tuple[int, ...] = ()
    indexer_rope_interleave: bool = True
    enable_dsa_multi_stream: bool = False
    index_share_for_mtp_iteration: bool = False
    enable_mlapo: bool = True
    enable_attn_dp_weight_sharding: bool = False
    enable_mega_moe: bool = False
    mega_moe_context: Tensor | None = None
    mega_moe_ccl_buffer_size: int = 0
    mega_moe_num_max_tokens_per_rank: int = 0
    num_speculative_tokens: int = 0


@dataclass
class DraftConfig:
    """Draft checkpoint fields with no upstream Transformers equivalent."""

    draft_vocab_size: int = 0
    block_size: int = 0
    conv_group_size: int = 0
    conv_kernel_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = False


_RUNTIME_FIELDS = frozenset(field.name for field in fields(ModelRuntime))
_RUNTIME_INTEGER_FIELDS = frozenset(field.name for field in fields(ModelRuntime) if type(field.default) is int)
_DRAFT_FIELDS = frozenset(field.name for field in fields(DraftConfig))
_ALIASES = {
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "n_kv_heads": "num_key_value_heads",
    "kda_num_heads": "linear_num_heads",
    "kda_head_dim": "linear_head_dim",
    "short_conv_kernel_size": "linear_conv_kernel_dim",
    "window_size": "sliding_window",
    "n_activated_experts": "num_experts_per_tok",
    "rope_head_dim": "qk_rope_head_dim",
}
_VISION_ALIASES = {
    "mm_num_hidden_layers": "depth",
    "mm_hidden_size": "hidden_size",
    "mm_hidden_act": "hidden_act",
    "mm_attention_bias": "attention_bias",
    "mm_dropout": "attention_dropout",
    "mm_num_attention_heads": "num_heads",
    "mm_num_channels": "in_channels",
    "mm_image_size": "image_size",
    "mm_patch_size": "patch_size",
    "mm_layer_norm_eps": "rms_norm_eps",
    "mm_spatial_merge_size": "spatial_merge_size",
    "mm_temporal_patch_size": "temporal_patch_size",
    "mm_projection_dim": "out_hidden_size",
    "mm_intermediate_size": "intermediate_size",
    "mm_initializer_range": "initializer_range",
    "mm_projection_intermediate_size": "projection_intermediate_size",
    "mm_swiglu_limit": "swiglu_limit",
    "mm_deepstack_visual_indexes": "deepstack_visual_indexes",
    "mm_num_position_embeddings": "num_position_embeddings",
}
_ROPE_ALIASES = {
    "rope_theta": "rope_theta",
    "partial_rotary_factor": "partial_rotary_factor",
    "original_max_position_embeddings": "original_max_position_embeddings",
    "rope_scaling_factor": "factor",
    "rope_beta_fast": "beta_fast",
    "rope_beta_slow": "beta_slow",
    "rope_mscale": "mscale",
    "rope_mscale_all_dim": "mscale_all_dim",
    "mrope_section": "mrope_section",
}


def _architecture_input(values: dict[str, Any], vision: bool) -> dict[str, Any]:
    if not vision:
        nested = values.get("text_config") or {}
        return {**nested, **values}
    nested = values.get("vision_config")
    if isinstance(nested, dict):
        result = dict(nested)
    elif any(key.startswith("mm_") for key in values):
        # Flat text fields must not leak into the vision tower (especially bias).
        result = {}
    else:
        result = dict(values)
    for old, new in _VISION_ALIASES.items():
        if values.get(old) not in (None, -1, ""):
            result[new] = values[old]
    if "in_chans" in result:
        result.setdefault("in_channels", result.pop("in_chans"))
    if "context_size" in result:
        result.setdefault("projection_intermediate_size", result.pop("context_size"))
    return result


def _normalize_rope(values: dict[str, Any], module: str) -> None:
    # ModelArgs.rope_scaling is an integer sentinel, not the HF dictionary.
    raw_rope = values.get("rope_parameters") or values.get("rope_scaling")
    rope = deepcopy(raw_rope) if isinstance(raw_rope, dict) else {}
    if "main" in rope or "compress" in rope:
        values["rope_parameters"] = rope
        return
    for old, new in _ROPE_ALIASES.items():
        value = values.get(old)
        if value is not None and value not in (0, -1, ""):
            rope.setdefault(new, value)
    for key in (
        "factor",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
        "original_max_position_embeddings",
        "mrope_section",
        "mrope_interleaved",
        "low_freq_factor",
        "high_freq_factor",
    ):
        value = values.get(f"rope_scaling_{key}")
        if value is not None and value not in (0, -1, ""):
            rope.setdefault(key, value)
    if module == "deepseek_v4":
        for key in ("factor", "beta_fast", "beta_slow"):
            if values.get(key):
                rope[key] = values[key]
        if values.get("rope_scaling_attn_factor"):
            rope["attention_factor"] = values["rope_scaling_attn_factor"]
        if "attn_factor" in rope:
            rope.setdefault("attention_factor", rope.pop("attn_factor"))
    if rope:
        rope_type = values.get("rope_scaling_rope_type") or ("yarn" if "factor" in rope else "default")
        rope.setdefault("rope_type", rope.pop("type", rope_type))
        values["rope_parameters"] = rope


def _normalize_architecture(values: dict[str, Any], module: str, name: str) -> dict[str, Any]:
    result = _architecture_input(values, "Vision" in name)
    for old, new in _ALIASES.items():
        if old in ("window_size", "n_activated_experts", "rope_head_dim"):
            continue
        if result.get(old) is not None:
            # Reflected ModelArgs values are the executor's effective overrides.
            result[new] = result[old]
    if "Vision" in name:
        if result.get("swiglu_limit") == 0:
            result.pop("swiglu_limit")
        return result

    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        if result.get(key) == []:
            result.pop(key)
    if not result.get("index_topk_pattern"):
        result.pop("index_topk_pattern", None)
    if "layer_types" in result and result["layer_types"] is not None:
        result["layer_types"] = [
            "indexed_attention" if kind == "deepseek_sparse_attention" else kind for kind in result["layer_types"]
        ]
    if module in ("deepseek_v32", "glm_moe_dsa", "glm5_next"):
        if "n_routed_experts" not in result:
            for key in ("num_local_experts", "num_experts"):
                if result.get(key) is not None:
                    result["n_routed_experts"] = result[key]
                    break
        # C++'s GQA default is unrelated to MLA's per-query head count.
        if "n_kv_heads" in values and module != "glm5_next":
            result.pop("num_key_value_heads", None)
    if module == "glm_moe_dsa" and str(values.get("model_type", "")).endswith("_mtp"):
        # Native MTP may carry the target model's schedules. Rebuild those for
        # the draft depth before upstream validation, as the executor expects.
        for key in ("indexer_types", "mlp_layer_types"):
            schedule = result.get(key)
            if schedule is not None and len(schedule) != result.get("num_hidden_layers"):
                result.pop(key)
    if module.startswith("qwen3_5") and "num_experts" not in result and result.get("n_routed_experts") is not None:
        result["num_experts"] = result["n_routed_experts"]
    if module == "qwen3":
        if result.get("sliding_window", 0) and result["sliding_window"] > 0:
            result.setdefault("use_sliding_window", True)
            result.setdefault("max_window_layers", 0)
        elif result.get("sliding_window") == 0:
            result["sliding_window"] = None
    if module == "glm5_next":
        for key in ("hc_mult", "hc_eps", "hc_sinkhorn_iters", "swiglu_limit"):
            if result.get(key) == 0:
                result.pop(key)
        for old, new in (("linear_num_key_heads", "linear_num_heads"), ("linear_key_head_dim", "linear_head_dim")):
            if result.get(old):
                result.setdefault(new, result[old])
        if result.get("linear_conv_kernel_dim") == 0:
            result.pop("linear_conv_kernel_dim")
    if module == "deepseek_v4":
        if result.get("window_size") not in (None, 0, -1):
            result["sliding_window"] = result["window_size"]
        if result.get("n_activated_experts") is not None:
            result["num_experts_per_tok"] = result["n_activated_experts"]
        if "qk_rope_head_dim" not in result and result.get("rope_head_dim"):
            result["qk_rope_head_dim"] = result["rope_head_dim"]
        if "n_hash_layers" in result:
            result["num_hash_layers"] = result["n_hash_layers"]
        if "compress_ratios" in result:
            ratios = [0 if ratio <= 1 else ratio for ratio in result["compress_ratios"]]
            n_layers = result.get("num_hidden_layers", len(ratios))
            result["compress_ratios"] = ratios + [0] * max(n_layers - len(ratios), 0)
        if result.get("sliding_window") in (0, -1):
            result.pop("sliding_window")
    _normalize_rope(result, module)
    if result.get("rope_theta") is None:
        result.pop("rope_theta", None)
    # Unused ModelArgs fields are reflected for every model.
    if result.get("partial_rotary_factor") == 0:
        result.pop("partial_rotary_factor")
    if "n_layers" in values and result.get("full_attention_interval") == 0:
        result.pop("full_attention_interval")
    return result


class ModelContext:
    """Layer-facing view of one HF config and independent execution state.

    Architecture defaults and validation come from the upstream class. Only
    fields required by xLLM but absent from that class live in ``options``;
    these are compatibility controls, not a second architecture schema.
    """

    config_module: ClassVar[str]
    config_name: ClassVar[str]
    draft_kind: ClassVar[str] = ""
    hf_config: PreTrainedConfig
    runtime: ModelRuntime
    draft: DraftConfig
    options: dict[str, Any]

    def __init__(
        self,
        *,
        hf_config: PreTrainedConfig | None = None,
        runtime: ModelRuntime | None = None,
        **values: Any,
    ) -> None:
        runtime_values = {key: value for key, value in values.items() if key in _RUNTIME_FIELDS and value is not None}
        for key in _RUNTIME_INTEGER_FIELDS & runtime_values.keys():
            runtime_values[key] = int(runtime_values[key])
        runtime_values.setdefault(
            "world_size",
            runtime_values.get("tp_size", 1) * runtime_values.get("dp_size", 1) * runtime_values.get("cp_size", 1),
        )
        if "layers_to_capture" in runtime_values:
            runtime_values["layers_to_capture"] = tuple(
                int(layer_id) for layer_id in runtime_values["layers_to_capture"]
            )
        if self.config_module == "deepseek_v4":
            device = str(values.get("device", "npu:0"))
            runtime_values.setdefault("tp_rank", int(device.rsplit(":", 1)[1]) if ":" in device else 0)
            runtime_values.setdefault("ep_size", runtime_values.get("tp_size", 1))
            runtime_values.setdefault("ep_rank", runtime_values["tp_rank"])
        if self.config_module == "qwen3_5":
            runtime_values.setdefault(
                "moe_tp_size", runtime_values["world_size"] // max(runtime_values.get("ep_size", 1), 1)
            )
        object.__setattr__(self, "runtime", runtime or ModelRuntime(**runtime_values))
        normalized = _normalize_architecture(values, self.config_module, self.config_name)
        if self.draft_kind and (normalized.get("sliding_window") or 0) > 0:
            # Draft checkpoints specify the window directly; ModelArgs also
            # reflects Qwen's unrelated use_sliding_window=False default.
            normalized["use_sliding_window"] = True
            normalized["max_window_layers"] = 0
        module, name = self.config_module, self.config_name
        if module == "qwen3_5" and (normalized.get("num_experts", 0) > 0 or "moe" in str(values.get("model_type", ""))):
            module, name = "qwen3_5_moe", "Qwen3_5MoeTextConfig"
        if hf_config is None:
            import transformers

            config_class = getattr(transformers, name, None)
            if config_class is None:
                raise ImportError(
                    f"{name} is required by this Python model but is unavailable in Transformers "
                    f"{transformers.__version__}. Install a version providing this config; "
                    "see xllm/python/README.md for the tested versions."
                )
            accepted = {field.name for field in fields(config_class)}
            # Upstream __post_init__ consumes these checkpoint compatibility keys.
            accepted.update({"rope_theta", "partial_rotary_factor", "full_attention_interval"})
            if module in ("glm_moe_dsa", "glm5_next"):
                accepted.update({"index_topk_pattern", "index_topk_freq", "index_skip_topk_offset"})
            if module == "glm5_next":
                accepted.add("linear_attn_config")
            if module == "deepseek_v4":
                accepted.update({"compress_ratios", "num_hash_layers", "qk_rope_head_dim"})
            parameters = {key: value for key, value in normalized.items() if key in accepted}
            if name == "Glm5NextTextConfig":
                n_layers = parameters.get("num_hidden_layers", config_class.num_hidden_layers)
                if "first_k_dense_replace" in normalized and not parameters.get("mlp_layer_types"):
                    n_dense = min(normalized["first_k_dense_replace"], n_layers)
                    parameters["mlp_layer_types"] = ["dense"] * n_dense + ["sparse"] * (n_layers - n_dense)
                full_layers = (normalized.get("linear_attn_config") or {}).get("full_attn_layers")
                if full_layers is not None and not parameters.get("layer_types"):
                    parameters["layer_types"] = [
                        "indexed_attention" if index in full_layers else "linear_attention" for index in range(n_layers)
                    ]
            if name == "Glm5NextVisionConfig" and parameters.get("projection_intermediate_size", 1) is None:
                parameters["projection_intermediate_size"] = parameters.get(
                    "out_hidden_size", config_class.out_hidden_size
                ) * parameters.get("in_channels", config_class.in_channels)
            hf_config = config_class.from_dict(parameters)
        object.__setattr__(self, "hf_config", hf_config)
        options = {
            "topk_method": "noaux_tc",
            "moe_layer_freq": 1,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "attn_output_gate": True,
            "index_kpool_compress": False,
            "scale_fmt": "ue8m0",
            "n_group": 0,
            "topk_group": 0,
            "index_topk_pattern": None,
            "index_topk_freq": 1,
            "index_skip_topk_offset": 2,
        }
        # Dense and MoE Qwen3.5 share layers; absent branch dimensions are zero.
        if self.config_module == "qwen3_5":
            options.update(
                dict.fromkeys(
                    (
                        "intermediate_size",
                        "num_experts",
                        "num_experts_per_tok",
                        "moe_intermediate_size",
                        "shared_expert_intermediate_size",
                    ),
                    0,
                )
            )
            options["norm_topk_prob"] = True
        object.__setattr__(
            self,
            "options",
            {key: normalized.get(key, default) for key, default in options.items() if not hasattr(hf_config, key)},
        )
        draft_values = {key: values[key] for key in _DRAFT_FIELDS if key in values}
        nested_draft = values.get("dflash_config") or {}
        for key in ("block_size", "conv_group_size", "conv_kernel_size", "selector_rank", "selector_top_k"):
            value = values.get(f"dflash2_{key}")
            if value is None:
                value = values.get(key)
            if value is None:
                value = nested_draft.get(key, 0)
            draft_values[key] = int(value)
        draft_values["draft_vocab_size"] = values.get("draft_vocab_size") or getattr(hf_config, "vocab_size", 0)
        object.__setattr__(self, "draft", DraftConfig(**draft_values))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        return cls(**values)

    def with_runtime(self, **changes: Any) -> Self:
        """Share the architecture without modifying the caller's parallel state."""
        result = copy(self)
        object.__setattr__(result, "runtime", replace(self.runtime, **changes))
        return result

    def _rope_parameters(self, *, compress: bool = False) -> dict[str, Any]:
        rope = getattr(self.hf_config, "rope_parameters", None) or {}
        if "main" in rope:
            return rope["compress" if compress else "main"]
        return rope

    def __getattr__(self, name: str) -> Any:
        if name == "model_type":
            return self.runtime.model_type or self.hf_config.model_type
        if name in _RUNTIME_FIELDS:
            return getattr(self.runtime, name)
        if name in _DRAFT_FIELDS and self.draft_kind:
            return getattr(self.draft, name)
        config = object.__getattribute__(self, "hf_config")
        if name == "compress_rope_theta" and self.config_module == "deepseek_v4":
            return self._rope_parameters(compress=True).get("rope_theta", config.compress_rope_theta)
        if name in _ROPE_ALIASES:
            rope = self._rope_parameters(compress=name not in ("rope_theta", "partial_rotary_factor"))
            key = _ROPE_ALIASES[name]
            defaults = {
                "factor": 1.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "mrope_section": (),
                "partial_rotary_factor": 1.0,
            }
            if name == "rope_mscale" and self.config_module == "deepseek_v4":
                return rope.get("attention_factor", 1.0)
            if key in rope:
                return rope[key]
            if hasattr(config, name):
                return getattr(config, name)
            if key == "original_max_position_embeddings":
                return rope.get(key, config.max_position_embeddings)
            if key == "rope_theta":
                return rope.get(key, 10000.0)
            return rope.get(key, defaults[key])
        if name == "compress_ratios":
            return [config.compress_rates.get(kind, 1) for kind in config.layer_types]
        if name == "n_hash_layers":
            return config.mlp_layer_types.count("hash_moe")
        if name == "first_k_dense_replace" and not hasattr(config, name):
            return (getattr(config, "mlp_layer_types", None) or []).count("dense")
        canonical = _ALIASES.get(name, name)
        if hasattr(config, canonical):
            value = getattr(config, canonical)
            return 0 if canonical == "sliding_window" and value is None else value
        if canonical == "sliding_window":
            return 0
        options = object.__getattribute__(self, "options")
        if name in options:
            return options[name]
        raise AttributeError(f"{type(self).__name__} has no field {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RUNTIME_FIELDS:
            setattr(self.runtime, name, value)
        elif name in _DRAFT_FIELDS and self.draft_kind:
            setattr(self.draft, name, value)
        elif name in self.options:
            self.options[name] = value
        else:
            canonical = _ALIASES.get(name, name)
            if not hasattr(self.hf_config, canonical):
                raise AttributeError(f"{type(self).__name__} has no field {name!r}")
            setattr(self.hf_config, canonical, value)
