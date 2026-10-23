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

"""BF16 persistent GDN state with FP32 active-state computation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.python.qwen3_5_test_utils import make_linear_config
from xllm.python.attention.backend import LayerCache
from xllm.python.kernels_npu.gated_delta_net import mega_gdn_decode, mega_gdn_prefill
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_metadata_accepts_ssm_storage_precision(dtype: torch.dtype) -> None:
    cfg = make_linear_config(
        linear_num_key_heads=1, linear_num_value_heads=3, linear_key_head_dim=128, linear_value_head_dim=128
    )
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    cache = LayerCache(
        key=None,
        value=None,
        conv=torch.zeros(8, 3, 640, dtype=torch.bfloat16),
        ssm=torch.zeros(8, 3, 128, 128, dtype=dtype),
    )
    builder.bind_layer_caches([cache])
    indices = torch.tensor([2, 6], dtype=torch.int32)
    meta = builder.build(
        SimpleNamespace(
            slot_mapping=torch.zeros(2, dtype=torch.int32),
            linear_state_indices=indices,
            is_prefill=False,
            is_chunked_prefill=False,
        )
    )
    assert meta.state_caches[0].ssm is cache.ssm
    assert cache.ssm.element_size() == (2 if dtype == torch.bfloat16 else 4)


def _inputs(device: torch.device, tokens: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)

    def rand(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.1).to(device=device, dtype=dtype)

    return {
        "qkv": rand(tokens, 640),
        "z": rand(tokens, 3, 128),
        "b": rand(tokens, 3),
        "a": rand(tokens, 3),
        "conv_weight": rand(4, 640),
        "conv_state": rand(8, 3, 640),
        "a_log": torch.zeros(3, device=device),
        "dt_bias": torch.zeros(3, device=device),
        "ssm_state": rand(8, 3, 128, 128),
        "norm_weight": torch.ones(128, dtype=torch.bfloat16, device=device),
    }


def _decode(inputs: dict[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
    return mega_gdn_decode(**inputs, read_state_indices=indices, write_state_indices=indices)


def _prefill(inputs: dict[str, torch.Tensor], read: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
    values = dict(inputs)
    values["mixed_qkv"] = values.pop("qkv")
    values["ssm_cache"] = values.pop("ssm_state")
    return mega_gdn_prefill(
        **values,
        conv_state_read_indices=read,
        conv_state_write_indices=write,
        ssm_state_read_indices=read,
        ssm_state_write_indices=write,
        cu_seqlens=torch.tensor([0, 129, 134], dtype=torch.int32, device=read.device),
        num_matrices=9,
    )


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
def test_bf16_prefill_and_decode_match_rounded_fp32_state() -> None:
    values = _inputs(torch.device("npu"), 134)
    reference = {name: value.clone() for name, value in values.items()}
    reference["ssm_state"] = reference["ssm_state"].float()
    read = torch.tensor([-1, 4], dtype=torch.int32, device="npu")
    write = torch.tensor([2, 6], dtype=torch.int32, device="npu")
    actual = _prefill(values, read, write)
    expected = _prefill(reference, read, write)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    torch.testing.assert_close(values["ssm_state"], reference["ssm_state"].bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(values["conv_state"], reference["conv_state"], rtol=0, atol=0)
    for _ in range(3):
        for name in ("qkv", "z", "a", "b"):
            values[name] = values[name][:2].contiguous()
            reference[name] = values[name]
        reference["ssm_state"].copy_(reference["ssm_state"].bfloat16().float())
        actual = _decode(values, write)
        expected = _decode(reference, write)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
        torch.testing.assert_close(values["ssm_state"], reference["ssm_state"].bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(values["conv_state"], reference["conv_state"], rtol=0, atol=0)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
def test_bf16_decode_graph_replays_updated_state_slots() -> None:
    values = _inputs(torch.device("npu"), 2)
    initial = {name: value.clone() for name, value in values.items()}
    indices = torch.tensor([2, 6], dtype=torch.int32, device="npu")
    for _ in range(3):
        _decode(values, indices)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        output = _decode(values, indices)
    for slots in ([2, 6], [1, 5]):
        values["ssm_state"].copy_(initial["ssm_state"])
        values["conv_state"].copy_(initial["conv_state"])
        indices.copy_(torch.tensor(slots, dtype=torch.int32, device="npu"))
        reference = {name: value.clone() for name, value in initial.items()}
        expected = _decode(reference, indices)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(values["ssm_state"], reference["ssm_state"], rtol=0, atol=0)
        torch.testing.assert_close(values["conv_state"], reference["conv_state"], rtol=0, atol=0)


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("checkpoint_stride", [1, 2])
@pytest.mark.parametrize("value_heads", [3, 24])
def test_bf16_state_movement_preserves_inactive_and_shared_rows(checkpoint_stride: int, value_heads: int) -> None:
    from xllm.python.kernels_npu.triton.gdn_state import gather_gdn_states, scatter_gdn_states

    torch.manual_seed(17)
    state = torch.randn(12 * checkpoint_stride, value_heads, 128, 128, device="npu").bfloat16()
    conv_width = (2 * (value_heads // 3) + value_heads) * 128
    conv = torch.randn(12, checkpoint_stride + 2, conv_width, device="npu").bfloat16()
    before_state, before_conv = state.clone(), conv.clone()
    conv_read = torch.tensor([-1, 7, 7], dtype=torch.int32, device="npu")
    state_read = torch.tensor([-1, 7 * checkpoint_stride, 7 * checkpoint_stride], dtype=torch.int32, device="npu")
    work_state, work_conv, indices = gather_gdn_states(state, conv, state_read, conv_read, checkpoint_stride)
    torch.testing.assert_close(work_state[0], torch.zeros_like(work_state[0]), rtol=0, atol=0)
    torch.testing.assert_close(work_conv[0], torch.zeros_like(work_conv[0]), rtol=0, atol=0)
    for row in (1, 2):
        torch.testing.assert_close(
            work_state[row * checkpoint_stride], state[7 * checkpoint_stride].float(), rtol=0, atol=0
        )
        torch.testing.assert_close(work_conv[row], conv[7], rtol=0, atol=0)
    expected_indices = torch.tensor(
        [
            [-1, 1, 2],
            [0, 1, 2],
            [-1, checkpoint_stride, 2 * checkpoint_stride],
            [0, checkpoint_stride, 2 * checkpoint_stride],
        ],
        dtype=torch.int32,
        device="npu",
    )
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    work_state[::checkpoint_stride].add_(0.12345)
    work_conv.add_(0.25)
    conv_write = torch.tensor([2, 5, 9], dtype=torch.int32, device="npu")
    state_write = conv_write * checkpoint_stride
    scatter_gdn_states(state, conv, work_state, work_conv, state_write, conv_write, checkpoint_stride)
    before_state.index_copy_(0, state_write.long(), work_state[::checkpoint_stride].bfloat16())
    before_conv.index_copy_(0, conv_write.long(), work_conv)
    torch.testing.assert_close(state, before_state, rtol=0, atol=0)
    torch.testing.assert_close(conv, before_conv, rtol=0, atol=0)
