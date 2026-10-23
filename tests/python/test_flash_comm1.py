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

"""FlashComm1 admission, layout, and real TP collective equivalence.

Run device cases with two or four idle NPUs:
  torchrun --standalone --nproc_per_node=2 -m pytest -q tests/python/test_flash_comm1.py -k distributed
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from scripts.logger import logger
from tests.python.qwen3_5_test_utils import make_config
from xllm.python import distributed
from xllm.python.attention.backend import LayerCache
from xllm.python.layers.flash_comm1 import FlashComm1Context, FlashComm1MetadataBuilder, get_flash_comm1_context
from xllm.python.layers.linear import RowParallelLinear
from xllm.python.layers.npu.flash_comm1 import FlashComm1GatedMLP, FlashComm1RowParallelLinear
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnMetadata
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder
from xllm.python.model_executor.forward_context import ForwardContext, forward_context


def _metadata(num_tokens: int, **overrides: bool) -> SimpleNamespace:
    fields = dict(is_prefill=True, is_chunked_prefill=False, is_mixed=False, is_spec_verify=False, is_dummy=False)
    fields.update(overrides)
    return SimpleNamespace(slot_mapping=torch.zeros(num_tokens, dtype=torch.int32), **fields)


def _capture_partials(layer: RowParallelLinear, monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    partials: list[torch.Tensor] = []
    matmul = layer._forward_matmul

    def capture(x: torch.Tensor) -> torch.Tensor:
        output = matmul(x)
        # AllReduce mutates the projection output. Snapshot its actual input
        # before either collective, without substituting any computation.
        partials.append(output.detach().clone())
        return output

    monkeypatch.setattr(layer, "_forward_matmul", capture)
    return partials


def _sum_reference(
    partial: torch.Tensor, world_size: int, bias: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    values = distributed.tp_all_gather(partial, dim=0, world_size=world_size)
    values = values.reshape(world_size, *partial.shape).cpu().double()
    reference = values.sum(dim=0)
    magnitude = values.abs().sum(dim=0)
    additions = world_size - 1
    if bias is not None:
        bias = bias.detach().cpu().double()
        reference += bias
        magnitude += bias.abs()
        additions += 1
    # Bound any BF16 summation order by gamma_n * sum(abs(operands)),
    # including the replicated bias addition. A relative-only tolerance on
    # the final sum is invalid near cancellation. FP64 is the reference.
    unit_roundoff = torch.finfo(partial.dtype).eps / 2
    gamma = additions * unit_roundoff / (1 - additions * unit_roundoff)
    return reference, gamma * magnitude


def _assert_sum_roundoff(
    output: torch.Tensor, reference: torch.Tensor, bound: torch.Tensor, label: str, rank: int
) -> None:
    error = (output.detach().cpu().double() - reference).abs()
    logger.info(
        "rank=%s %s max_abs_error=%s max_roundoff_bound=%s", rank, label, error.max().item(), bound.max().item()
    )
    assert torch.all(error <= bound), f"{label} exceeds the BF16 summation error bound"


@pytest.mark.parametrize("num_tokens,active", [(0, False), (31, False), (32, True), (33, True)])
def test_prefill_threshold(num_tokens: int, active: bool) -> None:
    context = FlashComm1MetadataBuilder(2, 0, 32).build(_metadata(num_tokens))
    assert (context is not None) == active
    if context is not None:
        assert context.local_num_tokens % 16 == 0
        assert context.local_num_tokens * context.tp_size >= num_tokens


@pytest.mark.parametrize("excluded", ["is_mixed", "is_spec_verify", "is_dummy"])
def test_excluded_batches_keep_replicated_rows(excluded: str) -> None:
    assert FlashComm1MetadataBuilder(2, 0, 0).build(_metadata(33, **{excluded: True})) is None


def test_decode_stays_replicated_and_pure_chunked_prefill_can_shard() -> None:
    builder = FlashComm1MetadataBuilder(2, 0, 0)
    assert builder.build(_metadata(33, is_prefill=False)) is None
    assert builder.build(_metadata(33, is_prefill=False, is_chunked_prefill=True)) is not None


def test_configuration_defaults_and_negative_threshold() -> None:
    assert not make_config().enable_flashcomm1
    assert make_config().flashcomm1_min_prefill_tokens == 8192
    cfg = make_config(enable_flashcomm1=True, flashcomm1_min_prefill_tokens=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        cfg.validate()


def test_context_is_scoped_to_one_forward() -> None:
    plan = FlashComm1Context(33, 32, 2, 0)
    context = ForwardContext(None, torch.device("cpu"), None, [], layer_shared_cache={FlashComm1Context: plan})
    with forward_context(context):
        assert get_flash_comm1_context() is plan
    assert get_flash_comm1_context() is None


@pytest.fixture(scope="module")
def npu_tp() -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (2, 4) or "LOCAL_RANK" not in os.environ:
        pytest.skip("requires two or four idle NPUs and torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    group = distributed.init_tp_group(
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]) + 7,
        rank,
        world_size,
        str(device),
        rank,
        world_size,
        0,
    )
    from xllm.python.distributed.npu.hccl import get_hccl_comm

    get_hccl_comm(group, device, initialize=True)
    return device, rank, world_size


@pytest.mark.parametrize("num_tokens", [1, 31, 32, 33, 97])
def test_distributed_projection_and_mlp_match_all_reduce(
    npu_tp: tuple[torch.device, int, int], num_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    device, rank, world_size = npu_tp
    torch.manual_seed(1234)
    hidden = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device=device)
    plan = FlashComm1MetadataBuilder(world_size, rank, 0).build(_metadata(num_tokens))
    assert plan is not None
    torch.testing.assert_close(plan.gather(plan.shard(hidden)), hidden, rtol=0, atol=0)

    torch.manual_seed(4321 + rank)
    local_hidden_size = 64 // world_size
    projection = FlashComm1RowParallelLinear(
        local_hidden_size, 64, world_size, bias=True, dtype=torch.bfloat16, device=device
    )
    mlp = FlashComm1GatedMLP(64, 128, world_size, torch.bfloat16, device)
    projection_partials = _capture_partials(projection, monkeypatch)
    mlp_partials = _capture_partials(mlp.down_proj, monkeypatch)
    with torch.no_grad():
        for parameter in (*projection.parameters(), *mlp.parameters()):
            parameter.normal_(std=0.02)
        projection.bias.fill_(0.125)
        local_input = hidden[:, rank * local_hidden_size : (rank + 1) * local_hidden_size].contiguous()
        expected_projection = projection(local_input)
        expected_mlp = mlp(hidden)
        context = ForwardContext(None, device, None, [], layer_shared_cache={FlashComm1Context: plan})
        with forward_context(context):
            actual_projection = plan.gather(projection(local_input))
            actual_mlp = plan.gather(mlp(plan.shard(hidden)))
    projection_ref, projection_bound = _sum_reference(projection_partials[0], world_size, projection.bias)
    mlp_ref, mlp_bound = _sum_reference(mlp_partials[0], world_size)
    torch.testing.assert_close(projection_partials[0], projection_partials[1], rtol=0, atol=0)
    torch.testing.assert_close(mlp_partials[0], mlp_partials[1], rtol=0, atol=0)
    for name, actual, expected, reference, bound in (
        ("projection", actual_projection, expected_projection, projection_ref, projection_bound),
        ("mlp", actual_mlp, expected_mlp, mlp_ref, mlp_bound),
    ):
        _assert_sum_roundoff(actual, reference, bound, f"{name}/reduce_scatter", rank)
        _assert_sum_roundoff(expected, reference, bound, f"{name}/all_reduce", rank)
        if world_size == 2:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distributed_gdn_preserves_prefill_and_decode_state(
    npu_tp: tuple[torch.device, int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    device, rank, world_size = npu_tp
    cfg = make_config(
        num_experts=0,
        tp_size=world_size,
        tp_rank=rank,
        world_size=world_size,
        moe_tp_size=world_size,
        linear_num_key_heads=world_size,
        linear_num_value_heads=world_size,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention"],
    )
    torch.manual_seed(5234 + rank)
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
    partials = _capture_partials(layer.out_proj, monkeypatch)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.02)
        layer.norm_weight.fill_(1)
    torch.manual_seed(8234)
    hidden = torch.randn(37, 64, dtype=torch.bfloat16, device=device)
    decode_hidden = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    metadata = _metadata(37)
    metadata.slot_mapping = metadata.slot_mapping.to(device)
    metadata.linear_state_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    metadata.has_initial_state = torch.zeros(2, dtype=torch.bool, device=device)
    metadata.q_seq_lens_host = torch.tensor([17, 20], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor([0, 17, 37], dtype=torch.int32, device=device)
    metadata.q_cu_seq_lens_host_values = [0, 17, 37]
    plan = FlashComm1MetadataBuilder(world_size, rank, 0).build(metadata)
    outputs = []
    for enabled in (False, True):
        cache = LayerCache(
            key=None,
            value=None,
            conv=torch.zeros(3, 3, 384, dtype=torch.bfloat16, device=device),
            ssm=torch.zeros(3, 1, 128, 128, dtype=torch.float32, device=device),
        )
        builder = Qwen3_5GdnMetadataBuilder(cfg)
        builder.bind_layer_caches([cache])
        shared = {GdnMetadata: builder.build(metadata), FlashComm1Context: plan if enabled else None}
        with (
            torch.no_grad(),
            forward_context(ForwardContext(None, device, metadata, [cache], layer_shared_cache=shared)),
        ):
            output = layer(plan.shard(hidden) if enabled else hidden)
            if enabled:
                output = plan.gather(output)
        prefill_conv, prefill_ssm = cache.conv.clone(), cache.ssm.clone()
        decode = _metadata(2, is_prefill=False)
        decode.slot_mapping = decode.slot_mapping.to(device)
        decode.linear_state_indices = metadata.linear_state_indices
        with (
            torch.no_grad(),
            forward_context(
                ForwardContext(None, device, decode, [cache], layer_shared_cache={GdnMetadata: builder.build(decode)})
            ),
        ):
            decode_output = layer(decode_hidden)
        outputs.append((output, prefill_conv, prefill_ssm, decode_output, cache.conv, cache.ssm))
    prefill_ref, prefill_bound = _sum_reference(partials[0], world_size)
    decode_ref, decode_bound = _sum_reference(partials[1], world_size)
    torch.testing.assert_close(partials[0], partials[2], rtol=0, atol=0)
    torch.testing.assert_close(partials[1], partials[3], rtol=0, atol=0)
    # Recurrent state is independent of the output reduction and must remain
    # bitwise identical both immediately after prefill and after decode.
    for index in (1, 2, 4, 5):
        torch.testing.assert_close(outputs[1][index], outputs[0][index], rtol=0, atol=0)
    for index, name, reference, bound in (
        (0, "gdn_prefill", prefill_ref, prefill_bound),
        (3, "gdn_decode", decode_ref, decode_bound),
    ):
        _assert_sum_roundoff(outputs[0][index], reference, bound, f"{name}/off", rank)
        _assert_sum_roundoff(outputs[1][index], reference, bound, f"{name}/on", rank)
        if world_size == 2:
            torch.testing.assert_close(outputs[1][index], outputs[0][index], rtol=0, atol=0)
