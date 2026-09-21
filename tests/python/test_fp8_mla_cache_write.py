# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Bit-exact device checks for the fused E4M3 MLA cache writer."""

from typing import Any

import pytest
import torch

pytest.importorskip("torch_npu")
tilelang = pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")

from xllm.python.attention.fp8_cache import dequantize_e4m3, quantize_e4m3  # noqa: E402
from xllm.python.kernels_npu.tilelang.fp8_mla_cache_write import (  # noqa: E402
    build_fp8_mla_cache_write_kernel,
)
from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS  # noqa: E402


@pytest.fixture(scope="module", params=("bf16", "float16", "float32"))
def compiled_kernel(request: pytest.FixtureRequest) -> tuple[torch.dtype, Any]:
    dtype = {"bf16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[request.param]
    kernels = {
        slot_dtype: tilelang.compile(
            build_fp8_mla_cache_write_kernel(request.param, slot_bytes),
            execution_backend="cython",
            pass_configs=DEFAULT_ASCEND_PASS_CONFIGS,
        )
        for slot_dtype, slot_bytes in ((torch.int32, 4), (torch.int64, 8))
    }
    return dtype, kernels


def _boundary_values(dtype: torch.dtype) -> torch.Tensor:
    if dtype in (torch.bfloat16, torch.float16):
        return torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    magnitudes = dequantize_e4m3(torch.arange(127, dtype=torch.uint8), torch.float32)
    midpoints = (magnitudes[1:] + magnitudes[:-1]) * 0.5
    below = torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf))
    above = torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf))
    positive = torch.cat((magnitudes, below, midpoints, above))
    special = torch.tensor(
        [0.0, -0.0, torch.inf, -torch.inf, torch.nan, -torch.nan, 449.0, -449.0], dtype=torch.float32
    )
    return torch.cat((positive, -positive, special))


def _inputs(values: torch.Tensor, slot_dtype: torch.dtype, padded_stride: bool) -> tuple[list[Any], torch.Tensor]:
    rows = (values.numel() + 575) // 576
    dense = torch.zeros(rows * 576, dtype=values.dtype)
    dense[: values.numel()] = values
    dense = dense.view(rows, 576)
    latent_stride = 544 if padded_stride else 512
    rope_stride = 96 if padded_stride else 64
    latent = torch.full((rows, latent_stride), 17.0, dtype=values.dtype)
    rope = torch.full((rows, rope_stride), -17.0, dtype=values.dtype)
    latent[:, :512] = dense[:, :512]
    rope[:, :64] = dense[:, 512:]
    slots = torch.arange(rows - 1, -1, -1, dtype=slot_dtype)
    latent_cache = torch.full((rows + 2, 512), 127, dtype=torch.uint8, device="npu")
    rope_cache = torch.full((rows + 2, 64), 127, dtype=torch.uint8, device="npu")
    args = [slots.npu(), latent.npu(), rope.npu(), latent_cache, rope_cache]
    return args, quantize_e4m3(dense)


def _check_cache(args: list[Any], expected: torch.Tensor) -> None:
    slots, _, _, latent_cache, rope_cache = args
    target = torch.full((latent_cache.size(0), 576), 127, dtype=torch.uint8)
    slots = slots.cpu().long()
    valid = slots >= 0
    target[slots[valid]] = expected[valid]
    actual = torch.cat((latent_cache.cpu(), rope_cache.cpu()), dim=-1)
    torch.testing.assert_close(actual, target, atol=0, rtol=0)


@pytest.mark.parametrize("slot_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("padded_stride", (False, True))
def test_all_bit_patterns_and_rounding_boundaries(
    compiled_kernel: tuple[torch.dtype, Any], slot_dtype: torch.dtype, padded_stride: bool
) -> None:
    dtype, kernels = compiled_kernel
    values = _boundary_values(dtype)
    assert values.dtype == dtype
    args, expected = _inputs(values, slot_dtype, padded_stride)
    kernels[slot_dtype](*args)
    _check_cache(args, expected)


@pytest.mark.parametrize("slot_dtype", (torch.int32, torch.int64))
def test_padding_slots_and_graph_replay(compiled_kernel: tuple[torch.dtype, Any], slot_dtype: torch.dtype) -> None:
    dtype, kernels = compiled_kernel
    kernel = kernels[slot_dtype]
    torch.manual_seed(123)
    args, expected = _inputs(torch.randn(49 * 576).to(dtype), slot_dtype, True)
    args[0][::3] = -1
    args[0][1] = -128
    kernel(*args)
    _check_cache(args, expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        kernel(*args)
    for iteration in range(2):
        updated = torch.randn(49, 576).to(dtype) * (iteration + 1)
        latent = args[1].cpu()
        rope = args[2].cpu()
        latent[:, :512] = updated[:, :512]
        rope[:, :64] = updated[:, 512:]
        args[1].copy_(latent)
        args[2].copy_(rope)
        args[0].copy_(torch.arange(49, dtype=slot_dtype))
        args[0][iteration::4] = -1
        args[3].fill_(127)
        args[4].fill_(127)
        graph.replay()
        _check_cache(args, quantize_e4m3(updated))
    args[0].fill_(-1)
    args[3].fill_(127)
    args[4].fill_(127)
    graph.replay()
    _check_cache(args, expected)
