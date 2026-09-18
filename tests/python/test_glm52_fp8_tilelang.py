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

"""Device tests for the FP8 sparse MLA DSL, including ACLGraph replay."""

from typing import Any

import pytest
import torch

pytest.importorskip("torch_npu")
tilelang = pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")

from xllm.python.attention.fp8_cache import (  # noqa: E402
    create_e4m3_decode_table,
    dequantize_e4m3,
    quantize_e4m3,
)
from xllm.python.kernels_npu.tilelang.glm52_fp8_sparse_mla_attention import (  # noqa: E402
    DEFAULT_ASCEND_PASS_CONFIGS,
    build_glm52_fp8_sparse_mla_attention_kernel,
)


@pytest.fixture(scope="module", params=(4, 8, 16))
def compiled_kernel(request: pytest.FixtureRequest) -> tuple[int, Any]:
    heads = request.param
    kernel = tilelang.compile(
        build_glm52_fp8_sparse_mla_attention_kernel(heads),
        execution_backend="cython",
        pass_configs=DEFAULT_ASCEND_PASS_CONFIGS,
    )
    return heads, kernel


def _make_inputs(heads: int, queries: int, length: int) -> tuple[list[Any], torch.Tensor]:
    torch.manual_seed(20260917)
    blocks = (length + 127) // 128
    raw_k = quantize_e4m3(torch.randn(blocks, 128, 1, 512).bfloat16() * 0.25)
    raw_rope = quantize_e4m3(torch.randn(blocks, 128, 1, 64).bfloat16() * 0.25)
    pages = torch.stack([torch.randperm(blocks, dtype=torch.int32) for _ in range(queries)])
    indices = torch.full((queries, 2048), -1, dtype=torch.int32)
    for row in range(queries):
        valid = min(length, 2048)
        indices[row, :valid] = torch.randperm(length, dtype=torch.int32)[:valid]
        if valid < 2048:
            # Interleave negative padding and out-of-range positive entries.
            indices[row, valid::2] = length + 127
            indices[row] = indices[row, torch.randperm(2048)]
    q = torch.randn(heads, queries, 512).bfloat16().mul_(0.25)
    q_rope = torch.randn(heads, queries, 64).bfloat16().mul_(0.25)
    logical_q = q.transpose(0, 1)
    logical_q_rope = q_rope.transpose(0, 1)
    output = torch.empty((queries, heads, 512), dtype=torch.bfloat16, device="npu")
    shapes = ((48, 64, 512), (48, 64, 64), (48, 16, 64), (48, 16, 64), (48, 16, 512), (24, 16, 512), (24, 16, 64))
    dtypes = (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
        torch.bfloat16,
    )
    workspaces = [torch.empty(shape, dtype=dtype, device="npu") for shape, dtype in zip(shapes, dtypes)]
    # The JIT adapter requires contiguous arguments. Pass the underlying Q
    # storage with the logical transposed strides, just as the C++ ABI does.
    args = [
        q.to("npu"),
        q_rope.to("npu"),
        raw_k.to("npu"),
        raw_rope.to("npu"),
        indices.to("npu"),
        pages.to("npu"),
        torch.full((queries,), length, dtype=torch.int32, device="npu"),
        create_e4m3_decode_table(torch.device("npu")),
        output,
        *workspaces,
        logical_q.stride(0),
        logical_q.stride(1),
        logical_q_rope.stride(0),
        logical_q_rope.stride(1),
        queries,
        blocks,
        0.0625,
    ]
    decoded_k = dequantize_e4m3(raw_k, torch.float32).reshape(-1, 512)
    decoded_rope = dequantize_e4m3(raw_rope, torch.float32).reshape(-1, 64)
    reference = []
    for row in range(queries):
        selected = indices[row][(indices[row] >= 0) & (indices[row] < length)].long()
        physical = pages[row, selected // 128].long() * 128 + selected % 128
        keys = decoded_k[physical]
        rope_keys = decoded_rope[physical]
        scores = (logical_q[row].float() @ keys.T + logical_q_rope[row].float() @ rope_keys.T) * 0.0625
        reference.append(torch.softmax(scores, -1) @ keys)
    return args, torch.stack(reference)


@pytest.mark.parametrize("queries,length", ((1, 129), (8, 4097), (25, 129), (49, 4097)))
def test_sparse_pages_and_query_strides(compiled_kernel: tuple[int, Any], queries: int, length: int) -> None:
    heads, kernel = compiled_kernel
    args, expected = _make_inputs(heads, queries, length)
    kernel(*args)
    torch.testing.assert_close(args[8].cpu().float(), expected, atol=2e-3, rtol=2e-2)


def test_all_e4m3_bytes_and_replayed_cache_updates(compiled_kernel: tuple[int, Any]) -> None:
    heads, kernel = compiled_kernel
    args, _ = _make_inputs(heads, 1, 1)
    raw = torch.arange(512, dtype=torch.int32).to(torch.uint8).reshape(1, 512)
    args[0].zero_()
    args[1].zero_()
    args[2][0, 0, 0].copy_(raw[0])
    args[4].fill_(-1)
    args[4][0, -1] = 0
    kernel(*args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        kernel(*args)
    for updated in (raw, raw.flip(-1)):
        args[2][0, 0, 0].copy_(updated[0])
        graph.replay()
        expected = dequantize_e4m3(updated).expand(heads, -1)
        torch.testing.assert_close(args[8][0].cpu(), expected, atol=0, rtol=0)
