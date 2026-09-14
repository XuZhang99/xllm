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

"""Device regression tests for mapped Host KV and ACLGraph replay.

Compile the hisparse_gather and hisparse_store AOT families first, then set
XLLM_HISPARSE_KERNEL_ROOT to the hisparse_gather output directory;
hisparse_store must be a sibling directory.
The fixture builds production C++ wrappers and the mapped-memory allocator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> Any:
    import torch_npu
    from torch.utils.cpp_extension import load

    kernel_root = os.environ.get("XLLM_HISPARSE_KERNEL_ROOT")
    if not kernel_root:
        pytest.skip("Set XLLM_HISPARSE_KERNEL_ROOT after compiling the AOT family")
    root = Path(__file__).resolve().parents[2]
    artifact = Path(kernel_root).resolve()
    assert (artifact / "registry.inc").is_file()
    build = tmp_path_factory.mktemp("hisparse_native")
    binding = build / "binding.cpp"
    binding.write_text(
        r"""
#include <torch/extension.h>
#include "core/platform/npu/mapped_host_memory.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
torch::Tensor alloc(int64_t n,int64_t d) {return xllm::allocate_mapped_host_tensor({n,d},torch::kBFloat16,torch::Device(torch::kPrivateUse1,0));}
torch::Tensor gather(const torch::Tensor& h,const torch::Tensor& c,const torch::Tensor& s,const torch::Tensor& m,const torch::Tensor& t,torch::Tensor o) {xllm::kernel::npu::tilelang::hisparse_gather_out(h,c,s,m,t,o); return o;}
torch::Tensor store(const torch::Tensor& v,const torch::Tensor& s,torch::Tensor h) {xllm::kernel::npu::tilelang::hisparse_store(v,s,h); return h;}
TORCH_LIBRARY_FRAGMENT(xllm_ops,m) {
m.def("hisparse_store(Tensor values, Tensor slots, Tensor(a!) host) -> Tensor(a!)");
m.def("hisparse_gather_out(Tensor h, Tensor c, Tensor s, Tensor m, Tensor t, Tensor(a!) o) -> Tensor(a!)");

}
TORCH_LIBRARY_IMPL(xllm_ops,PrivateUse1,m) {
m.impl("hisparse_store",TORCH_FN(store));
m.impl("hisparse_gather_out",TORCH_FN(gather));

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
m.def("alloc",&alloc);
m.def("sfa",&xllm::kernel::npu::sparse_flash_attention);
m.def("qli_metadata",[](torch::Tensor q,torch::Tensor k,int64_t max_k) {
return xllm::kernel::npu::quant_lightning_indexer_metadata(64,1,128,0,0,q,k,q.numel(),1,max_k,"TND","PA_BSND",2048,3,INT64_MAX,INT64_MAX,1,"npu:0");
});
m.def("qli",[](torch::Tensor q,torch::Tensor k,torch::Tensor w,torch::Tensor qs,torch::Tensor ks,torch::Tensor qlen,torch::Tensor klen,torch::Tensor pages,torch::Tensor metadata) {
return std::get<0>(xllm::kernel::npu::quant_lightning_indexer(q,k,w,qs,ks,0,0,qlen,klen,pages,metadata,"TND","PA_BSND",2048,3,INT64_MAX,INT64_MAX,1,false));
});

}

"""
    )
    package = Path(torch_npu.__file__).parent
    toolkit = Path(os.environ.get("NPU_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    store_artifact = artifact.parent / "hisparse_store"
    assert (store_artifact / "registry.inc").is_file()
    sources = [
        root / "xllm/core/platform/npu/mapped_host_memory.cpp",
        root / "xllm/core/kernels/npu/tilelang/hisparse_gather_wrapper.cpp",
        root / "xllm/core/kernels/npu/tilelang/hisparse_store_wrapper.cpp",
        root / "xllm/core/kernels/npu/xllm_ops/sparse_flash_attention.cpp",
        root / "xllm/core/kernels/npu/xllm_ops/quant_lightning_indexer.cpp",
        root / "xllm/core/kernels/npu/xllm_ops/quant_lightning_indexer_metadata.cpp",
        binding,
    ]
    module = load(
        name="xllm_hisparse_test_native",
        sources=[str(path) for path in sources],
        build_directory=str(build),
        extra_include_paths=[
            str(root / "xllm"),
            str(package / "include"),
            str(toolkit / "include"),
        ],
        extra_cflags=[
            "-O0",
            "-std=c++17",
            f'-DXLLM_TL_HISPARSE_GATHER_REGISTRY_INC=\\"{artifact}/registry.inc\\"',
            f'-DXLLM_TL_HISPARSE_STORE_REGISTRY_INC=\\"{store_artifact}/registry.inc\\"',
        ],
        extra_ldflags=[
            f"-L{package}/lib",
            "-ltorch_npu",
            f"-L{toolkit}/lib64",
            "-lascendcl",
            "-lglog",
            str(artifact / "d64/hisparse_gather_d64_kernel.o"),
            str(artifact / "d512/hisparse_gather_d512_kernel.o"),
            str(store_artifact / "d64/hisparse_store_d64_kernel.o"),
            str(store_artifact / "d512/hisparse_store_d512_kernel.o"),
            str(toolkit / "lib64/libascendc_runtime.a"),
            "-lruntime",
        ],
    )
    torch.npu.set_device(0)
    return module


def test_mapped_allocation_exceeds_memlock_limit(native: Any) -> None:
    host = native.alloc(131072, 512)  # 128 MiB, above the container 64 MiB limit.
    host.fill_(3)
    torch.npu.synchronize()
    torch.testing.assert_close(host[[0, 65536, 131071]].cpu(), torch.full((3, 512), 3, dtype=torch.bfloat16))


def test_mapped_store_replay_and_ignore_padding(native: Any) -> None:
    key, rope = native.alloc(256, 512), native.alloc(256, 64)
    key.zero_()
    rope.zero_()
    new_key = torch.ones((4, 1, 512), dtype=torch.bfloat16, device="npu")[1:]
    new_rope = torch.ones((4, 1, 64), dtype=torch.bfloat16, device="npu")[1:]
    slots = torch.tensor([1, 129, -1], dtype=torch.int32, device="npu")

    def write() -> None:
        torch.ops.xllm_ops.hisparse_store(new_key.view(3, 512), slots, key)
        torch.ops.xllm_ops.hisparse_store(new_rope.view(3, 64), slots, rope)

    write()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        write()
    slots.copy_(torch.tensor([2, 130, -1], dtype=torch.int32, device="npu"))
    new_key.fill_(7)
    new_rope.fill_(9)
    graph.replay()
    torch.npu.synchronize()
    expected_key = torch.zeros((256, 512), dtype=torch.bfloat16)
    expected_rope = torch.zeros((256, 64), dtype=torch.bfloat16)
    expected_key[[1, 129]], expected_rope[[1, 129]] = 1, 1
    expected_key[[2, 130]], expected_rope[[2, 130]] = 7, 9
    # Mapped NPU aliases are accessed by kernels, not a direct CPU memcpy.
    readback_slots = torch.arange(256, device="npu")
    torch.testing.assert_close(key.index_select(0, readback_slots).cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(rope.index_select(0, readback_slots).cpu(), expected_rope, rtol=0, atol=0)
    del graph


@pytest.mark.parametrize("dim", [64, 512])
def test_mapped_gather_replay_checks_hits_and_stale_tags(native: Any, dim: int) -> None:
    host = native.alloc(256, dim)
    reference = torch.randn(256, dim, dtype=torch.bfloat16, device="npu")
    host.copy_(reference)
    hot = reference[:16].clone()
    tags = torch.arange(16, dtype=torch.int32, device="npu")
    mapping = torch.full((256,), -1, dtype=torch.int32, device="npu")
    mapping[:16] = tags
    slots = torch.tensor([1, 6, 17, 255, -1, 256], dtype=torch.int32, device="npu")
    output = torch.empty((6, dim), dtype=torch.bfloat16, device="npu")
    torch.ops.xllm_ops.hisparse_gather_out(host, hot, slots, mapping, tags, output)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        torch.ops.xllm_ops.hisparse_gather_out(host, hot, slots, mapping, tags, output)
    tags[1] = 100
    hot[1].fill_(99)
    slots.copy_(torch.tensor([1, 2, 18, 254, -1, 999], dtype=torch.int32, device="npu"))
    graph.replay()
    torch.npu.synchronize()
    expected = torch.zeros_like(output)
    expected[:4] = reference[torch.tensor([1, 2, 18, 254], device="npu")]
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    del graph


def test_mapped_chunked_prefill_attention_matches_hbm(native: Any) -> None:
    key = torch.randn((4096, 512), dtype=torch.bfloat16, device="npu")
    rope = torch.randn((4096, 64), dtype=torch.bfloat16, device="npu")
    host_key, host_rope = native.alloc(4096, 512), native.alloc(4096, 64)
    slots = torch.arange(4096, dtype=torch.int32, device="npu")
    torch.ops.xllm_ops.hisparse_store(key, slots, host_key)
    torch.ops.xllm_ops.hisparse_store(rope, slots, host_rope)
    query = torch.randn((16, 16, 512), dtype=torch.bfloat16, device="npu")
    query_rope = torch.randn((16, 16, 64), dtype=torch.bfloat16, device="npu")
    indices = torch.arange(2048, dtype=torch.int32, device="npu").expand(16, 1, 2048).contiguous()
    pages = torch.arange(32, dtype=torch.int32, device="npu").view(1, 32)
    q_lengths = torch.tensor([16], dtype=torch.int32, device="npu")
    kv_lengths = torch.tensor([4096], dtype=torch.int32, device="npu")

    def attend(k: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return native.sfa(
            query,
            k.view(32, 128, 1, 512),
            k.view(32, 128, 1, 512),
            indices,
            pages,
            q_lengths,
            kv_lengths,
            query_rope,
            r.view(32, 128, 1, 64),
            576**-0.5,
            1,
            "TND",
            "PA_BSND",
            3,
        )

    expected, actual = attend(key, rope), attend(host_key, host_rope)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.01, atol=0.01)


def test_cache_replay_slot_reuse_and_attention_equivalence(native: Any) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.hisparse import HiSparseCache, HiSparseWorkspace
    from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context

    host_key, host_rope = native.alloc(8192, 512), native.alloc(8192, 64)
    key = torch.randn(8192, 512, dtype=torch.bfloat16, device="npu")
    rope = torch.randn(8192, 64, dtype=torch.bfloat16, device="npu")
    host_key.copy_(key)
    host_rope.copy_(rope)
    cache = HiSparseCache(
        LayerCache(key=host_key.view(-1, 128, 1, 512), value=host_rope.view(-1, 128, 1, 64)),
        2048,
        HiSparseWorkspace(4096, torch.device("npu:0")),
    )
    indices = (
        torch.stack(
            [
                torch.cat((torch.randperm(4095, device="npu")[:2047], torch.tensor([4095], device="npu")))
                for _ in range(2)
            ]
        )
        .to(torch.int32)
        .view(2, 1, 2048)
    )
    pages = torch.arange(64, dtype=torch.int32, device="npu").view(2, 32)
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="npu")
    writes = torch.tensor([4095, 8191], dtype=torch.int32, device="npu")
    new_key = key.index_select(0, writes.to(torch.int64))
    new_rope = rope.index_select(0, writes.to(torch.int64))
    context = ForwardContext(None, torch.device("npu:0"), None, [], execution_state=AclGraphExecutionState({}))

    def execute() -> tuple[torch.Tensor, ...]:
        with forward_context(context):
            cache.store(new_key, new_rope, writes)
            cache.invalidate(writes)
            return cache.materialize(indices, pages, lengths, writes, 128)

    execute()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        selected = execute()
    small_graph = torch.npu.NPUGraph()
    with torch.npu.graph(small_graph):
        cache.invalidate(writes[:1])
        small_selected = cache.materialize(indices[:1], pages[:1], lengths[:1], writes[:1], 128)
    assert small_selected[0].data_ptr() == selected[0].data_ptr()
    for step in range(3):
        indices.copy_(indices.flip(-1).contiguous())
        if step == 1:
            pages.copy_(pages.flip(0).contiguous())
            writes.copy_(writes.flip(0).contiguous())
        key[4095].fill_(step + 7)
        new_key.copy_(key.index_select(0, writes.to(torch.int64)))
        new_rope.copy_(rope.index_select(0, writes.to(torch.int64)))
        small_graph.replay()
        graph.replay()
        torch.npu.synchronize()
        physical = (indices[:, 0].to(torch.int64) + (pages[:, 0].to(torch.int64) * 128).view(-1, 1)).flatten()
        torch.testing.assert_close(selected[0].view(-1, 512).cpu(), key.index_select(0, physical).cpu(), rtol=0, atol=0)
        torch.testing.assert_close(selected[1].view(-1, 64).cpu(), rope.index_select(0, physical).cpu(), rtol=0, atol=0)

    query = torch.randn(2, 16, 512, dtype=torch.bfloat16, device="npu")
    query_rope = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="npu")
    q_lengths = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    baseline = native.sfa(
        query,
        host_key.view(-1, 128, 1, 512),
        host_key.view(-1, 128, 1, 512),
        indices,
        pages,
        q_lengths,
        lengths,
        query_rope,
        host_rope.view(-1, 128, 1, 64),
        576**-0.5,
        1,
        "TND",
        "PA_BSND",
        3,
    )

    def attend(compact: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return native.sfa(
            query,
            compact[0],
            compact[0],
            compact[2],
            compact[3],
            q_lengths,
            compact[4],
            query_rope,
            compact[1],
            576**-0.5,
            1,
            "TND",
            "PA_BSND",
            3,
        )

    actual = attend(selected)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), baseline.cpu(), rtol=0.01, atol=0.01)
    attention_graph = torch.npu.NPUGraph()
    with torch.npu.graph(attention_graph):
        graph_selected = execute()
        graph_output = attend(graph_selected)
    attention_graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(graph_output.cpu(), baseline.cpu(), rtol=0.01, atol=0.01)
    # A padded graph row must neither read KV nor invalidate real cache slots.
    writes[1] = -1
    attention_graph.replay()
    torch.npu.synchronize()
    assert torch.isfinite(graph_output.cpu()).all()
    assert (graph_selected[2][1].cpu() == -1).all()
    assert (selected[0].view(2, 2048, 512)[1].cpu() == 0).all()
    del attention_graph, small_graph, graph


def test_indexer_graph_refreshes_metadata_after_warmup(native: Any, monkeypatch) -> None:
    from xllm.python.attention import npu_paged_attention
    from xllm.python.model_executor.forward_context import (
        AclGraphCaptureContext,
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    backend = npu_paged_attention.NpuPagedAttentionBackend.__new__(npu_paged_attention.NpuPagedAttentionBackend)
    qlen = torch.arange(1, 5, dtype=torch.int32, device="npu")
    klen = torch.full((4,), 16, dtype=torch.int32, device="npu")
    backend._mla_actual_seq_q = qlen
    backend._mla_actual_seq_kv = klen
    backend._mla_max_seqlen_q = 1
    backend._mla_max_seqlen_k = 16384
    backend._mla_quant_indexer_metadata = {}
    monkeypatch.setattr(
        npu_paged_attention.kernels,
        "quant_lightning_indexer_metadata",
        lambda nq, nk, dim, q, k, mq, mk, count, ratio: native.qli_metadata(q, k, mk),
        raising=False,
    )
    query = torch.randint(-30, 30, (4, 64, 128), dtype=torch.int8, device="npu")
    key = torch.randint(-30, 30, (64, 128, 1, 128), dtype=torch.int8, device="npu")
    weights = torch.randn((4, 64), dtype=torch.float16, device="npu")
    query_scale = torch.ones((4, 64), dtype=torch.float16, device="npu")
    key_scale = torch.ones((64, 128, 1), dtype=torch.float16, device="npu")
    pages = torch.zeros((4, 128), dtype=torch.int32, device="npu")
    pages[0, :32] = torch.arange(32, dtype=torch.int32, device="npu")
    pages[1, :8] = torch.arange(32, 40, dtype=torch.int32, device="npu")
    pages[2, :3] = torch.arange(40, 43, dtype=torch.int32, device="npu")
    pages[3, 0] = 43

    def execute() -> torch.Tensor:
        metadata = backend._get_quant_indexer_metadata(64, 1, 128, 2048, 1)
        return native.qli(query, key, weights, query_scale, key_scale, qlen, klen, pages, metadata)

    state = AclGraphExecutionState({})
    with forward_context(ForwardContext(backend, torch.device("npu:0"), None, [], execution_state=state)):
        execute()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        context = ForwardContext(
            backend,
            torch.device("npu:0"),
            None,
            [],
            acl_graph=AclGraphCaptureContext(torch.npu.current_stream(), []),
            execution_state=state,
        )
        with forward_context(context):
            captured = execute()
    # A subsequent prefill/graph bucket clears the backend cache. The captured
    # producer and its storage must survive and follow new, padded KV lengths.
    backend._mla_quant_indexer_metadata.clear()
    churn = [torch.full((1024,), -1, dtype=torch.int32, device="npu") for _ in range(32)]
    for lengths in ([34, 35, 27, 1], [4096, 1024, 257, 1]):
        klen.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        graph.replay()
        torch.npu.synchronize()
        metadata = native.qli_metadata(qlen, klen, 16384)
        expected = native.qli(query, key, weights, query_scale, key_scale, qlen, klen, pages, metadata)
        torch.testing.assert_close(captured.cpu(), expected.cpu(), rtol=0, atol=0)
    del graph, churn
