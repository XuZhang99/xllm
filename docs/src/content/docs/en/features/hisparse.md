---
title: GLM HiSparse (PyTorch/NPU)
---

HiSparse keeps full BF16 MLA KV in registered Host memory and only a bounded
working set in HBM. GLM-5.3 uses the existing `glm_moe_dsa` Python model
implementation shared with GLM-5.2. The DSA index cache stays fully resident in
HBM; `--indexer_cache_dtype=int8` remains supported.
Attention projections select static or dynamic W8A8 activation quantization from
the checkpoint tensors, including GLM-5.3 `W8A8_DYNAMIC` weights.

## Enable

Install the matching xllm_ops package, including its AICPU indexer metadata
kernels, and load the package's custom OPP environment before starting workers.

Add these options to the GLM-5.3 launch command on every server rank:

```bash
--model_impl=python \
--python_model_path=/path/to/xllm \
--enable_graph=true \
--python_graph_backend=aclgraph \
--enable_hisparse=true \
--hisparse_device_buffer_size=8192 \
--hisparse_host_cache_size=8589934592 \
--kv_cache_dtype=auto \
--enable_prefix_cache=false \
--enable_in_batch_prefix_cache=false \
--enable_schedule_overlap=false \
--cp_size=1 --kv_split_size=1 --layerwise_split_size=1
```

`hisparse_device_buffer_size` is the **total hot token capacity per layer per
worker**, shared by all requests, not a per-request reservation.
`hisparse_host_cache_size` is the maximum full MLA KV **bytes per worker across
all layers**. Across 16 workers, an 8 GiB setting can lock up to 128 GiB of Host
memory. Provision the corresponding Host RAM; the NPU registration owns the pinned pages.
Logical cache capacity is bounded by both this Host budget and available HBM
for full index data, slot maps, hot KV, and selected-KV scratch buffers.

The first implementation supports online NPU workers, BF16 MLA latent=512 and
RoPE=64, ordinary prefill (including chunked prefill), and eager/ACLGraph decode.
Top-K must be divisible by the cache block size, and
`max_seqs_per_batch * index_topk` must not exceed 1048576.
It rejects PD disaggregation, CP/KV/layer splitting, prefix sharing, MTP,
scheduler overlap, XTensor, sleep mode, and spawned offline workers.
It does not provide the SGLang CUDA PD direct-to-host transfer protocol.

## Data path

A TileLang store kernel writes full KV to mapped Host storage and skips padded slots.
Prefill uses the existing sparse attention path. Decode translates each logical Top-K through the original block
table. A TileLang kernel checks the reverse map and hot-slot tags, reads hits
from HBM and misses from Host, and packs selected KV into one HBM buffer shared
across layers and graph buckets.
Attention receives compact indices and an independent compact block table; the
indexer continues using the original logical context and block table.

The hot cache is refreshed from the current selection up to its configured
capacity. Tags reject stale reverse mappings when requests change order or slots
are reused. Every new KV write invalidates its prior hot mapping. Padding uses a
separate map sentinel and never reads Host KV. There is no CPU Top-K readback or
host callback inside the decode graph.

This version does not implement LRU replacement or IndexShare cross-layer
prefetch. The model's IndexShare selection semantics remain unchanged. Lower HBM
usage does not guarantee lower latency: Host misses add transfer traffic, and
prefill attention also accesses Host storage.

## Focused validation

Build the AOT family in the NPU container:

```bash
python xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend --device a3 --output-root .tmp/hisparse \
  --kernels hisparse_gather hisparse_store
MAX_JOBS=4 \
XLLM_HISPARSE_KERNEL_ROOT=$PWD/.tmp/hisparse/targets/ascend/hisparse_gather \
python -m pytest tests/python/test_hisparse.py tests/python/test_hisparse_npu.py -q
```

The NPU tests compile production wrappers and the Host allocator, check hit/miss
and stale-tag behavior, replay with changed indices and recycled token slots,
exercise padding, and compare full-Host versus compact sparse attention. They
also verify that indexer metadata is produced inside the decode graph after
warmup and refreshed for changed sequence lengths, including 4096-token context.
Whole-model generation and accuracy/throughput evaluation must additionally run
in the deployment environment. These focused tests do not establish full-model
accuracy or latency.
