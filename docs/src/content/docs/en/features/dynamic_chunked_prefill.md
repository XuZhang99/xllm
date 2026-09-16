---
title: Dynamic Chunked Prefill
---

<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

Dynamic chunked prefill uses startup latency measurements and each request's
cached history to choose its next prefill chunk. It is disabled by default.

This ports the startup profiling and quadratic chunk predictor from
[vLLM-Ascend Dynamic CPP](https://github.com/vllm-project/vllm-ascend/blob/0526083dd4aae02a02afeb0de680c119d8e5e732/vllm_ascend/core/profiling_chunk_predictor.py),
with reference to SGLang's `DynamicChunkSizer`, into xLLM's C++ scheduler.
It does **not** add LLM pipeline parallelism (PP); full Dynamic CPP requires a
future PP execution path.

## Configuration

Add these flags to an existing LLM command, or use the same keys in config JSON:

```bash
--enable_chunked_prefill=true \
--enable_dynamic_chunking=true \
--max_tokens_per_chunk_for_prefill=4096 \
--dynamic_chunk_min_tokens=256 \
--dynamic_chunk_smooth_factor=1.0 \
--dynamic_chunk_profile_samples=16
```

| Option | Default | Meaning |
| --- | --- | --- |
| `enable_dynamic_chunking` | `false` | Enable startup profiling and dynamic chunks |
| `dynamic_chunk_min_tokens` | `256` | Positive minimum before alignment and budget caps; no larger than the base chunk |
| `dynamic_chunk_smooth_factor` | `1.0` | Weight in `(0, 1]`; smaller values stay closer to the base chunk |
| `dynamic_chunk_profile_samples` | `16` | Number of measured lengths, between 8 and 64 |

The base chunk is the smaller of `max_tokens_per_chunk_for_prefill` and
`max_tokens_per_batch`. Alignment is
`lcm(64, block_size * kv_split_size_effective)`. Profiling requires at least
8 distinct aligned lengths and enough KV capacity and model context for the base
chunk. Insufficient distinct lengths or uninformative measurements produce a
warning and select static chunks. Model execution failures propagate.

## Execution

`ProfileManager::run_request` warms each length and measures three executions,
using their median. Synthetic KV is released without populating the prefix cache.
The existing synchronous output path supplies execution and orchestration timing.

A normalized, nonnegative least-squares fit estimates `f(L) = a*L² + b*L + c`.
Given history `H`, the predictor solves
`a*x² + (2*a*H+b)*x = f(base) - f(0)` using a stable positive-root formula.
A linear fit is allowed when curvature cannot be identified.

The result is smoothed, aligned, and capped by the base chunk, remaining prompt,
and available token budget. A final partial block can finish without alignment.
When no aligned block fits, the request remains queued. Prefix matching precedes
prediction; mixed-batch budget redistribution cannot enlarge a chunk past its
dynamic cap. Existing KV, DP token-budget, and SLO checks still apply.

The target applies to **one request's chunk**, not the aggregate time of a batch.
The existing latency-aware scheduler remains responsible for batch latency limits.

## Scope and validation

- Integrates with `PrefillFirstPolicy`, `DecodeFirstPolicy`, and `UnifiedPolicy`
  through `ContinuousScheduler` and the corresponding `DisaggPDScheduler` path.
  Decode-only instances skip dynamic prefill profiling.
- Rejects `ZeroEvictionScheduler`, `PDOOCScheduler`, and linear-attention models
  using prefix-cache checkpoints tied to fixed chunk boundaries.
- Fitting is performed at startup only. The upstream online history-aware
  recalibration is not included in this phase.
- Tests cover predictor mathematics and limits, config JSON, and scheduler budget
  redistribution/queue retention. Device validation and performance comparisons
  must use the actual deployment model and workload.

Future work: validate long-context/concurrent/prefix-cache/CP/graph combinations
on idle NPUs; add bounded online calibration with explicit sample ownership; then
implement PP layer placement, inter-rank transport, in-flight batches, and output
routing before measuring pipeline bubble reductions. No PP performance claim is
implied by this feature.

### Migration validation (2026-09-15)

The branch `feat/dynamic-chunked-prefill` starts at main commit
`beabafad07953629e6af0fa54125126a1349ebc3`. Local and remote changed files were
verified individually. `python setup.py build` completed in `zx-xllm-npu`,
including the server, export module, and all test targets. Unchanged TileLang
cache entries were reused after checking dependency bytes, compiler fingerprints,
and binary SHA-256 hashes.

The 25 config tests, five predictor tests, and a scheduler regression test passed.
The scheduler test exercises all three policies with normal and sub-alignment
budgets. The NPU validation script is at
`/home/xu/scripts/validate_dynamic_chunk_npu.py` on the development host. Its
pre-launch check found other workloads on the devices, so it started no server.
Actual NPU profiling, long-input, prefix-cache, concurrency, and throughput results
remain unverified.

### NPU comparison (2026-09-16)

Completed fixed A1 / dynamic B / fixed A2 on the same 16-NPU Ascend 910C host,
using one binary and GLM-5.3-w8a8 weights. TP=16, DP/EP=1, base chunk and batch
budget=8192, max sequences=4, block=128, KV cache=8 GiB, prefix cache and schedule
overlap disabled, graph enabled. All 216 measured requests passed: 72 per round,
inputs 512/8192/16384 tokens, outputs 128 tokens, concurrency 1/4, excluding warmup.

- GSM8K, the same 200 examples, 4-shot, temperature=0, max_tokens=1024: both fixed
  and dynamic scored 198/200 (99%), with identical per-example correctness. The
  same example hit the output limit in both modes; response text was not identical.
  This subset screening does not establish full-dataset or general long-context accuracy.
- Five functional checks passed per round, including 15440-token retrieval and
  concurrent requests. The predictor fitted successfully and used chunks
  8192/3584/2816/848 for that long input.
- Dynamic throughput relative to the mean of A1/A2: -0.1%/+0.5% for 512-token
  concurrency 1/4, -0.3%/+1.5% for 8192 tokens, and -4.1%/+0.2% for 16384 tokens.
- Median 16384-token serial TTFT increased from about 5.96 s to 6.47 s (+8.5%),
  with nearly identical fixed baselines. There is no clear overall throughput
  benefit with this configuration; keep the feature default-off.
- The harness now cleans up this test's detached workers between rounds. No model
  or scheduler implementation changed during testing. Prefix-cache, CP, and PP
  combinations remain unvalidated.

Raw requests, profiling, per-example scores, and the A/B/A report are retained at
`/home/xu/scripts/dynamic-chunk-comparison-20260916-133244/` on the development host.
