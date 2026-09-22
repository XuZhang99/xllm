---
title: Pipeline Parallelism
---

<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

## Current implementation

The experimental Python/NPU GLM path (`model_type=glm_moe_dsa`) supports
pipeline stages combined with tensor parallelism. `pp_size` defaults to `1`.
For a world of 16 workers, `--pp_size=2` creates PP2 × TP8:
stage 0 owns ranks 0–7 and stage 1 owns ranks 8–15.

Stage `s` owns decoder layers in
`[floor(n_layers*s/pp_size), floor(n_layers*(s+1)/pp_size))`.
Only stage 0 loads embeddings; only the final stage loads the final normalization
and language-model head. Each stage loads its local decoder weights and allocates
local KV/index caches. Checkpoint layer names retain global indices, while the
attention cache indices are local. The common block count uses the largest stage's
per-block allocation, including its actual number of full indexer layers.

Matching TP lanes exchange hidden states and residuals over an HCCL PP subgroup.
If a boundary enters a shared indexer layer, the previous stage also sends the
int32 top-k indices. Sampling runs on TP rank 0 of the final stage, and the engine
routes that result back to the original sequences.

## Microbatch execution

The engine partitions a scheduled batch into up to `pp_size` nonempty microbatches
at sequence boundaries, preserving each sequence's assigned chunk budget.
Each stage has one ordered dispatch stream. It retires every TP worker in a
microbatch before preparing that stage's next input. Different stages can run
different microbatches concurrently. Outputs are applied only after all stages
complete successfully.

A single scheduled sequence creates one microbatch and cannot fill the pipeline.
Splitting consecutive chunks of one request into concurrent microbatches is not
implemented. Partitioning currently balances sequence counts, not predicted stage
latency. These distinctions matter when comparing this implementation with full
Dynamic Chunked Pipeline Parallel scheduling.

## Configuration

Add these options to **every rank's** existing GLM launch command:

```bash
--model_impl=python \
--pp_size=2 \
--dp_size=1 --ep_size=1 --cp_size=1 \
--enable_graph=false \
--enable_schedule_overlap=false \
--enable_shm=false \
--enable_chunked_prefill=true
```

TP size is derived from `nnodes / pp_size`; `pp_size` must divide the world size
and cannot exceed the decoder layer count. CLI and config JSON both accept
`pp_size`. Dynamic chunk sizing remains independently selectable with
`--enable_dynamic_chunking=true`; its startup profile measures the whole pipeline.
The current predictor does not fit separate stage costs.

This implementation requires online NPU Python GLM workers, eager execution,
DP=EP=CP=1 and no layerwise/KV splitting. Graph execution, speculative decoding,
EPLB, PD, host-cache offload, external KV storage, XTensor and the existing
multi-stream option are rejected. Beam-search kernel mode is also unsupported.

## Validation and follow-up

CPU tests cover stage ownership, shared top-k and residual equivalence, process
rank mapping, real four-process transport, ordered stage dispatch and unequal-stage
KV budgets. NPU full-model correctness and throughput are required before
production use; CPU transport validation does not validate HCCL execution.

Follow-up work includes stage latency profiling and weighted microbatch assignment,
consecutive prefill chunk interleaving, stage-local graph capture, and additional
parallel combinations and model families.

### Validation record (2026-09-16)

- Container `python setup.py build` completed successfully.
- Python GLM parallel/CP/indexer and collective regressions: 67 passed.
- C++ dispatcher: 1 passed; cache estimation: 24 passed; config: 26 passed.
- PP2 × TP8 GLM-5.3 with dynamic chunking: the launch guard detected foreign
  vLLM workers and stopped before starting our server. No NPU inference,
  accuracy or throughput result is available for this PP implementation.
- Developer-machine scripts: `/home/xu/scripts/pipeline_case.py` and
  `/home/xu/scripts/run_pipeline_smoke.sh`. The case runner supports `--pp-size`,
  `--dynamic`, `--smoke-only` and `--accuracy`. Use eager TP16 as the control
  when making PP performance comparisons.
