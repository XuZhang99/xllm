---
name: xllm-npu-profiler
description: Capture xLLM Ascend NPU profiling data, export timelines for https://ui.perfetto.dev, and analyze prefill/decode, operators, communication overlap, and host dispatch bottlenecks. Use for NPU timeline capture or Perfetto analysis of existing Ascend traces, not scheduler latency prediction sampling.
---

<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# xLLM NPU Profiling and Perfetto

Complete the workflow: warm up, capture a bounded window, export a timeline,
download it, open it in [Perfetto](https://ui.perfetto.dev), and diagnose specific
time intervals. If the user already has a trace, start with format validation
and visualization without restarting the service.

## Before capture

- Follow the repository's `AGENTS.md`. Discover the local checkout, SSH target,
  remote checkout, and container from the task environment. Do not assume a
  particular username, host, container name, or directory layout.
- Confirm container mounts and repository and script paths before synchronizing
  task files. Compare remotes, branches, HEADs, and relevant uncommitted contents
  between checkouts. Preserve unrelated changes; do not reset, clean, or overwrite
  the entire workspace.
- Read the deployment's server launch and request scripts to identify the model,
  ports, devices, and workload. An accuracy evaluation script is not the default
  trace workload. If a build is needed, run `python setup.py build` in the
  container checkout and record the actual executable path.
- Check `npu-smi info`, existing services, and free disk space. Do not stop other
  users' jobs. Record the device model, CANN/msprof versions, TP/DP/EP/CP, graph
  mode, input/output token counts, and concurrency. Do not infer hardware from a
  host alias.
- Create a separate run directory for each capture and retain previous data.
  Confirm service readiness, successful requests, and graph/compilation warmup
  before starting the measured capture.

## Choose a capture method

Prefer the [NPU capture and export procedure](references/capture.md): set
`PROFILING_MODE=dynamic` before service startup, attach to the service parent PID
in the same container PID namespace, control msprof with `start/stop/quit`, and
export every `PROF_*` after the collector exits and flushes its data. Setting the
variable only in the client or collector cannot change an existing service.

Inspect the platform branches of `WorkerImpl::start_profile/stop_profile` in
`xllm/core/runtime/worker_impl.cpp` for the current checkout. When this skill was
introduced, the NPU branch did not support the online HTTP profiler. Do not copy
CUDA `/start_profile` or `/stop_profile` instructions or assume `--profile_dir`
produces NPU traces. Neither `enable_profile_step_time` nor the latency tables
from `ProfileManager` are device timelines.

Keep captures short and preserve the deployment configuration. Choose a prefill
workload with long inputs and short outputs, or a decode workload with enough
steady decode steps. A decode capture still includes prefill; select the decode
interval explicitly. Capture an additional eager trace only when operator
mapping requires it, and retain the production graph trace as the primary evidence.

## Export and visualize

1. Follow [capture.md](references/capture.md) and enumerate files for every
   rank/device. Retain raw `PROF_*` directories and logs. Prefer complete
   `msprof_*.json` files; inspect the actual event format of existing
   `trace_view.json` or `*.pt.trace.json` files.
2. Check that files are nonempty, parse as JSON, and contain timestamped events.
   CSV files, databases, and metadata-only JSON are not usable timelines. Record
   file size, SHA-256, and rank/device identity.
3. Download the trace and **actually load it** using the
   [Perfetto analysis procedure](references/perfetto.md). Inspect the time range,
   Host API, device streams, kernels, and communication tracks. When browser
   controls are available, perform the website operations using supported local
   file selection or the documented native trace processor. If import is blocked,
   provide the exact local file path and manual steps, and report "exported; UI
   validation incomplete." Opening an empty website does not complete validation.
4. Capture screenshots and measurements for representative prefill and steady
   decode intervals. Support each conclusion with the trace file, rank/device,
   track, start/end times, units, and relevant event names.

## Diagnostic rules

- Check capture coverage first. Report missing CPU, HCCL, or rank tracks as "not
  captured," rather than concluding the activity did not occur. Do not invent
  prefill/decode labels when phases cannot be distinguished.
- Aggregate kernel calls, total duration, and mean duration by name, then interpret
  hotspots in their stream and phase context. Do not add CPU scopes, runtime APIs,
  and device kernels together as device time.
- Measure communication overlap using compute/communication interval intersections
  on the same clock and within the same window. Summed stream durations can exceed
  wall time; calculate busy/idle time using interval unions and state the denominator.
- Investigate host bubbles using the previous device task's end, the next task's
  start, intervening Host APIs, synchronization, copies, graph replay, and other
  streams. Blank space or a single threshold does not establish a host bottleneck.
- Compare rank skew only for matching requests/steps with verified clocks. A
  single-rank trace cannot characterize an entire TP/EP group. Do not concatenate
  independent JSON files and introduce PID/TID or clock collisions.
- Timelines alone do not establish KV fragmentation, HBM bandwidth utilization,
  or fusion opportunities. Obtain the relevant metrics, operator shapes, and
  current source before concluding. Separate observations, hypotheses, and tests.
- Profiling explains bottlenecks. User-visible speedups require matched
  before/after measurements with profiling disabled; cumulative operator time
  cannot directly establish a throughput improvement.

## Deliverables

Keep the following artifacts under one run directory and return clickable local
timeline and report paths:

```text
<run_id>/
  manifest.md           # Commit/branch, executable, environment, arguments, parent PID
  capture.log           # Start/stop/quit, errors, collector exit and flush evidence
  workload.log          # Readiness, warmup, successful measured requests, token counts
  export.log
  PROF_*/               # Raw/exported data; may remain remote with paths in manifest
  timelines/            # Local copies with rank/device subdirectories
  timeline_notes.md     # Bottlenecks, intervals, evidence, candidate changes and tests
  screenshots/          # Overview and key intervals when UI analysis was completed
```

Distinguish capture completion, timeline export, and Perfetto loading/analysis.
Retain artifacts and explain diagnostic limits after failed requests, empty traces,
export errors, warmup contamination, or missing tracks. Exit code zero alone does
not prove successful profiling. Clean up only collectors and temporary resources
created for this run, and record the final state of any service started for it.

## References

The linked references provide self-contained capture commands, analysis guidance,
and official tool documentation. No adjacent workflow repository is required.
