---
title: NPU CPU binding
---

xLLM can bind each standalone Ascend serving process and its runtime threads to a dedicated CPU pool. Enable it with:

```bash
xllm --model /path/to/model --enable_cpu_binding=true ...
```

The default is `false`. JSON configuration also accepts `enable_cpu_binding`; an explicit command-line value takes precedence. Use the same binary with `--enable_cpu_binding=false` for a performance baseline.

## CPU allocation

The policy follows the [vLLM Ascend CPU binding design](https://docs.vllm.ai/projects/ascend/en/v0.23.0/developer_guide/Design_Documents/cpu_binding.html):

- A3 uses `global_slice`: split the sorted startup CPU affinity mask across the **complete** logical NPU inventory from `npu-smi info -m`. Hidden NPUs retain their slices, so separately launched processes using the same cpuset do not overlap merely because each exposes a different NPU.
- A2 uses `topo_affinity`: intersect each NPU's topology affinity with the startup cpuset, extend a single-NUMA pool to the next NUMA node within that cpuset, and split identical pools among all sharing NPUs. Missing topology selects global slicing; incomplete or partially overlapping topology groups are rejected.
- `ASCEND_RT_VISIBLE_DEVICES` is resolved in its original order. Runtime device `0` with `ASCEND_RT_VISIBLE_DEVICES=12,3` uses global logical NPU `12`.
- Ordinary worker threads use all but the last two CPUs in their pool. Threads named `acl_thread` use the penultimate CPU; `release_thread` uses the final CPU. New ordinary threads inherit their creator's affinity. Placement is refreshed after runtime initialization, weight loading, and the first model forward to include lazily created runtime threads.

For an A3 host with 640 allowed CPUs and 16 NPUs, NPU `i` receives CPUs `40*i` through `40*i+39`: 38 worker CPUs, one ACL CPU and one release CPU. At least three allowed CPUs per NPU are required. Sparse cpusets and remainders are supported; the allocation never expands the startup mask.

The startup thread requests a preferred NUMA memory policy and attempts to migrate existing pages before model weights or pinned host buffers are loaded. Future threads inherit that policy. Unavailable memory-policy permissions are logged and CPU binding still proceeds. Preferred placement permits allocations on other nodes; it is not strict memory binding.

## Optional IRQ placement

```bash
xllm --model /path/to/model \
  --enable_cpu_binding=true --enable_npu_irq_binding=true ...
```

`enable_npu_irq_binding` defaults to `false` and requires CPU binding. It reserves the first two CPUs of each pool for the current device's SQ/CQ IRQs, leaving worker CPUs `pool[2:-2]`. This requires at least five CPUs per NPU, readable PCI/MSI interrupt information, and writable `/proc/irq/*/smp_affinity_list`. Missing prerequisites are logged; worker-thread placement remains active. xLLM does not stop `irqbalance`; IRQ affinity changes remain host settings after process exit.

## Verify

Startup logs report the global logical NPU, strategy, CPU list, ACL/release CPUs, thread counts, and memory-policy outcome. Check actual placement after a warmup request:

```bash
# PID is the relevant xLLM rank process.
taskset -apc "$PID"
# Read each thread's name and mask when auditing ACL/release separation.
cat /proc/"$PID"/task/"$TID"/comm
cat /proc/"$PID"/task/"$TID"/status
```

Compare steady-state workloads after warmup, holding model, parallelism, graph mode, caches, prompts and concurrency fixed. An off/on/off comparison exposes baseline drift. Check process ownership and HBM as well as NPU utilization before testing on a shared host.

## Scope

- Standalone A2/A3 serving on Linux aarch64, one NPU worker per process; applies to both native and Python model execution. Offline embedding and multi-device host processes do not initialize this policy.
- CPU affinity masks currently support up to `CPU_SETSIZE` CPUs (normally 1024); an unsupported mask is rejected. Ascend 950 has a different cluster/UVB policy and is explicitly skipped.
- Topology discovery uses `npu-smi` and `lscpu` with bounded command timeouts. Insufficient CPUs, unknown device mapping, or unavailable affinity operations produce a warning and skip placement. A failed thread update attempts to restore the previous masks.
- CPU pools can span NUMA nodes if the host CPU numbering/cpuset is not NUMA aligned. Processes with different, overlapping cpusets are not globally coordinated.
