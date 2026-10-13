# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Explicit NPU timing; graph failures propagate and are never timed as eager."""

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from xllm_kernel.numerics.suite import PreparedCase


def measure_graph(prepared: PreparedCase, row: dict[str, Any], *, iterations: int, repeats: int) -> None:
    """Validate graph replay before timing and retain captured outputs throughout."""
    capture_start = time.perf_counter()
    graph = torch.npu.NPUGraph()
    divisor = 64
    with torch.npu.graph(graph):
        outputs = [prepared.function() for _ in range(divisor)]
    graph.replay()
    torch.npu.synchronize()
    row["graph_accuracy"] = prepared.validate(lambda output=outputs[-1]: output)
    row["status"] = row["graph_accuracy"]["status"]
    row["capture_s"] = time.perf_counter() - capture_start
    if row["status"] == "passed":
        row["timing"] = measure(graph.replay, iterations=iterations, repeats=repeats, divisor=divisor)


def measure(function: Callable[[], Any], *, iterations: int, repeats: int, divisor: int = 1) -> dict:
    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        wall_start = time.perf_counter_ns()
        start.record()
        host_start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        host_end = time.perf_counter_ns()
        end.record()
        end.synchronize()
        wall_end = time.perf_counter_ns()
        count = iterations * divisor
        samples.append(
            {
                "host_enqueue_us": (host_end - host_start) / 1000 / count,
                "device_interval_us": start.elapsed_time(end) * 1000 / count,
                "wall_us": (wall_end - wall_start) / 1000 / count,
            }
        )
    report = {"samples": samples, "iterations": iterations, "repeats": repeats, "operations_per_call": divisor}
    for metric in samples[0]:
        values = sorted(row[metric] for row in samples)
        median = statistics.median(values)
        report[metric] = {
            "median": median,
            "p90_nearest_rank": values[(9 * len(values) + 9) // 10 - 1],
            "median_absolute_deviation": statistics.median(abs(v - median) for v in values),
        }
    return report
