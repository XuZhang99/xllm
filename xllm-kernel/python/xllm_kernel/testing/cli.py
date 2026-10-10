# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Minimal RMSNorm suite and report entry point; device imports are deferred."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# RMSNorm validation",
        "",
        f"Status: {report['status']}",
        "",
        "| Case | Implementation | Status | Device median (µs/op) |",
        "| --- | --- | --- | ---: |",
    ]
    for row in report["cases"]:
        median = row.get("timing", {}).get("device_interval_us", {}).get("median")
        lines.append(
            f"| {row['case_id']} | {row['implementation']} | {row['status']} | {median if median is not None else '—'} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main(*, benchmark: bool) -> None:
    parser = argparse.ArgumentParser(description="RMSNorm numerical checks and NPU timing; requires idle NPUs")
    parser.add_argument("--device", required=True, choices=["npu"])
    parser.add_argument("--mode", required=True, choices=["eager", "aclgraph"])
    parser.add_argument("--implementation", required=True, choices=["npu.xllm_native.rms_norm", "npu.xlite.rms_norm"])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 4096])
    parser.add_argument("--widths", type=int, nargs="+", default=[512, 2048, 6144])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["bfloat16", "float16", "float32"], default=["bfloat16", "float16"]
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument(
        "--revision", required=True, help="Exact source revision or an explicitly labeled dirty snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.widths + [args.iterations, args.repeats]) <= 0 or min(args.rows + [args.warmup]) < 0:
        parser.error("widths/iterations/repeats must be positive; rows/warmup must be nonnegative")

    import torch
    import torch_npu

    from xllm_kernel.benchmark.npu import measure
    from xllm_kernel.numerics.rms_norm import NumericsCase, inputs, verify
    from xllm_kernel.ops.normalization import RMS_NORM_CONTRACT
    from xllm_kernel.registry import KernelRegistry

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "not_run",
        "cases": [],
        "revision": args.revision,
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "platform": platform.platform(),
        },
        "mode": args.mode,
        "cache_policy": "hot_repeated_inputs_no_explicit_flush",
        "warmup": args.warmup,
        "note": "Device-event intervals include host submission gaps; graph replay amortizes launch overhead.",
        "measurement_target": "prepared_callable",
    }
    try:
        prepare_start = time.perf_counter()
        if args.implementation == "npu.xlite.rms_norm":
            from xllm_kernel.ops.normalization.xlite import register
        else:
            from xllm import xllm_export  # noqa: F401
            from xllm_kernel.ops.normalization.npu import register
        registry = KernelRegistry()
        register(registry)
        registry.freeze()
        plan = registry.prepare(
            RMS_NORM_CONTRACT, device=args.device, execution_mode=args.mode, implementation=args.implementation
        )
        report["preparation_s"] = time.perf_counter() - prepare_start
        report["selection"] = asdict(plan.spec)
        report["environment"]["device_name"] = torch.npu.get_device_name()
        cann_version = (
            Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")) / "version.cfg"
        )
        report["environment"]["cann_version"] = cann_version.read_text() if cann_version.is_file() else "unavailable"
        report["environment"]["npu_smi"] = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
        for dtype in args.dtypes:
            for width in args.widths:
                for rows in args.rows:
                    case = NumericsCase(rows, width, dtype)
                    row = {
                        "case_id": case.case_id,
                        "implementation": plan.spec.name,
                        "case": asdict(case),
                        "status": "not_run",
                    }
                    report["cases"].append(row)
                    if plan.spec.solution == "xlite" and (
                        dtype == "float32" or width % 64 or width > 8192 or rows * width > 2**32 - 1
                    ):
                        row.update(status="not_applicable", reason="outside declared xlite RMSNorm tensor domain")
                        continue
                    value, weight = inputs(case)
                    row["input_stride"] = list(value.stride())
                    row["weight_stride"] = list(weight.stride())
                    row["npu_format"] = torch_npu.get_npu_format(value)
                    function = lambda value=value, weight=weight, eps=case.eps: plan.function(value, weight, eps)
                    row.update(verify(plan.function, case, value, weight))
                    if row["status"] != "passed":
                        continue
                    warmup_start = time.perf_counter()
                    for _ in range(args.warmup):
                        function()
                    torch.npu.synchronize()
                    row["warmup_s"] = time.perf_counter() - warmup_start
                    divisor = 1
                    if args.mode == "aclgraph":
                        capture_start = time.perf_counter()
                        graph = torch.npu.NPUGraph()
                        divisor = 64 if benchmark else 1
                        with torch.npu.graph(graph):
                            outputs = [function() for _ in range(divisor)]
                        graph.replay()
                        torch.npu.synchronize()
                        graph_check = verify(lambda *_, output=outputs[-1]: output, case, value, weight)
                        row["graph_accuracy"] = graph_check
                        row["status"] = graph_check["status"]
                        row["capture_s"] = time.perf_counter() - capture_start
                        function = graph.replay
                    if benchmark and row["status"] == "passed":
                        row["timing"] = measure(
                            function, iterations=args.iterations, repeats=args.repeats, divisor=divisor
                        )
                    _write_report(args.output, report)
        passed = sum(row["status"] == "passed" for row in report["cases"])
        report["coverage"] = {
            state: sum(row["status"] == state for row in report["cases"])
            for state in ("passed", "failed", "not_applicable", "not_run")
        }
        report["status"] = (
            "passed"
            if passed and all(row["status"] in ("passed", "not_applicable") for row in report["cases"])
            else "failed"
        )
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        if report["cases"]:
            report["cases"][-1].update(status="error", error=report["error"])
        raise
    finally:
        _write_report(args.output, report)
    if report["status"] != "passed":
        raise SystemExit(1)
