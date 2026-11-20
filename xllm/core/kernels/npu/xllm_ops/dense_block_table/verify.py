# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the published functional checker and record paired performance and memory."""

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch_npu
import triton
from solution import prepare, run

_ROOT = Path(__file__).resolve().parent
_Call = Callable[[], torch.Tensor]


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def _make_inputs(case: dict[str, Any], values: list) -> tuple[torch.Tensor, torch.Tensor]:
    logical, mapping = (torch.tensor(value, dtype=torch.int32) for value in values)
    if "requests" in case:
        logical = logical.reshape(case["requests"], case["columns"])
    logical, mapping = logical.npu(), mapping.npu()
    if not case.get("offset"):
        return logical, mapping

    views = []
    for tensor in (logical, mapping):
        storage = tensor.new_empty(tensor.numel() + 1)
        view = storage[1:].view(tensor.shape)
        view.copy_(tensor)
        views.append(view)
    return views[0], views[1]


def _negative_tests(inputs: tuple[torch.Tensor, torch.Tensor]) -> list[dict[str, str]]:
    logical, mapping = inputs
    strided = logical.new_empty((*logical.shape, 2))[..., 0]
    strided.copy_(logical)
    negative_mapping = mapping.clone()
    negative_mapping.view(-1)[0] = -1
    out_of_range = logical.clone()
    out_of_range.view(-1)[0] = torch.iinfo(torch.int32).max
    invalid = (
        ("input_count", (logical,), {}),
        ("unknown_attribute", inputs, {"unknown": 1}),
        ("cpu_input", (logical.cpu(), mapping), {}),
        ("dtype", (logical.to(torch.int64), mapping), {}),
        ("noncontiguous", (strided, mapping), {}),
        ("negative_mapping", (logical, negative_mapping), {}),
        ("logical_address_out_of_range", (out_of_range, mapping), {}),
    )
    results = []
    for name, tensors, attrs in invalid:
        try:
            prepare(tensors, attrs)
        except (TypeError, ValueError) as error:
            results.append({"id": name, "status": "PASS", "exception": type(error).__name__})
        else:
            raise AssertionError(f"prepare accepted {name}")
    return results


def _check_outputs(
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[torch.Tensor],
    expected: torch.Tensor,
) -> None:
    # Empty tensors have null data pointers even when their storage is distinct.
    storage_ids = {tensor.untyped_storage()._cdata for tensor in inputs}
    for output in outputs:
        assert output.device == inputs[0].device
        assert output.shape == expected.shape and output.dtype == expected.dtype
        assert output.is_contiguous() and torch_npu.get_npu_format(output) == 2
        storage_id = output.untyped_storage()._cdata
        assert storage_id not in storage_ids, "output storage is shared"
        storage_ids.add(storage_id)
        assert torch.equal(output.cpu(), expected), "output differs from the CPU oracle"


def _memory(
    call: _Call,
    inputs: tuple[torch.Tensor, torch.Tensor],
    expected: torch.Tensor,
    changed: torch.Tensor,
    changed_expected: torch.Tensor,
) -> dict[str, Any]:
    logical = inputs[0]
    original = logical.cpu()
    current = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    retained = []
    for stream in streams:
        stream.wait_stream(current)
        with torch.npu.stream(stream):
            # Order a producer and a consumer on the same non-default stream.
            logical.copy_(changed, non_blocking=True)
            retained.extend(call() for _ in range(3))
            logical.copy_(original, non_blocking=True)
        current.wait_stream(stream)
    torch.npu.synchronize()
    _check_outputs(retained, inputs, changed_expected)
    del retained
    assert torch.equal(logical.cpu(), original)
    for _ in range(3):
        call()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    output = call()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - before
    size = output.numel() * output.element_size()
    _check_outputs((output,), inputs, expected)
    del output
    for _ in range(20):
        call()
    torch.npu.synchronize()
    after = torch.npu.memory_allocated()
    assert after == before, f"active allocation changed from {before} to {after} bytes"
    return {
        "two_streams": "PASS",
        "retained_outputs_independent_and_match_cpu": 6,
        "repeat20_released": "PASS",
        "active_before_bytes": before,
        "active_after_bytes": after,
        "peak_increment_bytes": peak,
        "output_bytes": size,
    }


def _time_round(call: _Call, iterations: int) -> float:
    torch.npu.synchronize()
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        call()
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start_ns) / (iterations * 1_000)


def _percentile(samples: list[float], quantile: float) -> float:
    index = (len(samples) - 1) * quantile
    lower = int(index)
    return samples[lower] + (index - lower) * (samples[lower + 1] - samples[lower])


def _benchmark(case_id: str, baseline: _Call, candidate: _Call, acceptance: dict[str, Any]) -> dict[str, Any]:
    protocol = acceptance["measurement_protocol"]
    counts = protocol["iterations_per_round"]
    warmup = protocol["warmup"]
    calls = {"baseline": baseline, "candidate": candidate}
    for call in calls.values():
        for _ in range(warmup):
            call()
        torch.npu.synchronize()

    samples = {"baseline": [], "candidate": []}
    order = []
    for round_index in range(protocol["rounds"]):
        sequence = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        order.append(sequence)
        for arm in sequence:
            samples[arm].append(_time_round(calls[arm], counts))

    baseline_rounds, candidate_rounds = samples["baseline"], samples["candidate"]
    paired = [base - new for base, new in zip(baseline_rounds, candidate_rounds)]
    rng = random.Random(acceptance["seed"])
    bootstrap = sorted(statistics.mean(rng.choices(paired, k=len(paired))) for _ in range(acceptance["resamples"]))
    interval = [_percentile(bootstrap, p) for p in (0.025, 0.975)]
    baseline_mean = statistics.mean(baseline_rounds)
    candidate_mean = statistics.mean(candidate_rounds)
    baseline_median = statistics.median(baseline_rounds)
    candidate_median = statistics.median(candidate_rounds)
    saved_mean = statistics.mean(paired)
    saved_median = baseline_median - candidate_median
    limit = acceptance["max_mean_us"].get(case_id)
    checks = (
        {}
        if limit is None
        else {
            "candidate_mean_within_limit": candidate_mean <= limit,
            "positive_mean_saving": saved_mean > 0,
            "positive_median_saving": saved_median > 0,
            "positive_ci95_lower_bound": interval[0] > 0,
        }
    )
    return {
        "id": case_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "gate_applied": limit is not None,
        "checks": checks,
        "max_mean_us": limit,
        "warmup": warmup,
        "iterations": counts,
        "round_order": order,
        "baseline_rounds_us": baseline_rounds,
        "candidate_rounds_us": candidate_rounds,
        "paired_savings_us": paired,
        "baseline_mean_us": baseline_mean,
        "candidate_mean_us": candidate_mean,
        "baseline_median_us": baseline_median,
        "candidate_median_us": candidate_median,
        "saved_mean_us": saved_mean,
        "saved_median_us": saved_median,
        "saved_mean_ci95_us": interval,
    }


def _source_and_environment(fixtures: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(_ROOT), *args], text=True).strip()

    paths = [_ROOT / name for name in ("solution.py", "dense_kernels.py", "native.cpp", "setup.py", "verify.py")] + [
        fixtures / name for name in ("golden.py", "generate_data.py", "case.json", "op.json")
    ]
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "triton": triton.__version__,
        "triton_ascend": version("triton-ascend"),
        "cann_home": os.environ.get("ASCEND_HOME_PATH"),
        "device": torch.npu.get_device_name(0),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "triton_target": str(triton.runtime.driver.active.get_current_target()),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }


def _verify(fixtures: Path, output_dir: Path, report: dict[str, Any], supplements: dict[str, Any]) -> None:
    functional_path = output_dir / "functional.json"
    with (output_dir / "functional.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(fixtures / "golden.py"),
                "--verify-functional",
                "--implementation",
                str(_ROOT),
                "--output",
                str(functional_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    report.update(json.loads(functional_path.read_text()))
    report["scope"] = "functional_performance_memory"
    report["status"] = "RUNNING"
    sys.path.insert(0, str(fixtures))
    from generate_data import generate
    from golden import calculate, reference

    torch.npu.set_device(0)
    report["environment"].update(_source_and_environment(fixtures))
    contract = json.loads((fixtures / "case.json").read_text())
    acceptance = contract["acceptance"]
    report["measurement_protocol"] = acceptance["measurement_protocol"]
    report["statistics"] = {key: acceptance[key] for key in ("resamples", "seed", "percentile")}
    baseline_prepare = reference("baseline").prepare
    for case in contract["input_cases"]:
        case_id = case["id"]
        values, attrs = generate(case)
        inputs = _make_inputs(case, values)
        logical, mapping = inputs
        prepared = prepare(inputs, attrs)
        single = mapping.ndim == 1
        shape = logical.shape if single else (mapping.size(0), *logical.shape)
        expected = torch.tensor(calculate(values, attrs, single), dtype=torch.int32).reshape(shape)
        _check_outputs((run(inputs, attrs), run(inputs, attrs)), inputs, expected)
        report["direct_run"].append({"id": case_id, "status": "PASS"})

        if not case.get("performance") and case_id not in contract["generalization"]["diagnostic_ids"]:
            continue
        baseline = baseline_prepare(inputs, attrs, prepared.info)
        if case.get("performance"):
            if not report["negative_from_performance_input"]:
                report["negative_from_performance_input"] = _negative_tests(inputs)
            changed = logical.cpu().flip(dims=(1,))
            changed_values = [changed.tolist(), values[1]]
            changed_expected = torch.tensor(calculate(changed_values, attrs, single), dtype=torch.int32).reshape(shape)
            report["memory"][case_id] = {
                "candidate": _memory(prepared.run, inputs, expected, changed, changed_expected),
                "baseline": _memory(baseline, inputs, expected, changed, changed_expected),
            }
            report["benchmarks"].append(_benchmark(case_id, baseline, prepared.run, acceptance))
            if case_id == "dense_perf_large":
                # Exercise the native path at scale with a mapping that exceeds the UB budget.
                wide_mapping = torch.cat((mapping, mapping[:, :1]), dim=1)
                native_inputs = (logical, wide_mapping)
                _check_outputs((run(native_inputs, {}),), native_inputs, expected)
                report["large_native_mapping"] = "PASS"
        else:
            supplements["benchmarks"].append(_benchmark(case_id, baseline, prepared.run, acceptance))

    generalization_ids = contract["generalization"]["case_ids"]
    supplements["cases"] = [case for case in report["cases"] if case["id"] in generalization_ids]
    assert [case["id"] for case in supplements["cases"]] == generalization_ids
    assert all(case["status"] == "PASS" for case in supplements["cases"])
    # Cover the smallest mapping, the minimum sentinel and a physical ID
    # whose bits would be a NaN if treated as a float by the UB gather.
    values = [[[0, 0, -(2**31), 0, -1, 0, -7, 0]], [2**31 - 1]]
    expected = torch.tensor(calculate(values, {}, True), dtype=torch.int32)
    inputs = _make_inputs({"offset": True}, values)
    _check_outputs((run(inputs, {}),), inputs, expected)
    report["ub_int32_boundaries"] = "PASS"
    supplements["status"] = "PASS"
    report["status"] = "PASS" if all(case["status"] == "PASS" for case in report["benchmarks"]) else "FAIL"


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path, help="directory containing the task fixtures")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for validation records")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    report = {
        "status": "RUNNING",
        "benchmarks": [],
        "memory": {},
        "direct_run": [],
        "negative_from_performance_input": [],
    }
    supplements = {"status": "RUNNING", "cases": [], "benchmarks": []}
    try:
        _verify(args.fixtures.resolve(), output_dir, report, supplements)
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        if supplements["status"] == "RUNNING":
            supplements.update(status="FAIL", error=repr(error))
        raise
    finally:
        _write_json(output_dir / "result.json", report)
        _write_json(output_dir / "generalization.json", supplements)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    sys.exit(_main())
