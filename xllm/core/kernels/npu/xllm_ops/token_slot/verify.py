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

"""Run the task-supplied oracle, baseline, and acceptance protocol."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from importlib import metadata
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns
from types import ModuleType
from typing import Any

import solution
import torch
import triton

_REPOSITORY = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(_REPOSITORY))
from scripts.logger import logger  # noqa: E402

_DEVICE = "npu:0"
_INT32_MAX = (1 << 31) - 1
_MAIN_CASES = ("slot_perf_small", "slot_perf_large")
_TensorCall = Callable[[], torch.Tensor]
_Record = dict[str, Any]


def _input(values: list, offset: int = 0) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.int32, device=_DEVICE)
    if offset:
        backing = tensor.new_empty(tensor.numel() + offset)
        view = backing[offset:].view(tensor.shape)
        view.copy_(tensor)
        return view
    return tensor


def _golden_tensor(golden: ModuleType, values: list, attrs: dict[str, int]) -> torch.Tensor:
    return torch.tensor(golden.calculate(values, attrs, not isinstance(values[1][0], list)), dtype=torch.int32)


def _baseline(golden: ModuleType, inputs: list[torch.Tensor], attrs: dict[str, int]) -> _TensorCall:
    mapping = inputs[1]
    info = dict(
        layers=mapping.size(0) if mapping.ndim == 2 else 1,
        blocks=mapping.size(-1),
        single=mapping.ndim == 1,
    )
    return golden.reference("baseline").prepare(inputs, attrs, info)


def _mismatches(output: torch.Tensor, expected: torch.Tensor) -> int:
    if output.shape != expected.shape:
        raise ValueError(f"output shape {tuple(output.shape)} != {tuple(expected.shape)}")
    return int((output.cpu() != expected).sum().item())


def _independent(tensors: list[torch.Tensor]) -> bool:
    # Empty tensors have a null data pointer even when their storage objects differ.
    return len({tensor.untyped_storage()._cdata for tensor in tensors}) == len(tensors)


def _accuracy(
    golden: ModuleType,
    values: list,
    attrs: dict[str, int],
    case: _Record,
) -> _Record:
    offset = int(case.get("offset", False))
    inputs = [_input(value, offset) for value in values]
    literal_expected = case.get("expected")
    expected = _golden_tensor(golden, values, attrs)
    if literal_expected is not None and expected.tolist() != literal_expected:
        raise ValueError("fixture literal differs from CPU golden")
    candidate = solution.prepare(inputs, attrs)
    baseline = _baseline(golden, inputs, attrs)
    results = [candidate.run(), baseline(), candidate.run(), baseline(), solution.run(inputs, attrs)]
    checks = {
        "candidate_mismatches": max(_mismatches(results[i], expected) for i in (0, 2, 4)),
        "baseline_mismatches": max(_mismatches(results[i], expected) for i in (1, 3)),
        "shape": all(output.shape == expected.shape for output in results),
        "dtype": all(output.dtype == torch.int32 for output in results),
        "npu": all(output.device == inputs[0].device for output in results),
        "contiguous": all(output.is_contiguous() for output in results),
        "inputs_unchanged": all(tensor.cpu().tolist() == value for tensor, value in zip(inputs, values)),
        "independent_storage": _independent(inputs + results),
        "input_offsets": all(tensor.storage_offset() == offset for tensor in inputs),
    }
    passed = all(value is True if isinstance(value, bool) else value == 0 for value in checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "shape": list(expected.shape),
        "input_shapes": [list(t.shape) for t in inputs],
        "storage_offsets": [t.storage_offset() for t in inputs],
        "attributes": attrs,
    }


def _negative(inputs: list[torch.Tensor], attrs: dict[str, int]) -> list[_Record]:
    slots, mapping = inputs
    noncontiguous = slots.new_empty((slots.numel(), 2))[:, 0]
    noncontiguous.copy_(slots)
    bad_slots, negative_mapping, overflow_mapping = slots.clone(), mapping.clone(), mapping.clone()
    bad_slots[0] = _INT32_MAX
    negative_mapping.flatten()[0] = -1
    overflow_mapping.flatten()[0] = _INT32_MAX
    cases = [
        ("input_count", [slots], attrs),
        ("unknown_attribute", inputs, dict(attrs, unknown=1)),
        ("cpu_input", [slots.cpu(), mapping], attrs),
        ("dtype", [slots.to(torch.int64), mapping], attrs),
        ("noncontiguous", [noncontiguous, mapping], attrs),
        ("negative_mapping", [slots, negative_mapping], attrs),
        ("logical_address_out_of_range", [bad_slots, mapping], attrs),
        ("zero_block_size", inputs, {"block_size": 0}),
        ("bool_block_size", inputs, {"block_size": True}),
        ("output_int32_overflow", [slots, overflow_mapping], attrs),
    ]
    records = []
    for name, tensors, attributes in cases:
        try:
            solution.prepare(tensors, attributes)
        except (TypeError, ValueError) as error:
            records.append(dict(id=name, status="PASS", exception=type(error).__name__, message=str(error)))
        else:
            records.append(dict(id=name, status="FAIL", exception=None))
    return records


def _state_check(
    golden: ModuleType, candidate: _TensorCall, inputs: list[torch.Tensor], attrs: dict[str, int], name: str
) -> _Record:
    expected = _golden_tensor(golden, [tensor.cpu().tolist() for tensor in inputs], attrs)
    output = candidate()
    mismatches = _mismatches(output, expected)
    return dict(id=name, status="FAIL" if mismatches else "PASS", candidate_mismatches=mismatches)


def _dynamic_checks(
    golden: ModuleType, values: list, attrs: dict[str, int], large: bool
) -> tuple[list[_Record], list[_Record]]:
    inputs = [_input(values[0]), _input(values[1])]
    slots, mapping = inputs
    candidate = solution.prepare(inputs, attrs).run
    slots.copy_(_input(list(reversed(values[0]))))
    dynamic = [_state_check(golden, candidate, inputs, attrs, "logical_values_changed")]
    mapping.copy_(_input([list(reversed(row)) for row in values[1]]))
    dynamic.append(_state_check(golden, candidate, inputs, attrs, "mapping_values_changed"))
    if large:
        page_size = attrs["block_size"]
        slots[0] = page_size - 1
        mapping[:, 0] = (_INT32_MAX + 1) // page_size - 1
        boundary = _state_check(golden, candidate, inputs, attrs, "large_path_int32_output_boundary")
        boundary["status"] = (
            "PASS" if boundary["status"] == "PASS" and candidate()[0, 0].item() == _INT32_MAX else "FAIL"
        )
        dynamic.append(boundary)

    rebindings = []
    for name, live, replacement_values in (
        ("mapping_set_storage", mapping, values[1]),
        ("logical_set_storage", slots, values[0]),
    ):
        replacement = _input(replacement_values, offset=1)
        old_ptr = live.data_ptr()
        live.set_(replacement.untyped_storage(), replacement.storage_offset(), live.size(), live.stride())
        record = _state_check(golden, candidate, inputs, attrs, name)
        record.update(
            pointer_changed=old_ptr != live.data_ptr(),
            storage_offset=live.storage_offset(),
            old_alignment_mod16=old_ptr % 16,
            new_alignment_mod16=live.data_ptr() % 16,
        )
        if not record["pointer_changed"] or record["storage_offset"] != 1:
            record["status"] = "FAIL"
        rebindings.append(record)
    return dynamic, rebindings


def _state_records(golden: ModuleType, values: list, attrs: dict[str, int], large: bool) -> _Record:
    dynamic, rebindings = _dynamic_checks(golden, values, attrs, large)
    passed = all(record["status"] == "PASS" for record in dynamic + rebindings)
    return dict(dynamic=dynamic, rebindings=rebindings, status="PASS" if passed else "FAIL")


def _boundary_cases() -> dict[str, list[_Record]]:
    result: dict[str, list[_Record]] = {"large_nonpower_boundaries": [], "logical_i32_boundaries": [], "extra": []}
    for size in (3, 7, 31):
        last_page = (_INT32_MAX - size + 1) // size
        slots = [-2147483648, -1, 0, size - 1, size, 2 * size - 1, 4096 * size - 1]
        repeats = (65536 + len(slots) - 1) // len(slots)
        row = [last_page - page for page in range(4096)]
        result["large_nonpower_boundaries"].append(
            dict(
                id=f"large_nonpower_{size}_near_i32",
                values=[(slots * repeats)[:65536], [row] * 32],
                attributes={"block_size": size},
            )
        )
    for size, route in ((1073741823, "nonpower"), (1073741824, "power2")):
        slots = [-2147483648, -1, 0, size - 1, size, 2 * size - 1, _INT32_MAX]
        row = [0, 1, 0]
        for scale, values in (("small", [slots, row]), ("large", [(slots * 147)[:1024], [row] * 32])):
            result["logical_i32_boundaries"].append(
                dict(id=f"{scale}_logical_i32_{route}", values=values, attributes={"block_size": size})
            )
    result["extra"].extend(
        [
            dict(
                id="global_mapping_read",
                values=[[-1, 0, 4095, 4096], list(range(4097))],
                attributes={"block_size": 1},
                offset=True,
            ),
            dict(
                id="max_page_size",
                values=[[-1, 0, _INT32_MAX - 1, _INT32_MAX], [0, 0]],
                attributes={"block_size": _INT32_MAX},
            ),
        ]
    )
    return result


def _time_round(call: _TensorCall, iterations: int) -> float:
    torch.npu.synchronize()
    started = perf_counter_ns()
    for _ in range(iterations):
        call()
    torch.npu.synchronize()
    return (perf_counter_ns() - started) / (1000 * iterations)


def _benchmark(calls: dict[str, _TensorCall], acceptance: _Record, target: float | None) -> _Record:
    protocol = acceptance["measurement_protocol"]
    for call in calls.values():
        _time_round(call, protocol["warmup"])
    times: dict[str, list[float]] = {key: [] for key in calls}
    orders = []
    for round_index in range(protocol["rounds"]):
        order = ["candidate", "baseline"] if round_index % 2 else ["baseline", "candidate"]
        orders.append(order)
        for key in order:
            times[key].append(_time_round(calls[key], protocol["iterations_per_round"]))

    baseline, candidate = times["baseline"], times["candidate"]
    differences = [a - b for a, b in zip(baseline, candidate)]
    rng = random.Random(acceptance["seed"])
    resampled = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(acceptance["resamples"]))

    def percentile(probability: float) -> float:
        index = probability * (len(resampled) - 1)
        left = int(index)
        return resampled[left] + (resampled[left + 1] - resampled[left]) * (index - left)

    interval = [percentile(0.025), percentile(0.975)]
    saved_median = median(baseline) - median(candidate)
    gates = (
        {}
        if target is None
        else dict(
            target_mean=mean(candidate) <= target,
            positive_mean_savings=mean(differences) > 0,
            positive_median_savings=saved_median > 0,
            positive_ci95_lower=interval[0] > 0,
        )
    )
    return dict(
        status="PASS" if all(gates.values()) else "FAIL",
        baseline_rounds_us=baseline,
        candidate_rounds_us=candidate,
        round_order=orders,
        warmup=protocol["warmup"],
        iterations=protocol["iterations_per_round"],
        baseline_mean_us=mean(baseline),
        candidate_mean_us=mean(candidate),
        baseline_median_us=median(baseline),
        candidate_median_us=median(candidate),
        saved_mean_us=mean(differences),
        saved_median_us=saved_median,
        saved_mean_ci95_us=interval,
        target_mean_us=target,
        gates=gates,
    )


def _memory(call: _TensorCall, inputs: list[torch.Tensor], expected: torch.Tensor) -> _Record:
    call()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    output = call()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - before
    output_bytes = output.numel() * output.element_size()
    del output
    torch.npu.synchronize()
    after_single = torch.npu.memory_allocated()

    producer = torch.npu.current_stream()
    streams = [torch.npu.Stream() for _ in range(2)]
    kept = []
    for stream in streams:
        stream.wait_stream(producer)
        with torch.npu.stream(stream):
            kept.extend(call() for _ in range(3))
    for stream in streams:
        stream.synchronize()
    streams_ok = all(_mismatches(output, expected) == 0 for output in kept)
    independent = _independent(inputs + kept)
    del kept
    torch.npu.synchronize()
    after_streams = torch.npu.memory_allocated()
    for _ in range(20):
        call()
    torch.npu.synchronize()
    after = torch.npu.memory_allocated()
    passed = streams_ok and independent and before == after_single == after_streams == after and peak >= output_bytes
    return dict(
        status="PASS" if passed else "FAIL",
        two_streams=streams_ok,
        retained_outputs_independent=independent,
        repeat20_released=before == after,
        active_before_bytes=before,
        active_after_single_bytes=after_single,
        active_after_streams_bytes=after_streams,
        active_after_bytes=after,
        peak_increment_bytes=peak,
        output_bytes=output_bytes,
    )


def _environment(fixtures: Path) -> _Record:
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    install_info = toolkit / "aarch64-linux/ascend_toolkit_install.info"
    cann = dict(line.split("=", 1) for line in install_info.read_text().splitlines() if "=" in line)
    target = triton.runtime.driver.active.get_current_target()
    git = ["git", "-c", f"safe.directory={_REPOSITORY}", "-C", str(_REPOSITORY)]
    files = [Path(__file__), Path(solution.__file__)] + [
        fixtures / name for name in ("case.json", "golden.py", "generate_data.py", "op.json")
    ]
    return dict(
        host=socket.gethostname(),
        utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        python=platform.python_version(),
        device=torch.npu.get_device_name(),
        triton_target=str(target),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        task_queue_enable=os.environ.get("TASK_QUEUE_ENABLE"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        cann=cann,
        versions={package: metadata.version(package) for package in ("torch", "torch-npu", "triton-ascend")},
        commit=subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(git + ["branch", "--show-current"], text=True).strip(),
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        formal_platform=str(target.arch).startswith("Ascend910_93") and "9.1.0" in cann.values(),
    )


def _checked(name: str, work: Callable[[], _Record]) -> _Record:
    try:
        record = work()
    except Exception as error:
        logger.exception(f"{name} failed")
        record = dict(status="FAIL", error=f"{type(error).__name__}: {error}")
    logger.info(f"{name} {record['status']}")
    return dict(id=name, **record)


def _write_report(path: Path, record: _Record) -> None:
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True, help="directory of the original task files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development", action="store_true", help="allow development measurements on 910B2")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        sys.path.insert(0, str(args.fixtures.resolve()))
        golden = importlib.import_module("golden")
        generator = importlib.import_module("generate_data")
        cases = json.loads((args.fixtures / "case.json").read_text())
        if torch.npu.device_count() != 1:
            raise RuntimeError("expose exactly one logical NPU using ASCEND_RT_VISIBLE_DEVICES")
        torch.npu.set_device(0)
        environment = _environment(args.fixtures)
        _write_report(output_dir / "environment.json", environment)
        logger.info(f"Device {environment['device']} on {environment['triton_target']}")

        main = dict(
            cases=[],
            negative=[],
            dynamic_updates={},
            storage_rebindings={},
            engineering_by_case={},
            benchmarks=[],
            status="FAIL",
            formal_acceptance="NOT_RUN",
        )
        generalization = dict(cases=[], benchmarks=[])
        definitions = _boundary_cases()
        groups = {"cases": main["cases"], **{key: [] for key in definitions}}
        fixtures: dict[str, tuple[list, dict[str, int]]] = {}
        for case in cases["input_cases"]:
            values, attrs = generator.generate(case)
            fixtures[case["id"]] = values, attrs
            report = generalization["cases"] if "generalization" in case else groups["cases"]
            report.append(_checked(case["id"], partial(_accuracy, golden, values, attrs, case)))
        for group, extra_cases in definitions.items():
            for case in extra_cases:
                groups[group].append(
                    _checked(case["id"], partial(_accuracy, golden, case["values"], case["attributes"], case))
                )
        main.update(groups)

        values, attrs = fixtures[_MAIN_CASES[0]]
        main["negative"] = _negative([_input(value) for value in values], attrs)
        for name in (*_MAIN_CASES, *cases["generalization"]["diagnostic_ids"]):
            values, attrs = fixtures[name]
            inputs = [_input(value) for value in values]
            calls = {"baseline": _baseline(golden, inputs, attrs), "candidate": solution.prepare(inputs, attrs).run}
            if name in _MAIN_CASES:
                expected = _golden_tensor(golden, values, attrs)
                for key, call in calls.items():
                    main["engineering_by_case"].setdefault(name, {})[key] = _checked(
                        f"{name}/{key}", partial(_memory, call, inputs, expected)
                    )
                state = _checked(
                    f"{name}/state", partial(_state_records, golden, values, attrs, name == _MAIN_CASES[1])
                )
                main["dynamic_updates"][name] = state.get("dynamic", [state])
                main["storage_rebindings"][name] = state.get("rebindings", [state])
            report = main if name in _MAIN_CASES else generalization
            target = cases["acceptance"]["max_mean_us"].get(name)
            report["benchmarks"].append(_checked(name, partial(_benchmark, calls, cases["acceptance"], target)))

        small_wide, large_wide, small_pow2, large_pow2 = groups["logical_i32_boundaries"]
        main.update(
            small_logical_i32_boundary=small_wide,
            large_logical_i32_boundary=large_wide,
            power2_logical_i32_boundaries=[large_pow2, small_pow2],
        )
        required = [
            *[record for records in groups.values() for record in records],
            *main["negative"],
            *[record for records in main["dynamic_updates"].values() for record in records],
            *[record for records in main["storage_rebindings"].values() for record in records],
            *[record for records in main["engineering_by_case"].values() for record in records.values()],
            *main["benchmarks"],
            *generalization["cases"],
            *generalization["benchmarks"],
        ]
        passed = all(record["status"] == "PASS" for record in required)
        main["status"] = "PASS" if passed else "FAIL"
        if environment["formal_platform"]:
            main["formal_acceptance"] = main["status"]
        _write_report(output_dir / "result.json", main)
        _write_report(output_dir / "generalization.json", generalization)
        return 0 if passed and (environment["formal_platform"] or args.development) else 1
    except Exception as error:
        logger.exception("Validation interrupted")
        _write_report(output_dir / "failure.json", dict(status="FAIL", error=f"{type(error).__name__}: {error}"))
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
