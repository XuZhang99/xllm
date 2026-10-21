#!/usr/bin/env python3
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

"""Wait until all visible NPU chips are idle, then exit immediately.

Requires only Python's standard library and npu-smi. Each poll emits one JSON
status line on stdout. Exit codes: 0 idle, 1 busy (--once) or overall timeout,
2 unavailable/invalid snapshot (--once), 130 interrupted. Query failures never
count as idle. This observes availability; it does not reserve devices.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_PAIR = re.compile(r"(\d+)\s+(\d+)")
_PCI = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")


@dataclass
class NpuDevice:
    npu_id: int
    chip_id: int
    physical_id: int
    health: str
    aicore_percent: int
    hbm_used_mb: int
    hbm_total_mb: int
    process_ids: list[int] = field(default_factory=list)


def parse_snapshot(output: str) -> list[NpuDevice]:
    """Parse the two-row device table and process table from npu-smi info."""
    if "HBM-Usage" not in output or "Process id" not in output:
        raise ValueError("Missing HBM or process table in npu-smi output")
    devices: dict[tuple[int, int], NpuDevice] = {}
    covered_boards: set[int] = set()
    board: int | None = None
    health = ""
    process_table = False
    for line in output.splitlines():
        if "Process id" in line:
            process_table = True
            continue
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        pair = _PAIR.fullmatch(columns[0])
        if process_table:
            empty = re.search(r"No running processes found in NPU\s+(\d+)", line)
            if empty:
                covered_boards.add(int(empty[1]))
                continue
            if pair is None:
                raise ValueError(f"Unrecognized process row: {line.strip()}")
            key = (int(pair[1]), int(pair[2]))
            if key not in devices or len(columns) < 4 or not columns[1].isdigit():
                raise ValueError(f"Invalid process row: {line.strip()}")
            devices[key].process_ids.append(int(columns[1]))
            covered_boards.add(key[0])
            continue
        if pair is not None and len(columns) >= 3 and _PCI.fullmatch(columns[1]):
            if board is None:
                raise ValueError("Device details appear before the NPU header")
            metrics = " ".join(columns[2:])
            memory = re.findall(r"(\d+)\s*/\s*(\d+)", metrics)
            utilization = metrics.split()[0]
            if not memory or not utilization.isdigit():
                raise ValueError(f"Unavailable utilization or HBM: {line.strip()}")
            used, total = map(int, memory[-1])
            if total <= 0 or used > total or int(utilization) > 100:
                raise ValueError(f"Invalid utilization or HBM: {line.strip()}")
            key = (board, int(pair[1]))
            if key in devices:
                raise ValueError(f"Duplicate NPU/chip entry: {key}")
            devices[key] = NpuDevice(board, int(pair[1]), int(pair[2]), health, int(utilization), used, total)
            continue
        header = re.fullmatch(r"(\d+)\s+\S+", columns[0])
        if header is not None and len(columns) >= 2:
            board, health = int(header[1]), columns[1]
    if not devices or covered_boards != {device.npu_id for device in devices.values()}:
        raise ValueError("Incomplete device/process inventory")
    physical_ids = {device.physical_id for device in devices.values()}
    if len(physical_ids) != len(devices):
        raise ValueError("Duplicate physical device IDs")
    return sorted(devices.values(), key=lambda device: device.physical_id)


def assess_snapshot(devices: list[NpuDevice], max_hbm_used_ratio: float = 0.1) -> dict:
    """Require healthy, inactive chips with no processes and recovered HBM."""
    if not devices or not math.isfinite(max_hbm_used_ratio) or not 0 <= max_hbm_used_ratio <= 1:
        raise ValueError("A nonempty inventory and a finite HBM ratio between 0 and 1 are required")
    busy = []
    for device in devices:
        reasons = []
        if device.health.upper() != "OK":
            reasons.append("health")
        if device.aicore_percent != 0:
            reasons.append("aicore")
        if device.process_ids:
            reasons.append("processes")
        if device.hbm_used_mb / device.hbm_total_mb > max_hbm_used_ratio:
            reasons.append("hbm")
        if reasons:
            busy.append({"physical_id": device.physical_id, "reasons": reasons})
    return {
        "state": "idle" if not busy else "busy",
        "device_count": len(devices),
        "busy_devices": busy,
        "process_count": len({pid for device in devices for pid in device.process_ids}),
        "hbm_used_percent": {
            str(device.physical_id): round(100 * device.hbm_used_mb / device.hbm_total_mb, 2) for device in devices
        },
    }


def _emit(status: dict) -> None:
    # JSON Lines is the CLI's program output, including unavailable snapshots.
    print(json.dumps({"time": datetime.now().astimezone().isoformat(timespec="seconds"), **status}), flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Check once without waiting")
    parser.add_argument("--interval", type=float, default=10, help="Polling interval in seconds (default: 10)")
    parser.add_argument("--timeout", type=float, default=0, help="Overall timeout in seconds; 0 waits indefinitely")
    parser.add_argument("--query-timeout", type=float, default=30, help="npu-smi query timeout in seconds")
    parser.add_argument("--expected-devices", type=int, help="Reject snapshots with a different chip count")
    parser.add_argument("--max-hbm-used-ratio", type=float, default=0.1, help="Maximum idle HBM usage (default: 0.1)")
    parser.add_argument("--npu-smi", default="npu-smi", help="npu-smi executable path")
    parser.add_argument("--snapshot-dir", type=Path, help="Retain raw snapshots for auditing")
    args = parser.parse_args(argv)
    for name in ("interval", "query_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and nonnegative")
    if not math.isfinite(args.max_hbm_used_ratio) or not 0 <= args.max_hbm_used_ratio <= 1:
        parser.error("--max-hbm-used-ratio must be between 0 and 1")
    if args.expected_devices is not None and args.expected_devices <= 0:
        parser.error("--expected-devices must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    deadline = time.monotonic() + args.timeout if args.timeout else None
    known_ids: set[int] | None = None
    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    while True:
        remaining = deadline - time.monotonic() if deadline is not None else math.inf
        if remaining <= 0:
            _emit({"state": "timeout"})
            return 1
        try:
            result = subprocess.run(
                [args.npu_smi, "info"],
                capture_output=True,
                text=True,
                timeout=min(args.query_timeout, remaining),
                check=True,
            )
            if args.snapshot_dir is not None:
                (args.snapshot_dir / f"{time.time_ns()}.txt").write_text(result.stdout)
            devices = parse_snapshot(result.stdout)
            if args.expected_devices is not None and len(devices) != args.expected_devices:
                raise ValueError(f"Expected {args.expected_devices} chips, received {len(devices)}")
            physical_ids = {device.physical_id for device in devices}
            if known_ids is not None and physical_ids != known_ids:
                raise ValueError("Device inventory changed between polls")
            known_ids = physical_ids
            status = assess_snapshot(devices, args.max_hbm_used_ratio)
        except FileNotFoundError as error:
            _emit({"state": "unavailable", "error": str(error)})
            return 2
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, OSError) as error:
            _emit({"state": "unavailable", "error": str(error)})
            if args.once:
                return 2
        else:
            _emit(status)
            if status["state"] == "idle":
                return 0
            if args.once:
                return 1
        remaining = deadline - time.monotonic() if deadline is not None else math.inf
        time.sleep(max(0, min(args.interval, remaining)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
