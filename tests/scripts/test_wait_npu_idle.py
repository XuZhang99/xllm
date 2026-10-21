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

import contextlib
import io
import json
import subprocess
import unittest
from unittest import mock

from scripts import wait_npu_idle

_SECOND_CHIP = """| 0 Ascend910 | OK | - 35 0 / 0 |
| 1 1 | 0000:9F:00.0 | 0 0 / 0 3000 / 65536 |
"""
_IDLE = (
    """| NPU Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |
| Chip Phy-ID | Bus-Id | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |
| 0 Ascend910 | OK | 178.2 35 0 / 0 |
| 0 0 | 0000:9D:00.0 | 0 0 / 0 3000 / 65536 |
"""
    + _SECOND_CHIP
    + """| NPU Chip | Process id | Process name | Process memory(MB) | Process id in container |
| No running processes found in NPU 0 |
"""
)
_FULL_HBM = _IDLE.replace("3000 / 65536", "65266/ 65536")


class NpuIdleTest(unittest.TestCase):
    def test_parse_both_chips_and_hbm(self) -> None:
        devices = wait_npu_idle.parse_snapshot(_IDLE)
        self.assertEqual([(d.npu_id, d.chip_id, d.physical_id) for d in devices], [(0, 0, 0), (0, 1, 1)])
        self.assertEqual([d.hbm_used_mb for d in devices], [3000, 3000])
        self.assertEqual(wait_npu_idle.assess_snapshot(devices)["state"], "idle")

    def test_full_hbm_without_processes_is_busy(self) -> None:
        status = wait_npu_idle.assess_snapshot(wait_npu_idle.parse_snapshot(_FULL_HBM))
        self.assertEqual(status["state"], "busy")
        self.assertEqual(status["process_count"], 0)
        self.assertEqual([d["reasons"] for d in status["busy_devices"]], [["hbm"], ["hbm"]])

    def test_host_process_blocks_idle_without_container_pid_or_name(self) -> None:
        snapshot = _IDLE.replace("| No running processes found in NPU 0 |", "| 0 1 | 12345 | | 0 | N/A |")
        status = wait_npu_idle.assess_snapshot(wait_npu_idle.parse_snapshot(snapshot))
        self.assertEqual(status["process_count"], 1)
        self.assertEqual(status["busy_devices"], [{"physical_id": 1, "reasons": ["processes"]}])

    def test_activity_or_bad_health_is_busy(self) -> None:
        for snapshot, reason in [
            (_IDLE.replace("| 0 0 / 0 3000", "| 1 0 / 0 3000"), "aicore"),
            (_IDLE.replace("| OK |", "| Warning |"), "health"),
        ]:
            with self.subTest(reason=reason):
                status = wait_npu_idle.assess_snapshot(wait_npu_idle.parse_snapshot(snapshot))
                self.assertEqual(status["state"], "busy")
                self.assertTrue(all(reason in device["reasons"] for device in status["busy_devices"]))

    def test_missing_metrics_or_process_inventory_is_rejected(self) -> None:
        for snapshot in (
            "",
            _IDLE.split("| NPU Chip")[0],
            _IDLE.replace("| No running processes found in NPU 0 |", ""),
            _IDLE.replace("| 0 0 / 0 3000", "| N/A 0 / 0 3000"),
            _IDLE.replace("| No running processes found in NPU 0 |", "| 9 9 | 12 | xllm | 0 | 12 |"),
            _IDLE.replace("| 1 1 |", "| 1 0 |"),
            _IDLE + "| unrecognized process row |\n",
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                wait_npu_idle.parse_snapshot(snapshot)

    def test_empty_inventory_cannot_be_idle(self) -> None:
        with self.assertRaises(ValueError):
            wait_npu_idle.assess_snapshot([])

    def test_once_exit_codes(self) -> None:
        for snapshot, expected in [(_IDLE, 0), (_FULL_HBM, 1), ("truncated", 2)]:
            with (
                self.subTest(expected=expected),
                mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, snapshot)),
                mock.patch.object(wait_npu_idle.time, "sleep") as sleep,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(wait_npu_idle.main(["--once", "--expected-devices", "2"]), expected)
                sleep.assert_not_called()

    def test_idle_returns_immediately_without_a_stability_period(self) -> None:
        responses = [subprocess.CompletedProcess([], 0, output) for output in (_FULL_HBM, _IDLE)]
        with (
            mock.patch.object(subprocess, "run", side_effect=responses) as query,
            mock.patch.object(wait_npu_idle.time, "sleep") as sleep,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(wait_npu_idle.main(["--interval", "2"]), 0)
            self.assertEqual(query.call_count, 2)
            sleep.assert_called_once_with(2)

    def test_query_failure_is_not_idle(self) -> None:
        responses = [subprocess.TimeoutExpired(["npu-smi", "info"], 30), subprocess.CompletedProcess([], 0, _IDLE)]
        output = io.StringIO()
        with (
            mock.patch.object(subprocess, "run", side_effect=responses),
            mock.patch.object(wait_npu_idle.time, "sleep"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(wait_npu_idle.main([]), 0)
        self.assertEqual(
            [json.loads(line)["state"] for line in output.getvalue().splitlines()], ["unavailable", "idle"]
        )

    def test_incomplete_expected_inventory_is_unavailable(self) -> None:
        with (
            mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, _IDLE)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(wait_npu_idle.main(["--once", "--expected-devices", "16"]), 2)

    def test_changing_inventory_cannot_end_wait(self) -> None:
        responses = [
            subprocess.CompletedProcess([], 0, output) for output in (_FULL_HBM, _IDLE.replace(_SECOND_CHIP, ""), _IDLE)
        ]
        output = io.StringIO()
        with (
            mock.patch.object(subprocess, "run", side_effect=responses),
            mock.patch.object(wait_npu_idle.time, "sleep"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(wait_npu_idle.main([]), 0)
        self.assertEqual(
            [json.loads(line)["state"] for line in output.getvalue().splitlines()], ["busy", "unavailable", "idle"]
        )

    def test_overall_timeout(self) -> None:
        with (
            mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, _FULL_HBM)),
            mock.patch.object(wait_npu_idle.time, "monotonic", side_effect=[0, 0, 3, 3]),
            mock.patch.object(wait_npu_idle.time, "sleep"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(wait_npu_idle.main(["--timeout", "2"]), 1)


if __name__ == "__main__":
    unittest.main()
