# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Auto-tuning profile for the `qwen3` model type.

The launcher loads the base `qwen3.json` config, builds a `context` describing
the current machine, and calls `tune(base_config, context)`. `tune` returns a
adjusted copy of the base config that is then written next to the launch
command and passed to the xllm binary via `--config_json_file`.

The helper functions below are worked examples of the kind of checks a profile
can perform: verifying the visible device count against the profile's optimal
`nnodes`, and nudging a few parameters based on the detected hardware.
"""

from __future__ import annotations

import copy
import platform
import re
import subprocess
from typing import Any, Dict

from scripts.logger import logger


def _detect_npu_chip() -> str:
    """Best-effort NPU chip name from `npu-smi info`.

    Never raises: `npu-smi` may be absent or permission-blocked. Returns a
    lowercase chip identifier (for example `910b`) or `"unknown"`.
    """
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"

    match = re.search(r"910[a-zA-Z0-9]*", completed.stdout)
    if match is None:
        return "unknown"
    return match.group(0).lower()


def detect_hardware() -> Dict[str, str]:
    """Return the current hardware descriptor: CPU arch and NPU chip.

    Both fields fall back gracefully so the caller can always rely on the keys
    being present.
    """
    return {
        "arch": platform.machine() or "unknown",
        "chip": _detect_npu_chip(),
    }


def check_device_count(base_config: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Check the visible device count against the profile's optimal `nnodes`.

    Returns True when they match. A mismatch is non-fatal: it is logged as a
    warning so the operator can decide whether to adjust the launch topology.
    """
    optimal_nnodes = base_config.get("nnodes")
    visible_device_count = context.get("visible_device_count")

    if optimal_nnodes is None or visible_device_count is None:
        logger.warning(
            "qwen3 auto-tuning: cannot verify device count "
            "(optimal nnodes=%s, visible devices=%s)",
            optimal_nnodes,
            visible_device_count,
        )
        return False

    if visible_device_count != optimal_nnodes:
        logger.warning(
            "qwen3 auto-tuning: visible device count %s does not match the "
            "profile's optimal nnodes %s; the tuned config keeps nnodes=%s.",
            visible_device_count,
            optimal_nnodes,
            visible_device_count,
        )
        return False

    logger.info(
        "qwen3 auto-tuning: visible device count %s matches optimal nnodes %s.",
        visible_device_count,
        optimal_nnodes,
    )
    return True


def tune(base_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a machine-adjusted copy of the base qwen3 config.

    The base config is never mutated. Adjustments here are intentionally simple
    and serve as a template for richer per-model tuning.
    """
    tuned_config = copy.deepcopy(base_config)

    hardware = context.get("hardware") or detect_hardware()
    logger.info(
        "qwen3 auto-tuning: detected hardware arch=%s chip=%s.",
        hardware.get("arch"),
        hardware.get("chip"),
    )

    check_device_count(base_config, context)

    # Align the launch topology with the devices actually visible on this host.
    visible_device_count = context.get("visible_device_count")
    if isinstance(visible_device_count, int) and visible_device_count > 0:
        tuned_config["nnodes"] = visible_device_count

    # Hardware-specific nudges (worked example). Only the 910b class has been
    # validated with graph mode + ATB here; be conservative on anything else.
    chip = hardware.get("chip", "unknown")
    if not chip.startswith("910b"):
        tuned_config["enable_graph"] = False
        logger.info(
            "qwen3 auto-tuning: chip=%s is not 910b; disabling graph mode.",
            chip,
        )

    if hardware.get("arch") == "aarch64":
        tuned_config["max_tokens_per_batch"] = min(
            tuned_config.get("max_tokens_per_batch", 8192), 4096
        )

    return tuned_config
