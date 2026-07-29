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
the current machine, and calls `tune(base_config, context)`. `tune` returns an
adjusted copy of the base config that is then written next to the launch
command and passed to the xllm binary via `--config_json_file`.

Shared helpers (`detect_hardware`, `check_device_count`) live in
`xllm.auto_config.utils` so other model profiles can reuse them; this module
only holds the qwen3-specific tuning policy.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from scripts.logger import logger
from xllm.auto_config.utils import (
    CpuArchEnum,
    Platform,
    check_device_count,
    detect_hardware,
)

MODEL_TYPE = "qwen3"


def tune(base_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a machine-adjusted copy of the base qwen3 config.

    The base config is never mutated. Adjustments here are intentionally simple
    and serve as a template for richer per-model tuning.
    """
    tuned_config = copy.deepcopy(base_config)

    hardware = context.get("hardware") or detect_hardware()
    logger.info(
        "qwen3 auto-tuning: detected hardware device_type=%s chip=%s arch=%s.",
        hardware.get("device_type"),
        hardware.get("chip"),
        hardware.get("arch"),
    )

    check_device_count(MODEL_TYPE, base_config, context)

    # Align the launch topology with the devices actually visible on this host.
    visible_device_count = context.get("visible_device_count")
    if isinstance(visible_device_count, int) and visible_device_count > 0:
        tuned_config["nnodes"] = visible_device_count

    # Hardware-specific nudges (worked example). Graph mode + ATB has only been
    # validated on the Ascend A2 (910b) generation here; be conservative
    # elsewhere.
    if not (Platform.is_npu() and Platform.get_ascend_soc_generation() == "a2"):
        tuned_config["enable_graph"] = False
        logger.info(
            "qwen3 auto-tuning: device_type=%s chip=%s is not Ascend A2; "
            "disabling graph mode.",
            hardware.get("device_type"),
            hardware.get("chip"),
        )

    if Platform.get_cpu_architecture() == CpuArchEnum.ARM:
        tuned_config["max_tokens_per_batch"] = min(
            tuned_config.get("max_tokens_per_batch", 8192), 4096
        )

    return tuned_config
