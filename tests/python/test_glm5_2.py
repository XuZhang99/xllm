# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

import pytest

pytest.importorskip("torch_npu")

from xllm.python.models.glm5_2 import Glm52Config


def test_config_defaults_parallel_fields() -> None:
    cfg = Glm52Config.from_dict({})

    assert cfg.tp_size == 1
    assert cfg.tp_rank == 0
    assert cfg.ep_size == 1
    assert cfg.ep_rank == 0
    assert cfg.dp_size == 1
    assert cfg.dp_rank == 0
    assert cfg.moe_tp_size == 1
    assert cfg.moe_tp_rank == 0
    assert cfg.world_size == 1


def test_config_reads_parallel_fields() -> None:
    cfg = Glm52Config.from_dict(
        {
            "tp_size": 16,
            "tp_rank": 7,
            "ep_size": 1,
            "ep_rank": 0,
            "dp_size": 1,
            "dp_rank": 0,
            "moe_tp_size": 16,
            "moe_tp_rank": 7,
            "world_size": 16,
        }
    )

    assert cfg.tp_size == 16
    assert cfg.tp_rank == 7
    assert cfg.ep_size == 1
    assert cfg.ep_rank == 0
    assert cfg.dp_size == 1
    assert cfg.dp_rank == 0
    assert cfg.moe_tp_size == 16
    assert cfg.moe_tp_rank == 7
    assert cfg.world_size == 16
