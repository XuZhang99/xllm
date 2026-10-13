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

"""Native RMSNorm through torch_npu, without an xLLM host or package extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch_npu

from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT
from xllm_kernel.registry import register_kernel

if TYPE_CHECKING:
    import torch


_native_rms_norm = torch_npu.npu_rms_norm


@register_kernel(
    name="npu.xllm_native.rms_norm",
    contract=RMS_NORM_CONTRACT,
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="torch_npu.npu_rms_norm tensor domain; no shape specialization",
    solution="native",
    priority=10,
)
def _rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _native_rms_norm(input, weight, epsilon=eps)[0]
