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

"""NPU dense projections that honor the current FlashComm1 token layout."""

from __future__ import annotations

import torch

from xllm.python.layers.flash_comm1 import get_flash_comm1_context
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.linear import RowParallelLinear


class FlashComm1RowParallelLinear(RowParallelLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_flash_comm1_context()
        if context is None:
            return super().forward(x)
        if not self.reduce_results or self.tp_size != context.tp_size:
            raise ValueError("FlashComm1 requires a reducing projection on the same TP group")
        use_mmrs = (
            context.enable_mmrs_fusion
            and self.tp_size in (2, 4, 8)
            and x.ndim == 2
            and 256 <= x.shape[1] < 65535
            and x.dtype in (torch.float16, torch.bfloat16)
            and self.weight.dtype == x.dtype
        )
        if use_mmrs:
            # The operator accepts a transposed right operand. Preserve the
            # checkpoint storage and avoid caching another full weight copy.
            weight = self.weight if self._weight_is_transposed else self.weight.transpose(0, 1)
            output = context.matmul_reduce_scatter(x, weight)
        else:
            output = context.reduce_scatter(self._forward_matmul(x))
        # MMRS does not support a nonzero fused bias. Add the replicated bias
        # once, after the reduction, including on padded rank-local rows.
        return output if self.bias is None else output + self.bias


class FlashComm1GatedMLP(GatedMLP):
    row_parallel_cls = FlashComm1RowParallelLinear

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_flash_comm1_context()
        if context is not None:
            hidden_states = context.gather(hidden_states)
        return super().forward(hidden_states)
