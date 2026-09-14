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

from __future__ import annotations

from typing import Any

import tilelang.language as T

PASS_CONFIGS = {"tl.ascend_auto_sync": True}
MAX_ROWS = 1048576


def build_hisparse_store_kernel(head_dim: int, vec_core_num: int) -> Any:
    """Write new MLA rows to mapped Host slots, ignoring graph padding."""
    if head_dim not in (64, 512) or vec_core_num <= 0 or vec_core_num % 2:
        raise ValueError("HiSparse requires MLA dimensions 64/512 and an even vector core count")

    @T.prim_func
    def store(
        values: T.Tensor((MAX_ROWS, head_dim), "bfloat16"),
        slots: T.Tensor((MAX_ROWS,), "int32"),
        host: T.Tensor((MAX_ROWS, head_dim), "bfloat16"),
        rows: T.int32,
        host_rows: T.int32,
    ):
        with T.Kernel(vec_core_num // 2, is_npu=True) as (cid, vid):  # noqa: SIM117
            with T.Scope("V"):
                row_data = T.alloc_ub((head_dim,), "bfloat16")
                for iteration in T.serial((rows + vec_core_num - 1) // vec_core_num):
                    row = iteration * vec_core_num + cid * 2 + vid
                    if row < rows:
                        slot = slots[row]
                        if slot >= 0 and slot < host_rows:
                            T.copy(values[row, 0:head_dim], row_data)
                            T.copy(row_data, host[slot, 0:head_dim])

    return store
