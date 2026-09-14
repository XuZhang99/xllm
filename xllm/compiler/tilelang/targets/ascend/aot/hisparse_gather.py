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

import tilelang
from compiler.tilelang.common.spec import DispatchField, TilelangKernel, register_kernel

from xllm.python.kernels_npu.tilelang import hisparse_gather as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class HisparseGatherKernel(TilelangKernel):
    KERNEL_NAME = "hisparse_gather"
    DISPATCH_SCHEMA = [DispatchField("head_dim", "int32")]
    SPECIALIZATIONS = [{"variant_key": f"d{d}", "head_dim": d} for d in (64, 512)]

    @staticmethod
    def generate_source(head_dim: int) -> str:
        tilelang.disable_cache()
        program = kernel_impl.build_hisparse_gather_kernel(head_dim, tilelang_utils.detect_vec_core_num())
        with tilelang.tvm.transform.PassContext(opt_level=3, config=kernel_impl.PASS_CONFIGS):
            return tilelang.engine.lower(program).kernel_source
