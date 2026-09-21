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

import tilelang
from compiler.tilelang.common.spec import DispatchField, TilelangKernel, register_kernel

from xllm.python.kernels_npu.tilelang import glm52_fp8_sparse_mla_attention as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.glm52_fp8_sparse_mla_attention import (
    SUPPORTED_NUM_HEADS,
    SUPPORTED_NUM_SPLITS,
    build_glm52_fp8_sparse_mla_merge_kernel,
)
from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class Glm52Fp8SparseMlaMergeKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("num_heads", "int32"), DispatchField("num_splits", "int32")]
    SPECIALIZATIONS = [
        {"variant_key": f"h{num_heads}_s{num_splits}", "num_heads": num_heads, "num_splits": num_splits}
        for num_heads in SUPPORTED_NUM_HEADS
        for num_splits in SUPPORTED_NUM_SPLITS[1:]
    ]

    @staticmethod
    def generate_source(num_heads: int, num_splits: int) -> str:
        tilelang.disable_cache()
        kernel = build_glm52_fp8_sparse_mla_merge_kernel(num_heads, num_splits)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
