# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

setup(
    name="xllm-dense-block-table",
    version="0.1.0",
    py_modules=["solution", "dense_kernels"],
    ext_modules=[
        NpuExtension(
            name="_dense_block_table_native",
            sources=["native.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
