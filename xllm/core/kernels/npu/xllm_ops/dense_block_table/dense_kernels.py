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

"""Device kernels for the two dense block table execution paths."""

import triton
import triton.language as tl


@triton.jit
def lookup_in_ub(
    logical_table: tl.tensor,
    mapping: tl.tensor,
    output: tl.tensor,
    table_size: tl.constexpr,
    blocks: tl.constexpr,
    mapping_tile: tl.constexpr,
    vector_tile: tl.constexpr,
    shards: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    layer = program // shards
    mapping_lane = tl.arange(0, mapping_tile)
    table_lane = tl.arange(0, vector_tile)
    pages = tl.load(mapping + layer * blocks + mapping_lane, mapping_lane < blocks, other=0)
    # Older Ascend compilers expose 32-bit gather only through float32.
    # Reinterpret the lanes without converting or computing on the values.
    page_bits = pages.to(tl.float32, bitcast=True)
    for tile in range(program % shards, triton.cdiv(table_size, vector_tile), shards):
        offset = tile * vector_tile + table_lane
        logical = tl.load(logical_table + offset, offset < table_size, other=-1)
        physical = tl.gather(page_bits, tl.maximum(logical, 0), axis=0).to(tl.int32, bitcast=True)
        tl.store(
            output + layer * table_size + offset,
            tl.where(logical < 0, -1, physical),
            offset < table_size,
        )


@triton.jit
def make_gather_indices(
    logical_table: tl.tensor,
    indices: tl.tensor,
    table_size: tl.constexpr,
    vector_tile: tl.constexpr,
) -> None:
    lane = tl.arange(0, vector_tile)
    for tile in range(tl.program_id(0), triton.cdiv(table_size, vector_tile), tl.num_programs(0)):
        offset = tile * vector_tile + lane
        active = offset < table_size
        logical = tl.load(logical_table + offset, active, other=0)
        tl.store(indices + offset, tl.maximum(logical, 0).to(tl.int64), active)


@triton.jit
def restore_padding(
    logical_table: tl.tensor,
    output: tl.tensor,
    table_size: tl.constexpr,
    output_size: tl.constexpr,
    vector_tile: tl.constexpr,
) -> None:
    lane = tl.arange(0, vector_tile)
    for tile in range(tl.program_id(0), triton.cdiv(output_size, vector_tile), tl.num_programs(0)):
        offset = tile * vector_tile + lane
        active = offset < output_size
        logical = tl.load(logical_table + offset % table_size, active, other=0)
        physical = tl.load(output + offset, active, other=0)
        tl.store(output + offset, tl.where(logical < 0, -1, physical), active)
