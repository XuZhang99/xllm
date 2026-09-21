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

"""Encode E4M3 and scatter the latent and RoPE rows in one device launch."""

from typing import TYPE_CHECKING

import tilelang.language as T

if TYPE_CHECKING:
    from tvm.tir import PrimFunc

NUM_ROWS = T.symbolic("num_rows")
LATENT_STRIDE = T.symbolic("latent_stride")
ROPE_STRIDE = T.symbolic("rope_stride")
CACHE_ROWS = T.symbolic("cache_rows")
LATENT_DIM = 512
ROPE_DIM = 64
ROW_DIM = LATENT_DIM + ROPE_DIM
SUPPORTED_DTYPES = ("bf16", "float16", "float32")


def build_fp8_mla_cache_write_kernel(dtype: str, slot_bytes: int) -> "PrimFunc":
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported FP8 MLA cache input dtype: {dtype}")
    if slot_bytes not in (4, 8):
        raise ValueError(f"unsupported FP8 MLA slot size: {slot_bytes}")
    input_dtype = "bfloat16" if dtype == "bf16" else dtype
    slot_dtype = "int64" if slot_bytes == 8 else "int32"

    @T.prim_func
    def fp8_mla_cache_write(
        slots: T.Tensor((NUM_ROWS,), slot_dtype),
        latent: T.Tensor((NUM_ROWS, LATENT_STRIDE), input_dtype),
        rope: T.Tensor((NUM_ROWS, ROPE_STRIDE), input_dtype),
        latent_cache: T.Tensor((CACHE_ROWS, LATENT_DIM), "uint8"),
        rope_cache: T.Tensor((CACHE_ROWS, ROPE_DIM), "uint8"),
    ) -> None:
        with T.Kernel(24, is_npu=True) as (cid, vid):
            task_id = cid * 2 + vid
            rows_per_task = (NUM_ROWS + 47) // 48
            row_start = task_id * rows_per_task
            row_end = T.min(row_start + rows_per_task, NUM_ROWS)
            with T.Scope("V"):
                input_ub = T.alloc_ub((ROW_DIM,), input_dtype)
                working = T.alloc_ub((ROW_DIM,), "float32")
                absolute = T.alloc_ub((ROW_DIM,), "float32")
                subnormal = T.alloc_ub((ROW_DIM,), "float32")
                sign = T.alloc_ub((ROW_DIM,), "float32")
                bits = T.alloc_ub((ROW_DIM,), "int32")
                normal_bits = T.alloc_ub((ROW_DIM,), "int32")
                half_bits = T.alloc_ub((ROW_DIM,), "int32")
                odd = T.alloc_ub((ROW_DIM,), "int32")
                subnormal_bits = T.alloc_ub((ROW_DIM,), "int32")
                sign_mask = T.alloc_ub((96,), "uint8")
                value_mask = T.alloc_ub((96,), "uint8")
                half_output = T.alloc_ub((ROW_DIM,), "float16")
                signed_output = T.alloc_ub((ROW_DIM,), "int8")
                encoded = T.alloc_ub((ROW_DIM,), "uint8")
                for row in T.serial(row_start, row_end):
                    slot = T.Cast("int32", slots[row])
                    if slot >= 0:
                        T.copy(latent[row, 0:LATENT_DIM], input_ub[0:LATENT_DIM])
                        T.copy(rope[row, 0:ROPE_DIM], input_ub[LATENT_DIM:ROW_DIM])
                        if input_dtype == "float32":
                            T.copy(input_ub, working)
                        else:
                            T.tile.cast(working, input_ub, "CAST_NONE", ROW_DIM)
                        T.tile.compare(sign_mask, working, 0.0, "LT")
                        T.tile.abs(absolute, working)
                        T.tile.compare(value_mask, working, working, "EQ")
                        T.tile.select(absolute, value_mask, absolute, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.min(absolute, absolute, 448.0)

                        # Round the FP32 significand to three bits, ties to even.
                        # Reinterpret the complete UB vector, as in TileLang's
                        # per_block_cast_lossless_kernel scale-bit extraction.
                        # The sync pass does not track reinterpret aliases.
                        T.barrier_all()
                        T.reinterpretcast(bits, absolute, "int32_t")
                        T.tile.bitwise_rshift(normal_bits, bits, 20)
                        T.tile.bitwise_rshift(half_bits, normal_bits, 1)
                        T.tile.bitwise_lshift(half_bits, half_bits, 1)
                        T.tile.sub(odd, normal_bits, half_bits)
                        T.tile.add(normal_bits, bits, 524287)
                        T.tile.add(normal_bits, normal_bits, odd)
                        T.tile.bitwise_rshift(normal_bits, normal_bits, 20)
                        T.tile.add(normal_bits, normal_bits, -960)
                        T.tile.cast(working, normal_bits, "CAST_NONE", ROW_DIM)

                        T.tile.mul(subnormal, absolute, 512.0)
                        T.tile.cast(subnormal_bits, subnormal, "CAST_RINT", ROW_DIM)
                        T.tile.cast(subnormal, subnormal_bits, "CAST_NONE", ROW_DIM)
                        T.tile.compare(value_mask, absolute, 0.015625, "GE")
                        T.tile.select(working, value_mask, working, subnormal, "VSEL_TENSOR_TENSOR_MODE")

                        # A3 supports FP16 -> int8. Use the signed byte with the
                        # same bits instead of saturating codes 128..254 to 127.
                        # x < 0 deliberately maps negative zero and NaNs to +0.
                        T.tile.fill(sign, -128.0)
                        T.tile.select(sign, sign_mask, sign, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                        T.tile.add(working, working, sign)
                        T.tile.cast(half_output, working, "CAST_NONE", ROW_DIM)
                        T.tile.cast(signed_output, half_output, "CAST_RINT", ROW_DIM)
                        T.barrier_all()
                        T.reinterpretcast(encoded, signed_output, "uint8_t")
                        T.copy(encoded[0:LATENT_DIM], latent_cache[slot, :])
                        T.copy(encoded[LATENT_DIM:ROW_DIM], rope_cache[slot, :])

    return fp8_mla_cache_write
