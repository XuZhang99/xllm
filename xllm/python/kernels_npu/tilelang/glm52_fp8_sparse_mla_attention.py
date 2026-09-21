#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING

import tilelang.language as T

if TYPE_CHECKING:
    from tvm.tir import PrimFunc

from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS

LATENT_DIM = 512
ROPE_DIM = 64
TOPK = 2048
BLOCK_SIZE = 128
CORE_NUM = 24
HEAD_TILE = 16
KV_TILE = 64
VEC_NUM = 2
VEC_HEAD_TILE = HEAD_TILE // VEC_NUM
VEC_KV_TILE = KV_TILE // VEC_NUM
KV_COPY_ROWS = 32
DECODE_ROWS = 4
PIPELINE_STAGES = 2
MAX_NUM_QUERIES = 1024
MAX_CACHE_BLOCKS = 32768
MAX_BLOCK_TABLE_LEN = 32768
DEFAULT_DTYPE = "bf16"
SUPPORTED_NUM_HEADS = (4, 8, 16)
SUPPORTED_NUM_SPLITS = (1, 2, 4, 8, 16)


def build_glm52_fp8_sparse_mla_attention_kernel(num_heads: int, num_splits: int = 1) -> "PrimFunc":
    if num_heads not in SUPPORTED_NUM_HEADS:
        raise ValueError(
            f"GLM-5.2 FP8 sparse MLA attention only supports num_heads in {SUPPORTED_NUM_HEADS}, got {num_heads}"
        )

    if num_splits not in SUPPORTED_NUM_SPLITS:
        raise ValueError(f"unsupported FP8 MLA split count: {num_splits}")
    num_kv_tiles = TOPK // KV_TILE // num_splits
    input_dtype = "bfloat16"
    accum_dtype = "float32"
    output_dtype = input_dtype if num_splits == 1 else accum_dtype
    index_dtype = "int32"

    @T.prim_func
    def glm52_fp8_sparse_mla_attention_kernel(
        q_latent: T.Tensor((1, MAX_NUM_QUERIES * num_heads * LATENT_DIM), input_dtype),
        q_rope: T.Tensor((1, MAX_NUM_QUERIES * num_heads * ROPE_DIM), input_dtype),
        nope_cache: T.Tensor((MAX_CACHE_BLOCKS, BLOCK_SIZE, LATENT_DIM), "uint8"),
        rope_cache: T.Tensor((MAX_CACHE_BLOCKS, BLOCK_SIZE, ROPE_DIM), "uint8"),
        topk_indices: T.Tensor((MAX_NUM_QUERIES, TOPK), index_dtype),
        block_table: T.Tensor((1, MAX_NUM_QUERIES * MAX_BLOCK_TABLE_LEN), index_dtype),
        actual_seq_lengths_kv: T.Tensor((MAX_NUM_QUERIES,), index_dtype),
        e4m3_decode_table: T.Tensor((256,), accum_dtype),
        output: T.Tensor((MAX_NUM_QUERIES * num_splits, num_heads, LATENT_DIM), output_dtype),
        split_stats: T.Tensor((MAX_NUM_QUERIES * num_splits, 2, HEAD_TILE), accum_dtype),
        workspace_k: T.Tensor((CORE_NUM, KV_TILE, LATENT_DIM), input_dtype),
        workspace_k_rope: T.Tensor((CORE_NUM, KV_TILE, ROPE_DIM), input_dtype),
        workspace_scores: T.Tensor((CORE_NUM, HEAD_TILE, KV_TILE), accum_dtype),
        workspace_probs: T.Tensor((CORE_NUM, HEAD_TILE, KV_TILE), input_dtype),
        workspace_output: T.Tensor((CORE_NUM, HEAD_TILE, LATENT_DIM), accum_dtype),
        workspace_q: T.Tensor((CORE_NUM, HEAD_TILE, LATENT_DIM), input_dtype),
        workspace_q_rope: T.Tensor((CORE_NUM, HEAD_TILE, ROPE_DIM), input_dtype),
        q_token_stride: T.int32,
        q_head_stride: T.int32,
        q_rope_token_stride: T.int32,
        q_rope_head_stride: T.int32,
        num_queries: T.int32,
        block_table_stride: T.int32,
        softmax_scale: T.float32,
    ):
        with T.Kernel(CORE_NUM, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1((HEAD_TILE, LATENT_DIM), input_dtype)
            q_rope_l1 = T.alloc_L1((HEAD_TILE, ROPE_DIM), input_dtype)
            k_l1 = T.alloc_L1((KV_TILE, LATENT_DIM), input_dtype)
            k_rope_l1 = T.alloc_L1((KV_TILE, ROPE_DIM), input_dtype)
            probs_l1 = T.alloc_L1((HEAD_TILE, KV_TILE), input_dtype)
            scores_l0c = T.alloc_L0C((HEAD_TILE, KV_TILE), accum_dtype)
            output_l0c = T.alloc_L0C((HEAD_TILE, LATENT_DIM), accum_dtype)

            indices_ub = T.alloc_ub((KV_TILE,), index_dtype)
            indices_fp32_ub = T.alloc_ub((KV_TILE,), accum_dtype)
            valid_mask_ub = T.alloc_ub((32,), "uint8")
            nonnegative_mask_ub = T.alloc_ub((32,), "uint8")
            valid_index_max_ub = T.alloc_ub((1,), accum_dtype)
            decode_table_ub = T.alloc_ub((256,), accum_dtype)
            # Older Gather uses the source extent as its output count. Only
            # the first 256 entries are addressed; pad the allocation so one
            # call can decode the entire four-row latent tile safely.
            decode_table_bf16_ub = T.alloc_ub((DECODE_ROWS * LATENT_DIM,), input_dtype)
            decoded_bf16_ub = T.alloc_ub((DECODE_ROWS, LATENT_DIM), input_dtype)
            decoded_rope_bf16_ub = T.alloc_ub((DECODE_ROWS, ROPE_DIM), input_dtype)
            raw_cache_ub = T.alloc_ub((DECODE_ROWS, LATENT_DIM), "uint8")
            raw_rope_cache_ub = T.alloc_ub((DECODE_ROWS, ROPE_DIM), "uint8")
            decode_fp16_ub = T.alloc_ub((DECODE_ROWS * LATENT_DIM,), "float16")
            decode_offsets_i32_ub = T.alloc_ub((DECODE_ROWS * LATENT_DIM,), index_dtype)
            decode_offsets_u32_ub = T.alloc_ub((DECODE_ROWS * LATENT_DIM,), "uint32")
            rope_decode_fp16_ub = T.alloc_ub((DECODE_ROWS * ROPE_DIM,), "float16")
            rope_decode_offsets_i32_ub = T.alloc_ub((DECODE_ROWS * ROPE_DIM,), index_dtype)
            rope_decode_offsets_u32_ub = T.alloc_ub((DECODE_ROWS * ROPE_DIM,), "uint32")
            k_gather_ub = T.alloc_ub((2, KV_COPY_ROWS, LATENT_DIM), input_dtype)
            k_rope_gather_ub = T.alloc_ub((2, KV_COPY_ROWS, ROPE_DIM), input_dtype)
            q_gather_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), input_dtype)
            q_rope_gather_ub = T.alloc_ub((VEC_HEAD_TILE, ROPE_DIM), input_dtype)

            score_max_ub = T.alloc_ub((VEC_HEAD_TILE, 1), accum_dtype)
            previous_score_max_ub = T.alloc_ub((VEC_HEAD_TILE, 1), accum_dtype)
            scores_ub = T.alloc_ub((VEC_HEAD_TILE, KV_TILE), accum_dtype)
            score_max_broadcast_ub = T.alloc_ub((VEC_HEAD_TILE, KV_TILE), accum_dtype)
            output_scale_broadcast_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), accum_dtype)
            score_sum_ub = T.alloc_ub((VEC_HEAD_TILE, 1), accum_dtype)
            normalizer_ub = T.alloc_ub((VEC_HEAD_TILE, 1), accum_dtype)
            probs_bf16_ub = T.alloc_ub((VEC_HEAD_TILE, KV_TILE), input_dtype)
            partial_output_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), accum_dtype)
            accumulated_output_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), accum_dtype)
            normalizer_broadcast_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), accum_dtype)
            output_bf16_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), input_dtype)

            num_tasks = num_queries * num_splits
            tasks_per_core = (num_tasks + CORE_NUM - 1) // CORE_NUM
            task_start = cid * tasks_per_core
            task_end = T.if_then_else(
                task_start + tasks_per_core < num_tasks,
                task_start + tasks_per_core,
                num_tasks,
            )

            if cid < num_tasks:
                T.copy(e4m3_decode_table, decode_table_ub)
                T.set_flag("mte2", "v", 4)
                T.wait_flag("mte2", "v", 4)
                # E4M3 values are exactly representable in BF16. Decode directly
                # into the gather tile instead of casting every decoded row.
                T.tile.cast(decode_table_bf16_ub, decode_table_ub, "CAST_RINT", 256)
                for task_idx in T.serial(task_start, task_end):
                    query_idx = task_idx // num_splits
                    split_idx = task_idx % num_splits
                    T.tile.fill(q_gather_ub, 0.0)
                    T.tile.fill(q_rope_gather_ub, 0.0)
                    T.set_flag("v", "mte2", 8)
                    T.wait_flag("v", "mte2", 8)
                    if num_heads <= VEC_HEAD_TILE:
                        if vid == 0:
                            for head_idx in range(num_heads):
                                T.copy(
                                    q_latent[
                                        0,
                                        query_idx * q_token_stride + head_idx * q_head_stride : query_idx
                                        * q_token_stride
                                        + head_idx * q_head_stride
                                        + LATENT_DIM,
                                    ],
                                    q_gather_ub[head_idx, :],
                                )
                                T.copy(
                                    q_rope[
                                        0,
                                        query_idx * q_rope_token_stride + head_idx * q_rope_head_stride : query_idx
                                        * q_rope_token_stride
                                        + head_idx * q_rope_head_stride
                                        + ROPE_DIM,
                                    ],
                                    q_rope_gather_ub[head_idx, :],
                                )
                    else:
                        for head_idx in range(VEC_HEAD_TILE):
                            global_head_idx = vid * VEC_HEAD_TILE + head_idx
                            T.copy(
                                q_latent[
                                    0,
                                    query_idx * q_token_stride + global_head_idx * q_head_stride : query_idx
                                    * q_token_stride
                                    + global_head_idx * q_head_stride
                                    + LATENT_DIM,
                                ],
                                q_gather_ub[head_idx, :],
                            )
                            T.copy(
                                q_rope[
                                    0,
                                    query_idx * q_rope_token_stride + global_head_idx * q_rope_head_stride : query_idx
                                    * q_rope_token_stride
                                    + global_head_idx * q_rope_head_stride
                                    + ROPE_DIM,
                                ],
                                q_rope_gather_ub[head_idx, :],
                            )
                    T.set_flag("mte2", "mte3", 6)
                    T.wait_flag("mte2", "mte3", 6)
                    T.copy(
                        q_gather_ub,
                        workspace_q[
                            cid,
                            vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                            :,
                        ],
                    )
                    T.copy(
                        q_rope_gather_ub,
                        workspace_q_rope[
                            cid,
                            vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                            :,
                        ],
                    )

                    T.copy(workspace_q[cid, :, :], q_l1)
                    T.copy(workspace_q_rope[cid, :, :], q_rope_l1)
                    T.set_flag("mte2", "mte1", 7)
                    T.wait_flag("mte2", "mte1", 7)

                    actual_kv_len = actual_seq_lengths_kv[query_idx]
                    T.tile.fill(accumulated_output_ub, 0.0)
                    T.tile.fill(normalizer_ub, 0.0)
                    T.tile.fill(score_max_ub, 2.0**30)

                    # CV pipelining expands the five GM intermediates below by
                    # PIPELINE_STAGES; the runtime and wrapper reserve both slots.
                    for tile_idx in T.Pipelined(num_kv_tiles, num_stages=PIPELINE_STAGES):
                        T.copy(
                            topk_indices[
                                query_idx,
                                (split_idx * num_kv_tiles + tile_idx) * KV_TILE : (
                                    split_idx * num_kv_tiles + tile_idx + 1
                                )
                                * KV_TILE,
                            ],
                            indices_ub,
                        )
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.copy(indices_ub, indices_fp32_ub)
                        T.pipe_barrier("v")
                        T.tile.compare(
                            valid_mask_ub,
                            indices_fp32_ub,
                            T.float32(actual_kv_len - 1),
                            "LE",
                        )
                        T.tile.compare(
                            nonnegative_mask_ub,
                            indices_fp32_ub,
                            T.float32(0.0),
                            "GE",
                        )
                        T.tile.bitwise_and(
                            valid_mask_ub,
                            valid_mask_ub,
                            nonnegative_mask_ub,
                        )

                        # Short contexts contain fully padded top-k tiles.
                        # Inspect the actual mask instead of assuming valid
                        # indices occupy a contiguous prefix.
                        # A dynamic serial loop preserves the guard through
                        # CrossCorePipeline, which hoists vector-only if bodies.
                        for _short_context in T.serial(T.if_then_else(actual_kv_len < TOPK, 1, 0)):
                            T.tile.select(
                                indices_fp32_ub,
                                valid_mask_ub,
                                indices_fp32_ub,
                                -1.0,
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                            T.pipe_barrier("v")
                            T.reduce_max(indices_fp32_ub, valid_index_max_ub, dim=0)
                            T.set_flag("v", "s", 3)
                            T.wait_flag("v", "s", 3)

                        # As in the paged DSA kernel, gather a group of sparse
                        # rows with MTE2 before handing a contiguous tile to MTE3.
                        # Amortize FP8 conversion and synchronization over four
                        # rows while bounding the temporary offset buffers in UB.
                        for copy_group in T.serial(VEC_KV_TILE // KV_COPY_ROWS):
                            global_copy_group = tile_idx * (VEC_KV_TILE // KV_COPY_ROWS) + copy_group
                            ping_pong = global_copy_group % 2
                            # Wait before overwriting the previous MTE3 source.
                            if global_copy_group > 1:
                                T.wait_flag("mte3", "v", ping_pong)
                            if actual_kv_len >= TOPK or valid_index_max_ub[0] >= 0.0:
                                for decode_group in T.serial(KV_COPY_ROWS // DECODE_ROWS):
                                    row_start = decode_group * DECODE_ROWS
                                    for copy_row in T.serial(DECODE_ROWS):
                                        index_in_tile = (
                                            vid * VEC_KV_TILE + copy_group * KV_COPY_ROWS + row_start + copy_row
                                        )
                                        sparse_index = indices_ub[index_in_tile]
                                        safe_sparse_index = T.if_then_else(
                                            sparse_index >= 0,
                                            T.if_then_else(sparse_index < actual_kv_len, sparse_index, 0),
                                            0,
                                        )
                                        logical_block = safe_sparse_index // BLOCK_SIZE
                                        physical_block = block_table[0, query_idx * block_table_stride + logical_block]
                                        physical_block = T.if_then_else(physical_block >= 0, physical_block, 0)
                                        block_offset = safe_sparse_index % BLOCK_SIZE
                                        T.copy(nope_cache[physical_block, block_offset, :], raw_cache_ub[copy_row, :])
                                        T.copy(
                                            rope_cache[physical_block, block_offset, :], raw_rope_cache_ub[copy_row, :]
                                        )
                                    T.set_flag("mte2", "v", 6)
                                    T.wait_flag("mte2", "v", 6)
                                    T.tile.cast(
                                        decode_fp16_ub,
                                        raw_cache_ub,
                                        "CAST_NONE",
                                        DECODE_ROWS * LATENT_DIM,
                                    )
                                    T.tile.cast(
                                        decode_offsets_i32_ub, decode_fp16_ub, "CAST_RINT", DECODE_ROWS * LATENT_DIM
                                    )
                                    T.pipe_barrier("v")
                                    T.tile.mul(decode_offsets_i32_ub, decode_offsets_i32_ub, 2)
                                    T.pipe_barrier("v")
                                    T.reinterpretcast(decode_offsets_u32_ub, decode_offsets_i32_ub, "uint32_t")
                                    T.tile.gather(decoded_bf16_ub, decode_table_bf16_ub, decode_offsets_u32_ub, 0)
                                    T.copy(
                                        decoded_bf16_ub, k_gather_ub[ping_pong, row_start : row_start + DECODE_ROWS, :]
                                    )
                                    T.pipe_barrier("v")
                                    T.tile.cast(
                                        rope_decode_fp16_ub,
                                        raw_rope_cache_ub,
                                        "CAST_NONE",
                                        DECODE_ROWS * ROPE_DIM,
                                    )
                                    T.tile.cast(
                                        rope_decode_offsets_i32_ub,
                                        rope_decode_fp16_ub,
                                        "CAST_RINT",
                                        DECODE_ROWS * ROPE_DIM,
                                    )
                                    T.pipe_barrier("v")
                                    T.tile.mul(
                                        rope_decode_offsets_i32_ub,
                                        rope_decode_offsets_i32_ub,
                                        2,
                                    )
                                    T.pipe_barrier("v")
                                    T.reinterpretcast(
                                        rope_decode_offsets_u32_ub, rope_decode_offsets_i32_ub, "uint32_t"
                                    )
                                    T.tile.gather(
                                        decoded_rope_bf16_ub,
                                        decode_table_bf16_ub[: DECODE_ROWS * ROPE_DIM],
                                        rope_decode_offsets_u32_ub,
                                        0,
                                    )
                                    T.copy(
                                        decoded_rope_bf16_ub,
                                        k_rope_gather_ub[ping_pong, row_start : row_start + DECODE_ROWS, :],
                                    )
                            else:
                                T.tile.fill(k_gather_ub[ping_pong, :, :], 0.0)
                                T.tile.fill(k_rope_gather_ub[ping_pong, :, :], 0.0)
                            T.set_flag("v", "mte3", ping_pong)
                            T.wait_flag("v", "mte3", ping_pong)
                            T.copy(
                                k_gather_ub[ping_pong, :, :],
                                workspace_k[
                                    cid,
                                    vid * VEC_KV_TILE + copy_group * KV_COPY_ROWS : vid * VEC_KV_TILE
                                    + (copy_group + 1) * KV_COPY_ROWS,
                                    :,
                                ],
                            )
                            T.copy(
                                k_rope_gather_ub[ping_pong, :, :],
                                workspace_k_rope[
                                    cid,
                                    vid * VEC_KV_TILE + copy_group * KV_COPY_ROWS : vid * VEC_KV_TILE
                                    + (copy_group + 1) * KV_COPY_ROWS,
                                    :,
                                ],
                            )
                            if global_copy_group < num_kv_tiles * (VEC_KV_TILE // KV_COPY_ROWS) - 2:
                                T.set_flag("mte3", "v", ping_pong)

                        T.copy(workspace_k[cid, :, :], k_l1)
                        T.copy(workspace_k_rope[cid, :, :], k_rope_l1)
                        T.set_flag("mte2", "mte1", 1)
                        T.wait_flag("mte2", "mte1", 1)
                        T.gemm_v0(
                            q_l1,
                            k_l1,
                            scores_l0c,
                            transpose_B=True,
                            init=True,
                        )
                        T.gemm_v0(
                            q_rope_l1,
                            k_rope_l1,
                            scores_l0c,
                            transpose_B=True,
                        )
                        T.set_flag("m", "fix", 2)
                        T.wait_flag("m", "fix", 2)
                        T.copy(scores_l0c, workspace_scores[cid, :, :])

                        T.copy(score_max_ub, previous_score_max_ub)
                        T.copy(
                            workspace_scores[
                                cid,
                                vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                                :,
                            ],
                            scores_ub,
                        )
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        for head_idx in T.serial(VEC_HEAD_TILE):
                            T.tile.select(
                                scores_ub[head_idx, :],
                                valid_mask_ub,
                                scores_ub[head_idx, :],
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                        T.pipe_barrier("v")
                        T.reduce_max(scores_ub, score_max_ub, dim=-1)
                        T.pipe_barrier("v")
                        T.tile.mul(score_max_ub, score_max_ub, -softmax_scale)
                        T.pipe_barrier("v")
                        T.tile.min(
                            score_max_ub,
                            score_max_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.broadcast(score_max_broadcast_ub, score_max_ub)
                        T.pipe_barrier("v")
                        T.tile.axpy(
                            score_max_broadcast_ub,
                            scores_ub,
                            softmax_scale,
                        )
                        T.pipe_barrier("v")
                        T.tile.exp(scores_ub, score_max_broadcast_ub)
                        T.pipe_barrier("v")
                        T.tile.sub(
                            previous_score_max_ub,
                            score_max_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.exp(
                            previous_score_max_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.copy(scores_ub, probs_bf16_ub)
                        T.pipe_barrier("v")
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(
                            probs_bf16_ub,
                            workspace_probs[
                                cid,
                                vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                                :,
                            ],
                        )

                        T.copy(workspace_probs[cid, :, :], probs_l1)
                        T.set_flag("mte2", "mte1", 3)
                        T.wait_flag("mte2", "mte1", 3)
                        T.gemm_v0(
                            probs_l1,
                            k_l1,
                            output_l0c,
                            init=True,
                        )
                        T.set_flag("m", "fix", 4)
                        T.wait_flag("m", "fix", 4)
                        T.copy(output_l0c, workspace_output[cid, :, :])

                        T.copy(
                            workspace_output[
                                cid,
                                vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                                :,
                            ],
                            partial_output_ub,
                        )
                        T.set_flag("mte2", "v", 2)
                        T.wait_flag("mte2", "v", 2)
                        T.reduce_sum(scores_ub, score_sum_ub, dim=-1)
                        T.pipe_barrier("v")
                        T.tile.mul(
                            normalizer_ub,
                            normalizer_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.add(normalizer_ub, normalizer_ub, score_sum_ub)
                        T.pipe_barrier("v")
                        T.tile.broadcast(
                            output_scale_broadcast_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.mul(
                            accumulated_output_ub,
                            accumulated_output_ub,
                            output_scale_broadcast_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.add(
                            accumulated_output_ub,
                            accumulated_output_ub,
                            partial_output_ub,
                        )

                    if num_splits == 1:
                        T.tile.max(normalizer_ub, normalizer_ub, 1.0)
                        T.tile.broadcast(normalizer_broadcast_ub, normalizer_ub)
                        T.pipe_barrier("v")
                        T.tile.div(
                            accumulated_output_ub,
                            accumulated_output_ub,
                            normalizer_broadcast_ub,
                        )
                        T.pipe_barrier("v")
                        T.copy(accumulated_output_ub, output_bf16_ub)
                        T.set_flag("v", "mte3", 9)
                        T.wait_flag("v", "mte3", 9)
                        if num_heads <= VEC_HEAD_TILE:
                            if vid == 0:
                                T.copy(
                                    output_bf16_ub[0:num_heads, :],
                                    output[task_idx, 0:num_heads, :],
                                )
                        else:
                            T.copy(
                                output_bf16_ub,
                                output[task_idx, vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE, :],
                            )
                    else:
                        # Keep the unnormalized numerator in FP32 until all
                        # KV shards are merged on the same execution stream.
                        T.set_flag("v", "mte3", 9)
                        T.wait_flag("v", "mte3", 9)
                        T.copy(
                            score_max_ub[:, 0],
                            split_stats[task_idx, 0, vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE],
                        )
                        T.copy(
                            normalizer_ub[:, 0],
                            split_stats[task_idx, 1, vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE],
                        )
                        if num_heads <= VEC_HEAD_TILE:
                            if vid == 0:
                                T.copy(
                                    accumulated_output_ub[0:num_heads, :],
                                    output[task_idx, 0:num_heads, :],
                                )
                        else:
                            T.copy(
                                accumulated_output_ub,
                                output[task_idx, vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE, :],
                            )

    return glm52_fp8_sparse_mla_attention_kernel


def build_glm52_fp8_sparse_mla_merge_kernel(num_heads: int, num_splits: int) -> "PrimFunc":
    if num_heads not in SUPPORTED_NUM_HEADS or num_splits not in SUPPORTED_NUM_SPLITS[1:]:
        raise ValueError(f"unsupported FP8 MLA merge shape: heads={num_heads}, splits={num_splits}")

    @T.prim_func
    def glm52_fp8_sparse_mla_merge_kernel(
        partial: T.Tensor((MAX_NUM_QUERIES * num_splits, num_heads, LATENT_DIM), "float32"),
        stats: T.Tensor((MAX_NUM_QUERIES * num_splits, 2, HEAD_TILE), "float32"),
        output: T.Tensor((MAX_NUM_QUERIES, num_heads, LATENT_DIM), "bfloat16"),
        num_queries: T.int32,
    ):
        with T.Kernel(CORE_NUM, is_npu=True) as (cid, vid):  # noqa: SIM117 - separate TileLang scopes
            with T.Scope("V"):
                partial_ub = T.alloc_ub((num_splits, LATENT_DIM), "float32")
                stats_ub = T.alloc_ub((num_splits, 2, HEAD_TILE), "float32")
                negative_max = T.alloc_ub((64,), "float32")
                denominator = T.alloc_ub((64,), "float32")
                weights = T.alloc_ub((64,), "float32")
                temporary = T.alloc_ub((64,), "float32")
                valid_mask = T.alloc_ub((32,), "uint8")
                minimum = T.alloc_ub((1,), "float32")
                total = T.alloc_ub((1,), "float32")
                accumulated = T.alloc_ub((LATENT_DIM,), "float32")
                result = T.alloc_ub((LATENT_DIM,), "bfloat16")
                rows_per_task = (num_queries * num_heads + 47) // 48
                row_start = (cid * 2 + vid) * rows_per_task
                row_end = T.if_then_else(
                    row_start + rows_per_task < num_queries * num_heads,
                    row_start + rows_per_task,
                    num_queries * num_heads,
                )
                for row in T.serial(row_start, row_end):
                    T.barrier_all()
                    query_idx = row // num_heads
                    head_idx = row % num_heads
                    for split_idx in T.serial(num_splits):
                        T.copy(partial[query_idx * num_splits + split_idx, head_idx, :], partial_ub[split_idx, :])
                        T.copy(stats[query_idx * num_splits + split_idx, :, :], stats_ub[split_idx, :, :])
                    T.tile.fill(negative_max, 3.402823466e38)
                    T.tile.fill(denominator, 0.0)
                    T.barrier_all()
                    for split_idx in T.serial(num_splits):
                        negative_max[split_idx] = stats_ub[split_idx, 0, head_idx]
                        denominator[split_idx] = stats_ub[split_idx, 1, head_idx]
                    T.barrier_all()

                    # Empty shards must not contribute, including when every
                    # shard is empty on a padded ACLGraph replay row.
                    T.tile.compare(valid_mask, denominator, 0.0, "GT")
                    T.pipe_barrier("v")
                    T.tile.select(negative_max, valid_mask, negative_max, 3.402823466e38, "VSEL_TENSOR_SCALAR_MODE")
                    T.pipe_barrier("v")
                    T.reduce_min(negative_max, minimum, dim=0)
                    T.pipe_barrier("v")
                    T.tile.broadcast(temporary, minimum, axis=0)
                    T.pipe_barrier("v")
                    T.tile.sub(weights, temporary, negative_max)
                    T.pipe_barrier("v")
                    T.tile.exp(weights, weights)
                    T.pipe_barrier("v")
                    T.tile.select(weights, valid_mask, weights, 0.0, "VSEL_TENSOR_SCALAR_MODE")
                    T.pipe_barrier("v")
                    T.tile.mul(temporary, weights, denominator)
                    T.pipe_barrier("v")
                    T.reduce_sum(temporary, total, dim=0)
                    T.pipe_barrier("v")
                    T.tile.max(total, total, 1.0)
                    T.pipe_barrier("v")
                    T.tile.broadcast(temporary, total, axis=0)
                    T.pipe_barrier("v")
                    T.tile.div(weights, weights, temporary)
                    T.tile.fill(accumulated, 0.0)
                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    for split_idx in T.serial(num_splits):
                        if weights[split_idx] > 0.0:
                            T.tile.axpy(accumulated, partial_ub[split_idx, :], weights[split_idx])
                            T.pipe_barrier("v")
                    T.tile.cast(result, accumulated, "CAST_RINT", LATENT_DIM)
                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.copy(result, output[query_idx, head_idx, :])
                    T.barrier_all()

    return glm52_fp8_sparse_mla_merge_kernel
