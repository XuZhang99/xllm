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

"""Fused indexed Conv/SSM movement for BF16 GDN cache storage."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import get_vectorcore_num


@triton.jit
def _move_gdn_states_kernel(
    ssm_pool: tl.tensor,
    conv_pool: tl.tensor,
    ssm_work: tl.tensor,
    conv_work: tl.tensor,
    ssm_indices: tl.tensor,
    conv_indices: tl.tensor,
    compact_indices: tl.tensor,
    BATCH: tl.constexpr,
    SSM_ROW: tl.constexpr,
    CONV_ROW: tl.constexpr,
    CHECKPOINT_STRIDE: tl.constexpr,
    GATHER: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    tiles_per_row = TILES_PER_ROW
    offsets = tl.arange(0, BLOCK)
    for task in range(tl.program_id(0), BATCH * tiles_per_row, tl.num_programs(0)):
        row = (task // tiles_per_row).to(tl.int64)
        start = (task % tiles_per_row) * BLOCK
        columns = start + offsets
        state_slot = tl.load(ssm_indices + row).to(tl.int64)
        state_mask = (columns < SSM_ROW) & (state_slot >= 0)
        work_offsets = row * CHECKPOINT_STRIDE * SSM_ROW + columns
        pool_offsets = tl.maximum(state_slot, 0) * SSM_ROW + columns
        if GATHER:
            values = tl.load(ssm_pool + pool_offsets, mask=state_mask, other=0.0).to(tl.float32)
            tl.store(ssm_work + work_offsets, values, mask=columns < SSM_ROW)
        else:
            values = tl.load(ssm_work + work_offsets, mask=columns < SSM_ROW, other=0.0)
            values = values.to(tl.bfloat16, fp_downcast_rounding="rtne")
            tl.store(ssm_pool + pool_offsets, values, mask=state_mask)

        if start < CONV_ROW:
            conv_slot = tl.load(conv_indices + row).to(tl.int64)
            conv_mask = (columns < CONV_ROW) & (conv_slot >= 0)
            if GATHER:
                history = tl.load(conv_pool + tl.maximum(conv_slot, 0) * CONV_ROW + columns, mask=conv_mask, other=0.0)
                tl.store(conv_work + row * CONV_ROW + columns, history, mask=columns < CONV_ROW)
            else:
                history = tl.load(conv_work + row * CONV_ROW + columns, mask=columns < CONV_ROW, other=0.0)
                tl.store(conv_pool + tl.maximum(conv_slot, 0) * CONV_ROW + columns, history, mask=conv_mask)
        if GATHER and start == 0:
            conv_slot = tl.load(conv_indices + row)
            tl.store(compact_indices + row, tl.where(conv_slot >= 0, row, -1))
            tl.store(compact_indices + BATCH + row, row)
            state_row = row * CHECKPOINT_STRIDE
            tl.store(compact_indices + 2 * BATCH + row, tl.where(state_slot >= 0, state_row, -1))
            tl.store(compact_indices + 3 * BATCH + row, state_row)


def _move_states(
    state: torch.Tensor,
    conv: torch.Tensor,
    work_state: torch.Tensor,
    work_conv: torch.Tensor,
    state_indices: torch.Tensor,
    conv_indices: torch.Tensor,
    compact_indices: torch.Tensor,
    checkpoint_stride: int,
    *,
    gather: bool,
) -> None:
    batch = state_indices.numel()
    if not state.is_contiguous() or not conv.is_contiguous():
        raise ValueError("GDN state pools must be contiguous")
    state_row = state.numel() // state.shape[0]
    conv_row = conv.numel() // conv.shape[0]
    block = 4096
    vector_cores = get_vectorcore_num()
    if vector_cores <= 0:
        raise RuntimeError("GDN state movement requires the NPU vector-core count")
    tiles_per_row = triton.cdiv(max(state_row, conv_row), block)
    programs = min(batch * tiles_per_row, vector_cores)
    _move_gdn_states_kernel[(programs,)](
        state,
        conv,
        work_state,
        work_conv,
        state_indices,
        conv_indices,
        compact_indices,
        BATCH=batch,
        SSM_ROW=state_row,
        CONV_ROW=conv_row,
        CHECKPOINT_STRIDE=checkpoint_stride,
        GATHER=gather,
        TILES_PER_ROW=tiles_per_row,
        BLOCK=block,
        num_warps=4,
    )


def gather_gdn_states(
    state: torch.Tensor,
    conv: torch.Tensor,
    state_indices: torch.Tensor,
    conv_indices: torch.Tensor,
    checkpoint_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack active rows and their slot metadata, converting SSM storage to FP32."""
    batch = state_indices.numel()
    work_state = torch.empty(
        (batch * checkpoint_stride, *state.shape[1:]),
        dtype=torch.float32,
        device=state.device,
    )
    work_conv = torch.empty((batch, *conv.shape[1:]), dtype=conv.dtype, device=conv.device)
    indices = torch.empty((4, batch), dtype=torch.int32, device=state.device)
    _move_states(
        state, conv, work_state, work_conv, state_indices, conv_indices, indices, checkpoint_stride, gather=True
    )
    return work_state, work_conv, indices


def scatter_gdn_states(
    state: torch.Tensor,
    conv: torch.Tensor,
    work_state: torch.Tensor,
    work_conv: torch.Tensor,
    state_indices: torch.Tensor,
    conv_indices: torch.Tensor,
    checkpoint_stride: int,
) -> None:
    """Commit active rows with round-to-nearest BF16 SSM storage."""
    _move_states(
        state,
        conv,
        work_state,
        work_conv,
        state_indices,
        conv_indices,
        state_indices,
        checkpoint_stride,
        gather=False,
    )


__all__ = ["gather_gdn_states", "scatter_gdn_states"]
