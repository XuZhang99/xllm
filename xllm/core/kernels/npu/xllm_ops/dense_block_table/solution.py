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

"""Translate dense logical block tables on Ascend."""

from collections.abc import Callable, Sequence
from typing import Protocol

import torch
import torch_npu
import triton
from triton.runtime.jit import JITFunction

# isort: split
from _dense_block_table_native import translate
from dense_kernels import lookup_in_ub, make_gather_indices, restore_padding

_MAX_INT32 = 2**31 - 1
_ACL_FORMAT_ND = 2
_MAX_UB_MAPPING_SIZE = 4096
_MAX_UB_OUTPUT_SIZE = 1 << 20
_VECTOR_TILE = 256
_MAX_PROGRAMS = 40


class Prepared(Protocol):
    def run(self) -> torch.Tensor: ...


def _compile(
    kernel: JITFunction,
    programs: int,
    *args: torch.Tensor | torch.dtype,
    **constants: int,
) -> Callable[..., None]:
    grid = (programs, 1, 1)
    return kernel.warmup(*args, grid=grid, **constants)[grid]


class _PreparedBlockTable:
    def __init__(self, logical_table: torch.Tensor, mapping: torch.Tensor) -> None:
        self.inputs = (logical_table, mapping)
        self.n = logical_table.numel()
        self._single = mapping.ndim == 1
        self._blocks = mapping.shape[-1]
        self._layers = 1 if self._single else mapping.shape[0]
        self.shape = tuple(logical_table.shape) if self._single else (self._layers, *logical_table.shape)
        self._kernel = None
        self._restore_padding = None
        if self.n:
            with torch.npu.device(logical_table.device):
                self._prepare_kernels(logical_table, mapping)

    def _prepare_kernels(self, logical_table: torch.Tensor, mapping: torch.Tensor) -> None:
        output_size = self._layers * self.n
        # A layer boundary must align with the 32-byte NPU write granularity.
        use_ub = (
            self._blocks <= _MAX_UB_MAPPING_SIZE
            and output_size <= _MAX_UB_OUTPUT_SIZE
            and self._layers <= _MAX_PROGRAMS
            and self.n % 8 == 0
        )
        if use_ub:
            vector_tile = 256 if self.n < 2048 else 1024
            shards = min(triton.cdiv(self.n, vector_tile), _MAX_PROGRAMS // self._layers)
            self._kernel = _compile(
                lookup_in_ub,
                shards * self._layers,
                logical_table,
                mapping,
                torch.int32,
                table_size=self.n,
                blocks=self._blocks,
                mapping_tile=triton.next_power_of_2(self._blocks),
                vector_tile=vector_tile,
                shards=shards,
            )
            return

        # The native large index_select route requires int64 indices.
        self._kernel = _compile(
            make_gather_indices,
            min(triton.cdiv(self.n, _VECTOR_TILE), _MAX_PROGRAMS),
            logical_table,
            torch.int64,
            table_size=self.n,
            vector_tile=_VECTOR_TILE,
        )
        self._restore_padding = _compile(
            restore_padding,
            min(triton.cdiv(output_size, _VECTOR_TILE), _MAX_PROGRAMS),
            logical_table,
            torch.int32,
            table_size=self.n,
            output_size=output_size,
            vector_tile=_VECTOR_TILE,
        )

    @property
    def attributes(self) -> dict[str, int]:
        return {}

    @property
    def info(self) -> dict[str, int | bool]:
        return {"layers": self._layers, "blocks": self._blocks, "single": self._single}

    def run(self) -> torch.Tensor:
        logical_table, mapping = self.inputs
        if self.n == 0:
            return torch.empty(self.shape, dtype=torch.int32, device=logical_table.device)

        return translate(logical_table, mapping, self._layers, self.shape, self._kernel, self._restore_padding)


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> Prepared:
    if not isinstance(inputs, Sequence) or len(inputs) != 2:
        raise ValueError("inputs must contain exactly two tensors (X and M)")
    if not isinstance(attributes, dict) or attributes:
        raise ValueError("attributes must be an empty dict")

    logical_table, mapping = inputs
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            raise TypeError("X and M must be int32 tensors")
        if tensor.device.type != "npu" or not tensor.is_contiguous():
            raise ValueError("X and M must be contiguous NPU tensors")
        if torch_npu.get_npu_format(tensor) != _ACL_FORMAT_ND:
            raise ValueError("X and M must use ND format")
    if logical_table.device != mapping.device:
        raise ValueError("X and M must be on the same NPU")
    if logical_table.ndim != 2 or mapping.ndim not in (1, 2) or mapping.numel() == 0:
        raise ValueError("X must be [R,C] and M must be nonempty [B] or [L,B]")

    layers = 1 if mapping.ndim == 1 else mapping.shape[0]
    if max(mapping.numel(), logical_table.numel() * layers) > _MAX_INT32:
        raise ValueError("X, M and output must each have at most 2147483647 elements")
    if mapping.min().item() < 0:
        raise ValueError("M must contain only nonnegative physical page IDs")
    if logical_table.numel() and logical_table.max().item() >= mapping.shape[-1]:
        raise ValueError("nonnegative logical page IDs in X must be less than B")
    return _PreparedBlockTable(logical_table, mapping)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    return prepare(inputs, attributes).run()
