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

"""Translate logical KV-cache token slots to physical slots on Ascend."""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_INT32_MAX = (1 << 31) - 1
_MAX_UB_PAGES = 4096


class Prepared(Protocol):
    def run(self) -> torch.Tensor: ...


class _Launcher(Protocol):
    def __call__(self, slots: torch.Tensor, mapping: torch.Tensor, output: torch.Tensor, *, stream: int) -> None: ...


@triton.jit(do_not_specialize_on_alignment=("slots", "mapping", "output"))
def _translate(
    slots: tl.tensor,
    mapping: tl.tensor,
    output: tl.tensor,
    num_slots: tl.constexpr,
    num_pages: tl.constexpr,
    page_size: tl.constexpr,
    page_shift: tl.constexpr,
    reciprocal: tl.constexpr,
    page_tile: tl.constexpr,
    slot_tile: tl.constexpr,
) -> None:
    layer = tl.program_id(0)
    positions = tl.program_id(1) * slot_tile + tl.arange(0, slot_tile)
    in_bounds = positions < num_slots
    logical_slot = tl.load(slots + positions, in_bounds, other=-1)
    valid = logical_slot >= 0
    safe_slot = tl.maximum(logical_slot, 0)

    if page_shift >= 0:
        page = safe_slot >> page_shift
        offset = safe_slot & (page_size - 1)
    else:
        # The fixed-point quotient can underestimate by one. Its product
        # with page_size is bounded by safe_slot, so the residual fits int32.
        estimate = ((safe_slot.to(tl.int64) * reciprocal) >> 32).to(tl.int32)
        residual = safe_slot - estimate * page_size
        carry = residual >= page_size
        page = estimate + carry.to(tl.int32)
        offset = residual - tl.where(carry, page_size, 0)

    mapping_row = mapping + layer * num_pages
    if page_tile:
        pages = tl.arange(0, page_tile)
        physical_pages = tl.load(mapping_row + pages, pages < num_pages, other=0)
        # Ascend Triton 3.2.2 accepts float32, but not int32, in UB gather.
        # Gather only moves bits, preserving all 32 bits of the page ID.
        physical_page = tl.gather(physical_pages.to(tl.float32, bitcast=True), page, axis=0).to(tl.int32, bitcast=True)
    else:
        physical_page = tl.load(mapping_row + page, valid, other=0)

    translated = tl.where(valid, physical_page * page_size + offset, -1)
    output_row = output + layer * num_slots
    tl.store(output_row + positions, translated, in_bounds)


@dataclass(frozen=True)
class _Prepared:
    _slots: torch.Tensor
    _mapping: torch.Tensor
    _shape: tuple[int, ...]
    _launch: _Launcher | None
    _guard_device: bool = False

    def run(self) -> torch.Tensor:
        device = self._slots.device
        output = torch.empty(self._shape, dtype=torch.int32, device=device)
        if self._launch is not None:
            guard = torch.npu.device(device) if self._guard_device else nullcontext()
            with guard:
                # The Ascend driver obtains a raw stream without draining its queue.
                stream = triton.runtime.driver.active.get_current_stream(device.index)
                self._launch(self._slots, self._mapping, output, stream=stream)
        return output


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> Prepared:
    """Validate the initial values and prepare a launch for these tensor objects."""
    if not isinstance(inputs, Sequence) or len(inputs) != 2:
        raise ValueError("inputs must contain slots and M")
    if not isinstance(attributes, dict) or attributes.keys() != {"block_size"}:
        raise ValueError("attributes must contain exactly block_size")
    page_size = attributes["block_size"]
    if type(page_size) is not int or not 1 <= page_size <= _INT32_MAX:
        raise ValueError("block_size must be an int in [1, 2147483647]")

    slots, mapping = inputs
    for name, tensor in zip(("slots", "M"), inputs):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be an int32 tensor")
        if tensor.device.type != "npu" or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous and on an NPU")
        if tensor.numel() > _INT32_MAX:
            raise ValueError(f"{name} exceeds the int32 element limit")
    if slots.device != mapping.device:
        raise ValueError("slots and M must be on the same NPU")
    if slots.ndim != 1 or mapping.ndim not in (1, 2) or not mapping.numel():
        raise ValueError("expected slots[N] and nonempty M[B] or M[L,B]")

    num_slots = slots.numel()
    num_pages = mapping.shape[-1]
    layers = mapping.shape[0] if mapping.ndim == 2 else 1
    if layers * num_slots > _INT32_MAX:
        raise ValueError("output exceeds the int32 element limit")

    bounds = [mapping.amin(), mapping.amax()]
    if num_slots:
        bounds.append(slots.amax())
    min_page, max_page, *max_slot = torch.stack(bounds).cpu().tolist()
    if min_page < 0 or max_page > (_INT32_MAX - page_size + 1) // page_size:
        raise ValueError("M contains a negative page or a page outside int32")
    if max_slot and max_slot[0] >= num_pages * page_size:
        raise ValueError("slots contain a logical page outside M")

    shape = (layers, num_slots) if mapping.ndim == 2 else (num_slots,)
    if not num_slots:
        return _Prepared(slots, mapping, shape, None)

    page_shift = page_size.bit_length() - 1 if page_size & (page_size - 1) == 0 else -1
    slot_tile = min(2048, triton.next_power_of_2(max(32, num_slots)))
    page_tile = max(8, triton.next_power_of_2(num_pages)) if num_pages <= _MAX_UB_PAGES else 0
    grid = (layers, triton.cdiv(num_slots, slot_tile), 1)
    with torch.npu.device(slots.device):
        # Warmup compiles without reading pointers. The launcher receives the
        # live tensors and the current stream on every call.
        kernel = _translate.warmup(
            torch.int32,
            torch.int32,
            torch.int32,
            num_slots=num_slots,
            num_pages=num_pages,
            page_size=page_size,
            page_shift=page_shift,
            reciprocal=(1 << 32) // page_size if page_shift < 0 else 0,
            page_tile=page_tile,
            slot_tile=slot_tile,
            grid=grid,
        )
        launch = kernel[grid]
    return _Prepared(slots, mapping, shape, launch, _guard_device=torch.npu.device_count() > 1)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    """Validate and translate once."""
    return prepare(inputs, attributes).run()
