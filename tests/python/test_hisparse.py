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

"""HiSparse logical paging and hot-cache lifecycle regression tests."""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.hisparse import physical_topk_slots


def test_physical_topk_uses_logical_pages_and_masks_padding() -> None:
    topk = torch.tensor([[[0, 127, 128, 255, -1, 999]], [[0, 1, 2, -1, 0, 1]]], dtype=torch.int32)
    table = torch.tensor([[7, 3], [11, -1]], dtype=torch.int32)
    result = physical_topk_slots(topk, table, torch.tensor([200, 2]), torch.tensor([584, -1]), 128)
    assert result.tolist() == [896, 1023, 384, -1, -1, -1, -1, -1, -1, -1, -1, -1]


def test_reordered_requests_resolve_their_own_physical_blocks() -> None:
    topk = torch.tensor([[[129, 3]], [[3, 129]]], dtype=torch.int32)
    table = torch.tensor([[5, 9], [8, 2]], dtype=torch.int32)
    lengths = torch.tensor([256, 256])
    slots = torch.tensor([10, 20])
    expected = physical_topk_slots(topk, table, lengths, slots, 128).view(2, 2)
    actual = physical_topk_slots(topk.flip(0), table.flip(0), lengths, slots.flip(0), 128).view(2, 2)
    torch.testing.assert_close(actual, expected.flip(0))


@pytest.mark.parametrize("page_size", [16, 128])
def test_missing_physical_page_is_never_read(page_size: int) -> None:
    result = physical_topk_slots(
        torch.tensor([[[0, page_size]]]),
        torch.tensor([[-1, 3]]),
        torch.tensor([page_size + 1]),
        torch.tensor([100]),
        page_size,
    )
    assert result.tolist() == [-1, 3 * page_size]
