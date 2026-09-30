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

"""Tests for Python process-group rendezvous ownership."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from xllm.python.distributed import collectives


@pytest.fixture(autouse=True)
def _isolated_collective_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(collectives, "_groups", {})
    monkeypatch.setattr(collectives, "_group_ranks", {})
    monkeypatch.setattr(collectives, "_stores", {})
    monkeypatch.setattr(collectives, "_world_topology", None)
    monkeypatch.setattr(collectives, "_world_initialized", False)
    yield
    collectives._shutdown_process_groups()


def test_npu_world_registers_shutdown_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the NPU lifecycle with a CPU/Gloo world, without device kernels.
    store = dist.HashStore()
    register = MagicMock()
    monkeypatch.setattr(collectives.atexit, "register", register)
    monkeypatch.setattr(collectives, "_USE_PYTHON_NPU_GROUPS", True)
    monkeypatch.setattr(collectives, "_backend_for", lambda device: "gloo")
    monkeypatch.setattr(collectives, "_shared_store", lambda *args: store)

    collectives._ensure_world("localhost", 0, torch.device("cpu"), 0, 1)
    collectives._ensure_world("localhost", 0, torch.device("cpu"), 0, 1)

    register.assert_called_once_with(collectives._shutdown_process_groups)
    assert collectives._groups[("attn_dp", "cpu")] is dist.group.WORLD
    assert collectives._group_ranks[("attn_dp", "cpu")] == (0,)


def test_shutdown_process_groups_releases_world_and_cached_groups() -> None:
    store = dist.HashStore()
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    collectives._world_initialized = True
    collectives._groups[("tp", "cpu")] = dist.new_group(ranks=[0], backend="gloo")
    collectives._group_ranks[("tp", "cpu")] = (0,)
    collectives._stores[("localhost", 0)] = store
    collectives._world_topology = [{"hostname": "localhost", "device_index": None}]

    collectives._shutdown_process_groups()

    assert not dist.is_initialized()
    assert not collectives._groups
    assert not collectives._group_ranks
    assert not collectives._stores
    assert collectives._world_topology is None
    assert not collectives._world_initialized
    collectives._shutdown_process_groups()


def test_topology_failure_keeps_world_and_store_owned_for_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed topology exchange still releases the initialized Gloo world."""
    register = MagicMock()
    monkeypatch.setattr(collectives.atexit, "register", register)
    monkeypatch.setattr(collectives, "_USE_PYTHON_NPU_GROUPS", True)
    monkeypatch.setattr(collectives, "_backend_for", lambda device: "gloo")
    monkeypatch.setattr(
        collectives, "_exchange_world_topology", MagicMock(side_effect=RuntimeError("topology exchange failed"))
    )
    collectives._stores[("localhost", 0)] = dist.HashStore()

    try:
        with pytest.raises(RuntimeError, match="topology exchange failed"):
            collectives._ensure_world("localhost", 0, torch.device("cpu"), 0, 1)

        assert dist.is_initialized()
        register.assert_called_once_with(collectives._shutdown_process_groups)
        register.call_args.args[0]()

        assert not dist.is_initialized()
        assert not collectives._groups
        assert not collectives._group_ranks
        assert not collectives._stores
        assert collectives._world_topology is None
        assert not collectives._world_initialized
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_shutdown_process_groups_does_not_destroy_an_unowned_world() -> None:
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        collectives._shutdown_process_groups()
        assert dist.is_initialized()
    finally:
        dist.destroy_process_group()
