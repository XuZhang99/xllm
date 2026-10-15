# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Lazy, explicit preparation of an operation's independently owned backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from xllm_kernel.registry import KernelRegistry, KernelSpec, OpContract, PreparedKernel


@dataclass(frozen=True)
class Backend:
    """Declare a loader, supported execution modes, and a backend's input domain."""

    name: str
    loader: str
    input_domain: str
    execution_modes: tuple[str, ...] = ("eager", "aclgraph")


@dataclass(frozen=True)
class Operator:
    """Prepare only the requested provider; never probe or fall back at execution."""

    contract: OpContract
    backends: tuple[Backend, ...]

    @property
    def implementations(self) -> tuple[str, ...]:
        return tuple(f"npu.{backend.name}.{self.contract.operation}" for backend in self.backends)

    def prepare(self, device: str, mode: str, implementation: str) -> PreparedKernel:
        if device != "npu":
            raise ValueError(f"{self.contract.operation} requires device='npu'")
        names = dict(zip(self.implementations, self.backends, strict=True))
        if implementation not in names:
            raise ValueError(f"unsupported implementation {implementation!r}; choose from {tuple(names)}")
        backend = names[implementation]
        if mode not in backend.execution_modes:
            raise ValueError(f"{implementation} does not support {mode}")
        module, symbol = backend.loader.split(":", 1)
        function = getattr(import_module(module), symbol)()
        if not callable(function):
            raise TypeError(f"{backend.loader} must return a callable")
        registry = KernelRegistry()
        registry.register(
            KernelSpec(
                implementation,
                self.contract,
                device,
                backend.execution_modes,
                backend.input_domain,
                solution=backend.name,
            ),
            function,
        )
        registry.freeze()
        return registry.prepare(self.contract, device=device, execution_mode=mode, implementation=implementation)


def native_operator(name: str) -> Callable[..., Any]:
    """Resolve a host-owned schema after the caller has loaded its native library."""
    import torch

    operator = getattr(torch.ops.xllm_ops, name, None)
    if operator is None:
        from xllm import xllm_export  # noqa: F401

        operator = getattr(torch.ops.xllm_ops, name, None)
    if operator is None:
        raise RuntimeError(f"load the matching xLLM native library before preparing xllm_ops::{name}")
    return operator
