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

"""Load all operation candidates and publish immutable startup bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from threading import RLock
from types import MappingProxyType

from xllm_kernel.catalog import load_operations
from xllm_kernel.registry import KernelRegistry, PreparedKernel
from xllm_kernel.selection import resolve_overrides, select_kernel

_configuration: tuple[str, str, tuple[tuple[str, str], ...]] | None = None
_prepared: Mapping[str, PreparedKernel] | None = None
_lock = RLock()


def initialize(
    *,
    device: str,
    execution_mode: str = "aclgraph",
    overrides: Mapping[str, str] | None = None,
) -> Mapping[str, PreparedKernel]:
    """Prepare every catalog operation once and return bindings keyed by name.

    Available backend modules register their implementations on import. Missing
    optional dependencies exclude a module; errors in an installed backend
    propagate. All modules load before the shared registry is frozen. Defaults
    use compatibility and priority; overrides name a solution or implementation.
    Environment overrides take precedence over the mapping. Repeating the same
    effective configuration returns the same read-only mapping. A policy change
    requires a new worker, and forward execution never reselects or falls back.
    """
    global _prepared, _configuration
    with _lock:
        operations = load_operations()
        choices = resolve_overrides(tuple(op.contract.operation for op in operations), overrides)
        configuration = (device, execution_mode, tuple(sorted(choices.items())))
        if _prepared is not None:
            if configuration != _configuration:
                raise RuntimeError("xllm_kernel is already initialized; restart the worker to change kernel policy")
            return _prepared
        if not operations or any(not any(b.device == device for b in op.backends) for op in operations):
            raise ValueError(f"kernel catalog does not provide all required operation adapters for {device!r}")

        unavailable: dict[str, tuple[tuple[str, str], ...]] = {}
        for operation in operations:
            excluded = []
            for backend in operation.backends:
                if backend.device != device:
                    continue
                missing = [name for name in backend.requires if find_spec(name) is None]
                if missing:
                    excluded.append((backend.module, f"missing dependencies: {', '.join(missing)}"))
                    continue
                import_module(backend.module)
            unavailable[operation.contract.operation] = tuple(excluded)

        KernelRegistry.get().freeze()
        bindings = {}
        for operation in operations:
            name = operation.contract.operation
            try:
                plan = select_kernel(
                    operation.contract,
                    device=device,
                    execution_mode=execution_mode,
                    override=choices.get(name),
                )
            except ValueError as error:
                details = "; ".join(f"{module}: {reason}" for module, reason in unavailable[name])
                raise ValueError(f"{error}; unavailable backends: {details or 'none'}") from error
            bindings[name] = replace(plan, excluded=plan.excluded + unavailable[name])
        # Publish only after every operation has a valid binding.
        for operation in operations:
            operation.prepared = bindings[operation.contract.operation]
        _configuration = configuration
        _prepared = MappingProxyType(bindings)
        return _prepared
