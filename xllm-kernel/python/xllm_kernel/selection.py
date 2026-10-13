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

"""Shared preparation-time selection and per-operation overrides."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from xllm_kernel.registry import KernelRegistry, OpContract, PreparedKernel

_PREFIX = "XLLM_KERNEL_OVERRIDE_"


def resolve_overrides(operations: tuple[str, ...], overrides: Mapping[str, str] | None) -> dict[str, str]:
    """Resolve environment overrides over explicit per-operation choices."""
    if len(set(operations)) != len(operations):
        raise ValueError("duplicate operation in kernel catalog")
    if any(re.fullmatch(r"[a-z][a-z0-9_]*", name) is None for name in operations):
        raise ValueError("kernel operation names must be lowercase snake_case")
    resolved = dict(overrides or {})
    unknown = set(resolved) - set(operations)
    if unknown:
        raise ValueError(f"unknown kernel override operation(s): {sorted(unknown)}")
    keys = {_PREFIX + name.upper(): name for name in operations}
    for key, value in os.environ.items():
        if not key.startswith(_PREFIX):
            continue
        if key not in keys:
            raise ValueError(f"unknown kernel override environment variable {key!r}")
        if value:
            resolved[keys[key]] = value
    if any(not isinstance(value, str) or not value.strip() for value in resolved.values()):
        raise ValueError("kernel overrides must be nonempty implementation or solution names")
    return resolved


def select_kernel(
    contract: OpContract,
    *,
    device: str,
    execution_mode: str,
    override: str | None = None,
) -> PreparedKernel:
    """Select by compatibility, priority and stable name; overrides keep validation.

    An override names an exact implementation or a solution. Exact names take
    precedence. This function only runs during preparation, after registry freeze.
    It never benchmarks a candidate or retries a failed implementation.
    """
    registry = KernelRegistry.get()
    implementation = None
    solution = None
    if override is not None:
        if any(spec.name == override for spec in registry.specs()):
            implementation = override
        else:
            solution = override
    return registry.prepare(
        contract,
        device=device,
        execution_mode=execution_mode,
        implementation=implementation,
        solution=solution,
    )
