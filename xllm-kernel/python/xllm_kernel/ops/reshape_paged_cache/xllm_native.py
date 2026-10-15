# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""reshape_paged_cache xllm_native provider, resolved before graph capture."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load() -> Callable[..., Any]:
    from xllm_kernel.operator import native_operator

    return native_operator("reshape_paged_cache")
