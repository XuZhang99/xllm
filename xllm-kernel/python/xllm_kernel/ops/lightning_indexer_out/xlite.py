# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
from collections.abc import Callable
from typing import Any


def load() -> Callable[..., Any]:
    from xllm_kernel.backends.xlite import load_operator

    return load_operator("lightning_indexer_out")
