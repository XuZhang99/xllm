# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Preparation selects exactly one provider before device execution."""

from types import SimpleNamespace

import pytest
import xllm_kernel.operator as preparation
from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract


def test_selection_only_loads_the_requested_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    function = lambda value: value

    def imported(name: str) -> SimpleNamespace:
        calls.append(name)
        return SimpleNamespace(load=lambda: function)

    monkeypatch.setattr(preparation, "import_module", imported)
    operation = Operator(
        OpContract("example", 1, (), False, "exact"),
        (
            Backend("first", "first:load", "domain"),
            Backend("second", "second:load", "domain"),
        ),
    )
    plan = operation.prepare("npu", "aclgraph", "npu.second.example")
    assert calls == ["second"]
    assert plan.function is function
    assert plan.spec.contract == operation.contract
    assert plan.spec.solution == "second"


def test_unsupported_selection_fails_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(name: str) -> None:
        raise AssertionError(f"must not import {name}")

    monkeypatch.setattr(preparation, "import_module", unexpected)
    operation = Operator(
        OpContract("example", 1, (), False, "exact"), (Backend("only", "only:load", "domain", ("eager",)),)
    )
    for device, mode, implementation in (
        ("cuda", "eager", "npu.only.example"),
        ("npu", "aclgraph", "npu.only.example"),
        ("npu", "eager", "npu.missing.example"),
    ):
        with pytest.raises(ValueError):
            operation.prepare(device, mode, implementation)
