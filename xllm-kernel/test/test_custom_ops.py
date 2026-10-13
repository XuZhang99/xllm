# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Torch registration contracts for Python kernels and extension-owned schemas."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from xllm_kernel.custom_ops import register_custom_op


def _add_one(input: torch.Tensor) -> torch.Tensor:
    return input + 1


def _fake_add_one(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)


@pytest.mark.parametrize("with_fake", [False, True])
def test_python_operator_accepts_optional_fake(with_fake: bool) -> None:
    op = register_custom_op(
        f"test_add_one_{with_fake}",
        _add_one,
        mutates_args=(),
        dispatch_key="CPU",
        fake_impl=_fake_add_one if with_fake else None,
    )
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.testing.assert_close(op(value), value + 1)
    if with_fake:
        with FakeTensorMode():
            value = torch.empty(2, 3)
            output = op(value)
            assert (output.shape, output.dtype, output.device) == (value.shape, value.dtype, value.device)
            assert not torch._C._is_alias_of(output, value)


def test_existing_operator_can_add_fake_without_redefining_kernel() -> None:
    op = register_custom_op("test_existing_add", _add_one, mutates_args=(), dispatch_key="CPU")
    assert register_custom_op("test_existing_add", fake_impl=_fake_add_one) is op
    torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))
    with FakeTensorMode():
        assert op(torch.empty(3)).shape == (3,)
    with pytest.raises(RuntimeError, match="same name and overload name"):
        register_custom_op("test_existing_add", _add_one, mutates_args=(), dispatch_key="CPU")
    torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))


@pytest.mark.parametrize("metadata", [{"dispatch_key": "CPU"}, {"mutates_args": ()}])
def test_python_operator_requires_explicit_registration_metadata(metadata: dict) -> None:
    with pytest.raises(ValueError, match="explicit mutates_args and dispatch_key"):
        register_custom_op("test_missing_metadata", _add_one, **metadata)


def test_existing_operator_does_not_ignore_registration_metadata() -> None:
    op = register_custom_op("test_existing_metadata", _add_one, mutates_args=(), dispatch_key="CPU")
    with pytest.raises(ValueError, match="retain their registered mutation and dispatch metadata"):
        register_custom_op("test_existing_metadata", mutates_args=("input",))
    torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))
