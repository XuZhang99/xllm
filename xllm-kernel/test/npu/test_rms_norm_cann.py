# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""CANN RMSNorm uses the same independent reference as other providers."""

import pytest
from operator_checks import check_suite
from xllm_kernel.numerics.rms_norm import SUITE


@pytest.mark.parametrize("mode", ("eager", "aclgraph"))
def test_cann_rms_norm(mode: str) -> None:
    check_suite(SUITE, "npu.cann.rms_norm", mode)
