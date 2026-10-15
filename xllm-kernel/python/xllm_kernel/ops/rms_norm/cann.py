# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Direct torch_npu RMSNorm provider with the existing functional contract."""

from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT
from xllm_kernel.registry import KernelRegistry, KernelSpec


def register(registry: KernelRegistry) -> None:
    import torch
    import torch_npu

    def run(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return torch_npu.npu_rms_norm(value, weight, eps)[0]

    registry.register(
        KernelSpec(
            "npu.cann.rms_norm",
            RMS_NORM_CONTRACT,
            "npu",
            ("eager", "aclgraph"),
            "torch_npu RMSNorm tensor domain",
            solution="cann",
        ),
        run,
    )
