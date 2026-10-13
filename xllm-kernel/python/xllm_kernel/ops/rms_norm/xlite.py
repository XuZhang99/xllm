# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Optional xlite solution, loaded and checked only during preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from xllm_kernel.custom_ops import register_custom_op
from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT
from xllm_kernel.registry import register_kernel

if TYPE_CHECKING:
    import torch

IMPLEMENTATION = "npu.xlite.rms_norm"


def _fake_rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import torch

    if input.ndim < 1 or weight.ndim != 1 or input.shape[-1] != weight.numel():
        raise ValueError("xlite RMSNorm requires a weight matching the input's last dimension")
    if input.device.type != "npu" or input.device != weight.device:
        raise ValueError("xlite RMSNorm requires tensors on the same NPU")
    if input.dtype not in (torch.float16, torch.bfloat16) or input.dtype != weight.dtype:
        raise ValueError("xlite RMSNorm requires matching FP16/BF16 tensors")
    if weight.numel() <= 0 or weight.numel() > 8192 or weight.numel() % 64:
        raise ValueError("xlite RMSNorm requires a width divisible by 64 up to 8192")
    if input.numel() > 2**32 - 1:
        raise ValueError("xlite RMSNorm requires element offsets to fit uint32")
    if not math.isfinite(eps) or eps < 0 or eps > torch.finfo(torch.float32).max:
        raise ValueError("xlite RMSNorm requires a finite nonnegative FP32 epsilon")
    return torch.empty(input.shape, dtype=input.dtype, device=input.device)


def _load_extension() -> None:
    """Require the tested launcher ABI before loading our Torch operator."""
    import torch
    import torch_npu

    extension = importlib.util.find_spec("xllm_kernel._xlite")
    dependency = importlib.util.find_spec("xlite")
    if extension is None or extension.origin is None or dependency is None or dependency.origin is None:
        raise RuntimeError(
            "xlite RMSNorm requires its optional bridge and GVirt; build xllm-kernel with "
            "XLLM_KERNEL_BUILD_XLITE=1 in the matching NPU environment before worker startup"
        )
    manifest_path = Path(extension.origin).with_name("_xlite_build.json")
    if not manifest_path.is_file():
        raise RuntimeError("xlite bridge build manifest is missing; rebuild the extension")
    manifest = json.loads(manifest_path.read_text())
    xlite = Path(dependency.origin).parent
    revision = (xlite / ".xlite_git_head").read_text().strip()
    if manifest["schema_version"] != 1 or revision != manifest["gvirt_revision"]:
        raise RuntimeError("xlite launcher revision does not match the bridge build")
    if (torch.__version__, torch_npu.__version__) != (manifest["torch"], manifest["torch_npu"]):
        raise RuntimeError("xlite bridge PyTorch/torch_npu versions changed; rebuild the extension")
    for name, expected in manifest["libraries"].items():
        if hashlib.sha256((xlite / "lib" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"xlite launcher binary changed: {name}; rebuild the bridge")
    torch.ops.load_library(extension.origin)


_load_extension()
register_kernel(
    name=IMPLEMENTATION,
    contract=RMS_NORM_CONTRACT,
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="FP16/BF16, same NPU/dtype, ND, width divisible by 64 and <=8192, numel<=2**32-1",
    solution="xlite",
)(register_custom_op("xlite_rms_norm", fake_impl=_fake_rms_norm))
