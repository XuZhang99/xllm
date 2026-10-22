---
title: "Hardware Platforms"
description: "Platform-specific guides for running xLLM on supported hardware."
sidebar:
  order: 1
---

Choose the guide for your accelerator to prepare the container, install xLLM, select devices, and verify the service.

## Choose a platform

| Platform | Build target | Device selection | Container image |
| --- | --- | --- | --- |
| [Ascend NPU](/en/hardware/ascend_npu/) | `--device npu` | `ASCEND_RT_VISIBLE_DEVICES` | Listed in the guide |
| [Cambricon MLU](/en/hardware/cambricon_mlu/) | `--device mlu` | `MLU_VISIBLE_DEVICES` | Vendor-provided |
| [Hygon DCU](/en/hardware/hygon_dcu/) | `--device dcu` | `HIP_VISIBLE_DEVICES` | Listed in the guide |
| [MetaX MACA](/en/hardware/metax_maca/) | `--device maca` | `CUDA_VISIBLE_DEVICES` | Listed in the guide |
| [Mthreads MUSA](/en/hardware/musa/) | `--device musa` | `MUSA_VISIBLE_DEVICES` | Listed in the guide |
| [NVIDIA GPU](/en/hardware/nvidia_gpu/) | `--device cuda` | `CUDA_VISIBLE_DEVICES` | Listed in the guide |

Image availability and CPU architectures differ by platform; the [Quick Start selector](/en/getting_started/quick_start/) lists the available combinations.
