---
title: "硬件平台"
description: "xLLM 支持硬件平台的环境、启动和模型支持入口。"
sidebar:
  order: 1
---

根据加速器类型选择指南，依次准备容器、安装 xLLM、选择设备并验证服务。

## 选择平台

| 平台 | 构建目标 | 设备选择变量 | 容器镜像 |
| --- | --- | --- | --- |
| [昇腾 NPU](/zh/hardware/ascend_npu/) | `--device npu` | `ASCEND_RT_VISIBLE_DEVICES` | 见平台指南 |
| [寒武纪 MLU](/zh/hardware/cambricon_mlu/) | `--device mlu` | `MLU_VISIBLE_DEVICES` | 由厂商提供 |
| [海光 DCU](/zh/hardware/hygon_dcu/) | `--device dcu` | `HIP_VISIBLE_DEVICES` | 见平台指南 |
| [沐曦 MACA](/zh/hardware/metax_maca/) | `--device maca` | `CUDA_VISIBLE_DEVICES` | 见平台指南 |
| [摩尔线程 MUSA](/zh/hardware/musa/) | `--device musa` | `MUSA_VISIBLE_DEVICES` | 见平台指南 |
| [英伟达 GPU](/zh/hardware/nvidia_gpu/) | `--device cuda` | `CUDA_VISIBLE_DEVICES` | 见平台指南 |

各平台提供的镜像和 CPU 架构组合不同，可通过 [快速开始选择器](/zh/getting_started/quick_start/) 查看。
