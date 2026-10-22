---
title: "英伟达 GPU"
description: "使用 CUDA 后端在 英伟达 GPU 上运行 xLLM。"
sidebar:
  order: 7
---

本页介绍如何配置 英伟达 GPU 环境、安装 xLLM，并按 [启动 xllm](/zh/getting_started/launch_xllm/) 使用统一的 `xllm serve` 命令启动服务。

## 前置准备

- 在 Linux 宿主机上安装 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。启动容器前，确认 `nvidia-smi` 能识别 GPU。
- 在 [模型支持列表](/zh/supported_models/) 中确认后端支持目标模型，并将权重放在容器内可访问的目录。
- 下方命令将宿主机主目录挂载到容器内同一路径。若权重存放在其他位置，请增加类似 `-v /path/to/models:/models:ro` 的挂载，并在 `MODEL_PATH` 中使用容器内路径。

## 启动容器

以下命令在宿主机执行。镜像与 [快速开始](/zh/getting_started/quick_start/) 中的平台配置一致。

```bash
XLLM_IMAGE="quay.io/jd_xllm/xllm-ai:xllm-dev-cuda-x86"
docker pull "$XLLM_IMAGE"
```

```bash
docker run -it \
  --name xllm-cuda \
  --gpus all \
  --privileged \
  --shm-size=128g \
  --ipc=host \
  --network=host \
  --pid=host \
  -v "$HOME:$HOME" \
  -w "$HOME" \
  "$XLLM_IMAGE" \
  /bin/bash
```

退出后容器会保留，使用 `docker start -ai xllm-cuda` 重新进入已停止的容器；若容器仍在运行，使用 `docker exec -it xllm-cuda bash`。

## 安装 xLLM

以下命令在容器内执行。若镜像已安装 xLLM wheel 且支持 `xllm serve`，可跳过源码安装。否则，克隆仓库（或进入已有源码目录）、初始化子模块，并为当前后端构建和安装 wheel：

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

# 第一次需要进行 pre-commit 安装
pip install pre-commit
pre-commit install

git submodule update --init --recursive
python setup.py install --device cuda
```

也可以直接使用 `python setup.py install` 安装；省略 `--device` 时，安装脚本会根据当前 PyTorch 环境自动识别设备后端。

`python setup.py build` 仅构建产物；统一启动命令需要安装 wheel 及其包含的服务二进制。继续前先检查已安装的命令：

```bash
command -v xllm
xllm serve --help
```

## 启动服务

将 `MODEL_PATH` 替换为权重目录后，在容器内执行。下方示例启动一张卡；请结合目标模型部署配方和 [CLI 参数参考](/zh/cli_reference/) 调整。

```bash
#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/path/to/model/Qwen3-8B"
PORT=8010
MASTER_NODE_ADDR="127.0.0.1:9748"
NNODES=1

xllm serve \
  --model="$MODEL_PATH" \
  --model_id=Qwen3-8B \
  --port="$PORT" \
  --nnodes="$NNODES" \
  --master_node_addr="$MASTER_NODE_ADDR" \
  --block_size=32 \
  --max_memory_utilization=0.8 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=true
```

单机双卡时，将脚本中的设备选择和 `NNODES` 两行替换为：

```bash
export CUDA_VISIBLE_DEVICES=0,1
NNODES=2
```

只需执行一次 `xllm serve`。启动器自动分配 rank 和各 rank 的端口，将日志写入 `log/node_0.log`、`log/node_1.log` 等文件，并在前台运行。不要传入 `--node_rank` 或再套一层按设备启动的循环。

## 验证服务

在另一个 shell 中，等待模型加载完成后查询 rank 0 的服务（本例端口为 `8010`）：

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

模型列表应包含 `--model_id` 指定的 `Qwen3-8B`。调用 [OpenAI 兼容 API](/zh/basic_usage/openai_api/) 时使用该 ID；若修改 `PORT`，请求地址也要同步调整。

## 平台说明

- 容器命令保留 `--gpus all`，由 NVIDIA 容器运行时暴露 GPU；`CUDA_VISIBLE_DEVICES` 用于选择 xLLM 实际使用的可见 GPU。
- 示例使用 `--block_size=32`、`--max_memory_utilization=0.8` 和 `--enable_chunked_prefill=false`，请根据模型和可用显存调整。
- 调试时可设置 `CUDA_LAUNCH_BLOCKING=1`，使 CUDA 调用同步执行；正常服务和性能测量时不要开启。
- CUDA timeline 采集见 [在线服务性能分析](/zh/dev_guide/online_profiling/)。
