---
title: "启动xllm"
sidebar:
  order: 3
---


需要提前安装 xllm wheel 包或者使用 release 镜像：

```bash
# from source
python setup.py install
```

启动 xLLM 前，根据硬件平台设置对应的环境变量，指定使用的设备：

- 昇腾 NPU：`export ASCEND_RT_VISIBLE_DEVICES=0`
- NVIDIA GPU：`export CUDA_VISIBLE_DEVICES=0`
- 寒武纪 MLU：`export MLU_VISIBLE_DEVICES=0`
- 海光 DCU：`export HIP_VISIBLE_DEVICES=0`
- 沐曦 MACA：`export CUDA_VISIBLE_DEVICES=0`
- 摩尔线程 MUSA：`export MUSA_VISIBLE_DEVICES=0`

使用逗号分隔设备 ID 来指定多个设备，例如 `export ASCEND_RT_VISIBLE_DEVICES=0,1`。单机部署时，将 `NNODES` 设置为所选设备的数量（一张卡对应一个 node）。以下脚本适用于单机单卡和单机多卡：

```bash
#!/bin/bash
set -e

MODEL_PATH="/path/to/model"
PORT=8010
MASTER_NODE_ADDR="127.0.0.1:9748"
NNODES=1

xllm serve --model=$MODEL_PATH \
       --port=$PORT \
       --nnodes=$NNODES \
       --master_node_addr=$MASTER_NODE_ADDR \
       --block_size=128 \
       --max_memory_utilization=0.86 \
       --enable_prefix_cache=false \
       --enable_chunked_prefill=true \
       --enable_schedule_overlap=true \
       --enable_graph=true
```

多机部署时，需要添加 `--machine_rank` 参数，指定当前机器的编号。
