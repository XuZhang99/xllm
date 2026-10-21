---
title: "Launch xllm"
sidebar:
  order: 3
---


Install the xllm wheel package in advance, or use a release image:

```bash
# from source
python setup.py install
```

Before launching xLLM, select the devices by setting the environment variable for your hardware platform:

- Ascend NPU: `export ASCEND_RT_VISIBLE_DEVICES=0`
- NVIDIA GPU: `export CUDA_VISIBLE_DEVICES=0`
- Cambricon MLU: `export MLU_VISIBLE_DEVICES=0`
- Hygon DCU: `export HIP_VISIBLE_DEVICES=0`
- MetaX MACA: `export CUDA_VISIBLE_DEVICES=0`
- Mthreads MUSA: `export MUSA_VISIBLE_DEVICES=0`

Use comma-separated device IDs to select multiple devices, for example, `export ASCEND_RT_VISIBLE_DEVICES=0,1`. For single-machine deployments, set `NNODES` to the number of selected devices (one device per node). The following script supports both single-device and multi-device launches:

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

When deploying on multiple machines, you need to add the `--machine_rank` parameter to specify the rank of the current machine.
