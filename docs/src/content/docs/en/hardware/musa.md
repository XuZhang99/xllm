---
title: "Mthreads MUSA"
description: "Run xLLM on Mthreads MUSA GPUs."
sidebar:
  order: 6
---

Set up an xLLM environment for Mthreads MUSA and start a service with the unified `xllm serve` command from [Launch xllm](/en/getting_started/launch_xllm/).

## Prerequisites

- Install the Mthreads driver and Docker on the Linux host. Confirm that `/dev/mtgpu0` and `/dev/dri` are available before starting the container.
- Check the [Model Support List](/en/supported_models/) for your backend and prepare the model weights in a directory accessible inside the container.
- The commands below mount your home directory at the same path. For weights stored elsewhere, add a mount such as `-v /path/to/models:/models:ro` and use the container path in `MODEL_PATH`.

## Start the container

Run the following commands on the host. The image is the platform image listed in [Quick Start](/en/getting_started/quick_start/).

```bash
XLLM_IMAGE="registry.mthreads.com/presale/devtech/xllm:0710"
docker pull "$XLLM_IMAGE"
```

```bash
docker run -it \
  --name xllm-musa \
  --ipc=host \
  --network=host \
  --privileged \
  --shm-size=128g \
  --device=/dev/mtgpu0 \
  --device=/dev/dri \
  --group-add video \
  --ulimit memlock=-1 \
  -v "$HOME:$HOME" \
  -w "$HOME" \
  "$XLLM_IMAGE" \
  /bin/bash
```

For additional devices, add the corresponding `--device=/dev/mtgpuN` entries before creating the container.

The container remains after exit. Re-enter it with `docker start -ai xllm-musa` when stopped, or `docker exec -it xllm-musa bash` while it is running.

## Install xLLM

Run inside the container. If your image already includes the xLLM wheel and supports `xllm serve`, skip the source installation. Otherwise, clone the repository (or enter your existing checkout), initialize its submodules, and build and install the wheel for this backend:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

# Install pre-commit on first use
pip install pre-commit
pre-commit install

git submodule update --init --recursive
python setup.py install --device musa
```

You can also install directly with `python setup.py install`; the installer automatically detects the backend from the current PyTorch environment when `--device` is omitted.

`python setup.py build` only builds the artifacts; the unified command requires the installed wheel and its packaged server binary. Check the installed command before continuing:

```bash
command -v xllm
xllm serve --help
```

## Launch the service

Run inside the container after replacing `MODEL_PATH` with your weights directory. This example starts one device. Adjust the backend settings using the target model recipe and [CLI Reference](/en/cli_reference/).

```bash
#!/bin/bash
set -e

export MUSA_VISIBLE_DEVICES=0

MODEL_PATH="/path/to/model/Qwen3.5-27B"
PORT=8010
MASTER_NODE_ADDR="127.0.0.1:9748"
NNODES=1

xllm serve \
  --model="$MODEL_PATH" \
  --model_id=Qwen3.5-27B \
  --port="$PORT" \
  --nnodes="$NNODES" \
  --master_node_addr="$MASTER_NODE_ADDR" \
  --block_size=64 \
  --max_memory_utilization=0.8 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_graph=true
```

For two devices on one machine, replace the device selection and `NNODES` lines in the script with:

```bash
export MUSA_VISIBLE_DEVICES=0,1
NNODES=2
```

Run `xllm serve` once. The launcher assigns ranks and per-rank ports, writes `log/node_0.log`, `log/node_1.log`, and so on, and stays in the foreground. Do not pass `--node_rank` or wrap the command in a per-device loop.

## Verify the service

In another shell, wait for model loading to finish and query the rank-0 service (port `8010` in this example):

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

The model list should include `Qwen3.5-27B`, as set by `--model_id`. Use that ID in requests to the [OpenAI-Compatible APIs](/en/basic_usage/openai_api/). If you change `PORT`, update the request URLs as well.

## Platform notes

- The container example exposes `/dev/mtgpu0`. Add the device mounts required for any additional GPUs before selecting them with `MUSA_VISIBLE_DEVICES`.
- This example uses Qwen3.5-27B with `--block_size=64`, chunked prefill, schedule overlap, and graph execution. Choose the number of GPUs to fit the model and KV cache.
