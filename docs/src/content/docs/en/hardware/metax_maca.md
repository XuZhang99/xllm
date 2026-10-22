---
title: "MetaX MACA"
description: "Run xLLM on MetaX MACA devices with the MetaX MACA backend."
sidebar:
  order: 5
---

Set up an xLLM environment for MetaX MACA and start a service with the unified `xllm serve` command from [Launch xllm](/en/getting_started/launch_xllm/).

## Prerequisites

- Install the MetaX driver, MACA environment, and Docker on the Linux host. Confirm that `/opt/maca` and the device nodes used below are available; the container configuration includes InfiniBand device access.
- Check the [Model Support List](/en/supported_models/) for your backend and prepare the model weights in a directory accessible inside the container.
- The commands below mount your home directory at the same path. For weights stored elsewhere, add a mount such as `-v /path/to/models:/models:ro` and use the container path in `MODEL_PATH`.

## Start the container

Run the following commands on the host. The image is the platform image listed in [Quick Start](/en/getting_started/quick_start/).

```bash
XLLM_IMAGE="pub-registry1.metax-tech.com/dev-m01421/xllm-maca3.7.1.9:v1"
docker pull "$XLLM_IMAGE"
```

```bash
docker run -it \
  --name xllm-maca \
  --ipc=host \
  -u 0 \
  --network=host \
  --privileged \
  --shm-size=100g \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --group-add video \
  --ulimit memlock=-1 \
  -v /opt/maca:/opt/maca \
  -v "$HOME:$HOME" \
  -w "$HOME" \
  "$XLLM_IMAGE" \
  /bin/bash
```

The container remains after exit. Re-enter it with `docker start -ai xllm-maca` when stopped, or `docker exec -it xllm-maca bash` while it is running.

## Install xLLM

Run inside the container. If your image already includes the xLLM wheel and supports `xllm serve`, skip the source installation. Otherwise, clone the repository (or enter your existing checkout), initialize its submodules, and build and install the wheel for this backend:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

# Install pre-commit on first use
pip install pre-commit
pre-commit install

git submodule update --init --recursive
python setup.py install --device maca
```

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

export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_OPS_PATH=/opt/conda/lib/python3.10/site-packages/flashinfer/data/aot/

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
  --block_size=128 \
  --max_memory_utilization=0.86 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=true
```

For two devices on one machine, replace the device selection and `NNODES` lines in the script with:

```bash
export CUDA_VISIBLE_DEVICES=0,1
NNODES=2
```

Run `xllm serve` once. The launcher assigns ranks and per-rank ports, writes `log/node_0.log`, `log/node_1.log`, and so on, and stays in the foreground. Do not pass `--node_rank` or wrap the command in a per-device loop.

## Verify the service

In another shell, wait for model loading to finish and query the rank-0 service (port `8010` in this example):

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

The model list should include `Qwen3-8B`, as set by `--model_id`. Use that ID in requests to the [OpenAI-Compatible APIs](/en/basic_usage/openai_api/). If you change `PORT`, update the request URLs as well.

## Platform notes

- Automatic device detection does not currently include MACA. Build the wheel with `--device maca`; runtime device selection uses `CUDA_VISIBLE_DEVICES`.
- `FLASHINFER_OPS_PATH` in the launch script matches the listed image. If you use another image or Python environment, set it to that installation's FlashInfer `data/aot/` directory.
- The example uses `--block_size=128`, `--max_memory_utilization=0.86`, and `--enable_chunked_prefill=false` for MACA.
