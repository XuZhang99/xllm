---
title: "Hygon DCU"
description: "Run xLLM on Hygon DCU devices with the Hygon DCU backend."
sidebar:
  order: 4
---

Set up an xLLM environment for Hygon DCU and start a service with the unified `xllm serve` command from [Launch xllm](/en/getting_started/launch_xllm/).

## Prerequisites

- Install the Hygon DCU driver and Docker on the Linux host. Confirm that the device nodes `/dev/kfd`, `/dev/dri`, and `/dev/mkfd`, and the `/opt/hyhal` directory required by the container are available.
- Check the [Model Support List](/en/supported_models/) for your backend and prepare the model weights in a directory accessible inside the container.
- The commands below mount your home directory at the same path. For weights stored elsewhere, add a mount such as `-v /path/to/models:/models:ro` and use the container path in `MODEL_PATH`.

## Start the container

Run the following commands on the host. The image is the platform image listed in [Quick Start](/en/getting_started/quick_start/).

```bash
XLLM_IMAGE="harbor.sourcefind.cn:5443/dcu/admin/base/custom:xllm-dev-dcu-x86-20260617"
docker pull "$XLLM_IMAGE"
```

```bash
docker run -it \
  --name xllm-dcu \
  --ipc=host \
  -u 0 \
  --privileged \
  --network=host \
  --shm-size=256g \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mkfd \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v /opt/hyhal:/opt/hyhal \
  -v "$HOME:$HOME" \
  -w "$HOME" \
  "$XLLM_IMAGE" \
  /bin/bash
```

The container remains after exit. Re-enter it with `docker start -ai xllm-dcu` when stopped, or `docker exec -it xllm-dcu bash` while it is running.

## Install xLLM

Run inside the container. If your image already includes the xLLM wheel and supports `xllm serve`, skip the source installation. Otherwise, clone the repository (or enter your existing checkout), initialize its submodules, and build and install the wheel for this backend:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm

# Install pre-commit on first use
pip install pre-commit
pre-commit install

git submodule update --init --recursive
python setup.py install --device dcu
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

export HIP_VISIBLE_DEVICES=0

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
  --block_size=128
```

For two devices on one machine, replace the device selection and `NNODES` lines in the script with:

```bash
export HIP_VISIBLE_DEVICES=0,1
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

- The container mounts the host device nodes and `/opt/hyhal`; keep the host driver and container runtime compatible.
- Build the wheel with `--device dcu`. Runtime device selection uses `HIP_VISIBLE_DEVICES`.
- The launch example uses `--block_size=128`; choose memory and parallelism settings according to the target model.
