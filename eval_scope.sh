#!/usr/bin/env bash
# pip install evalscope
set -euo pipefail

MODEL="Qwen3-8B"
PORT=8010
HOST="127.0.0.1"
echo "Using API host: ${HOST}"
echo "Using API port: ${PORT}"


evalscope eval \
  --api-url "http://${HOST}:${PORT}/v1" \
  --api-key EMPTY \
  --eval-type openai_api \
  --model ${MODEL} \
  --datasets gsm8k \
  --generation-config '{"do_sample":true,"temperature":0.6, "top_p": 0.95, "max_tokens":1024, "seed":1, "top_k": 20, "extra_body":{"chat_template_kwargs": {"enable_thinking":false, "thinking":false}}}'\
  --dataset-args '{"gsm8k": {"few_shot_num": 4, "few_shot_random": false}}' \
  --eval-batch-size 64