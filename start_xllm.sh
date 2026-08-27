#!/bin/bash
set -e

rm -rf core.*

python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
# export ASCEND_RT_VISIBLE_DEVICES=12
export HCCL_IF_BASE_PORT=43428  # HCCL communication base port


MODEL_PATH="/home/models/GLM-5.2-w8a8"
MASTER_NODE_ADDR="127.0.0.1:9900"
START_PORT=29000
START_DEVICE=0
LOG_DIR="log"
NNODES=16

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  /home/zhangxu/xllm/build/lib.linux-aarch64-cpython-311/xllm/xllm \
    --model $MODEL_PATH \
    --model_impl="python" \
    --python_model_path="/home/zhangxu/xllm" \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --block_size=128 \
    --max_tokens_per_batch=81920 \
    --max_seqs_per_batch=2048 \
    --max_memory_utilization=0.8 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=false \
    --kv_cache_dtype="int8" \
    --indexer_cache_dtype="int8" \
    --node_rank=$i > $LOG_FILE 2>&1 &
done
