---
title: "GLM5-W8A8"
sidebar:
  order: 2
---
+ Source code: https://github.com/jd-opensource/xllm

+ Available in China: https://gitcode.com/xLLM-AI/xllm

+ Weight download: [modelscope-GLM-5-W8A8](https://www.modelscope.cn/models/Eco-Tech/GLM-5-W8A8-xLLM/files)
  
## 1. Pull the Image Environment

First, download the image provided by xLLM:

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-20260306
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-20260306
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260306
```

**Note**: Performance stress testing has not been performed on A2 machines.

Then create the corresponding container:

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host \
 -v /var/queue_schedule:/var/queue_schedule \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
 -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
 -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
 -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
 -v /var/log/npu/slog/:/var/log/npu/slog \
 -v ~/.ssh:/root/.ssh  \
 -v /var/log/npu/profiling/:/var/log/npu/profiling \
 -v /var/log/npu/dump/:/var/log/npu/dump \
 -v /runtime/:/runtime/ -v /etc/hccn.conf:/etc/hccn.conf \
 -v /export/home:/export/home \
 -v /home/:/home/  \
 -w /export/home \
 quay.io/jd_xllm/xllm-ai:xllm-dev-hb-rc2-x86
```

## 2. Pull the Source Code and Build

Download the official repository and module dependencies:

```bash
git clone https://github.com/jd-opensource/xllm
cd xllm 
git checkout preview/glm-5
git submodule init
git submodule update
```

Download and install dependencies:

```bash
pip install --upgrade pre-commit
yum install numactl
```

Run the build. The executable `build/xllm/core/server/xllm` will be generated under `build/`:

```bash
python setup.py build
```

## 3. Start the Model

### If the service is being started for the first time after the machine has rebooted, run the following script first to initialize the devices

If this is skipped and the NPU has not been initialized, the xLLM process may fail to start.

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### Environment Variables

```bash
##### 1. Configure dependency path environment variables
# export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
# export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
# export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
# export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
# export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"

# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/lib/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD

# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2. Configure log-related environment variables
rm -rf /root/ascend/log/
rm -rf core.*

##### 3. Configure performance and communication-related environment variables
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

export OMP_NUM_THREADS=12
export ALLOW_INTERNAL_FORMAT=1

export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export ATB_CONVERT_NCHW_TO_AND=1
export ATB_LAUNCH_KERNEL_WITH_TILING=1
export ATB_OPERATION_EXECUTE_ASYNC=2
export ATB_CONTEXT_WORKSPACE_SIZE=0
export INF_NAN_MODE_ENABLE=1
export HCCL_EXEC_TIMEOUT=300
export HCCL_CONNECT_TIMEOUT=300
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_BASE_PORT=2864
```

## Startup Command - GLM-5 (W8A8 weights can be started on a single machine)

```bash
BATCH_SIZE=256
# Maximum inference batch size
XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
# Inference entry binary path, which is the build artifact from the previous step
MODEL_PATH=/path/to/GLM-5-W8A8/
# Model path, here using the int8-quantized GLM-5
DRAFT_MODEL_PATH=/path/to/GLM-5-W8A8/GLM-5-W8A8-MTP/
# Exported MTP weights for GLM-5

MASTER_NODE_ADDR="11.87.49.110:10015"
LOCAL_HOST="11.87.49.110"
# Service port
START_PORT=18994
START_DEVICE=0
LOG_DIR="logs"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=32 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --enable_graph_mode_decode_no_padding=true \
    --draft_model=$DRAFT_MODEL_PATH \
    --draft_devices="npu:$DEVICE" \
    --num_speculative_tokens=1 \
    --ep_size=8 \
    --dp_size=1 \
    > $LOG_FILE 2>&1 &
done

# numactl -C xxxxx          Bind cores by affinity. NUMA affinity query command: npu-smi info -t topo.
# --max_memory_utilization  Maximum single-card memory usage ratio.
# --max_tokens_per_batch    Maximum token count per batch. Mainly limits prefill.
# --max_seqs_per_batch      Maximum request count per batch. Mainly limits decode.
# --communication_backend   Communication backend. Options: hccl / lccl. hccl is recommended here.
# --enable_schedule_overlap Enable async scheduling.
# --enable_prefix_cache     Enable prefix cache.
# --enable_chunked_prefill  Enable chunked prefill.
# --enable_graph            Enable aclgraph.
# --draft_model             MTP weight path.
# --draft_devices           MTP inference device, the same as the main model.
# --num_speculative_tokens  Number of tokens predicted by MTP.
```

When the log contains `"Brpc Server Started"`, the service has started successfully.

## Other Optional Environment Variables

```bash
# Enable deterministic computation
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# Enable dynamic profiling mode
# export PROFILING_MODE=dynamic
# \rm -rf ~/dynamic_profiling_socket_*
```

## Startup Command - Two-Machine Startup Example

### Node0 (master)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.110"
START_PORT=15890
START_DEVICE=0
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ ))do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i));  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=4 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --enable_graph_mode_decode_no_padding=true \
    --ep_size=16 \
    --dp_size=1 \
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

#### Node1 (worker)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.111"
START_PORT=15890
START_DEVICE=0
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ ))do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i));  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((DEVICE*40))-$((DEVICE*40+39)) $XLLM_PATH \    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$((i + LOCAL_NODES)) \
    --max_memory_utilization=0.85 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=4 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --enable_graph_mode_decode_no_padding=true \
    --ep_size=16 \
    --dp_size=1 \
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

#### ranktable Example

ranktable configuration guide: https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/hccl/hcclug/hcclug_000014.html

```json
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "11.87.49.110",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "11.86.23.210",
                    "rank_id": "0"
                },
                ...
                {
                    "device_id": "7",
                    "device_ip": "11.86.23.217",
                    "rank_id": "7"
                }
            ],
            "host_nic_ip": "reserve"
        },
        {
            "server_id": "11.87.49.111",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "11.87.63.202",
                    "rank_id": "8"
                },
                ...
                {
                    "device_id": "7",
                    "device_ip": "11.87.63.209",
                    "rank_id": "15"
                }
            ],
            "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```

## View Device NUMA Affinity

Command:

```bash
npu-smi info -t topo
```

In the preceding commands:

```bash
numactl -C $((DEVICE*12))-$((DEVICE*12+11))
```

indicates that the process is bound to the corresponding affinity cores. You can modify the bound core IDs according to the machine.

## EX3. GLM-5 Weight Quantization 

### Install msmodelslim

```bash
git clone https://gitcode.com/shenxiaolong/msmodelslim.git 
cd msmodelslim
bash install.sh
```

### Modify tokenizer_config.json

```bash
  "extra_special_tokens" 
    change to "additional_special_tokens"

  "tokenizer_class": "TokenizersBackend" 
    change to "tokenizer_class": "PreTrainedTokenizer"
```

### Quantize W8A8 Weights from GLM-5-BF16 Weights

```bash
### Preprocess MTP-related weights
python example/GLM5/extract_mtp.py --model-dir ${model_path}

# Specify the transformers version
pip install transformers==4.48.2

# Run quantization and generate quantized weights
msmodelslim quant  --model_path ${model_path}  --save_path ${save_path}  --model_type DeepSeek-V3.2  --quant_type w8a8  --trust_remote_code True

# Copy the chat_template file
cp ${model_path}/chat_template.jinja ${save_path}

# Export quantized MTP weights for xLLM inference
python example/GLM5/export_mtp.py --input-dir  ${int8_save_path} --output-dir  ${mtp_save_path}
```

## PD Disaggregation

### Install etcd and xllm-service

#### PD Disaggregated Deployment

`xllm` supports PD disaggregated deployment. This must be used together with another open-source library, [xllm service](https://github.com/jd-opensource/xllm-service).

##### xLLM Service Dependencies

First, download and install `xllm service`, similar to installing and building `xllm`:

```bash
git clone https://github.com/jd-opensource/xllm-service
cd xllm_service
git submodule init
git submodule update
```

##### Install etcd

`xllm_service` depends on [etcd](https://github.com/etcd-io/etcd). Use the official etcd [installation script](https://github.com/etcd-io/etcd/releases) to install it. The default installation path used by the script is `/tmp/etcd-download-test/etcd`. You can manually modify the installation path in the script, or move it manually after the script finishes:

```bash
mv /tmp/etcd-download-test/etcd /path/to/your/etcd
```

##### Build xLLM Service

Apply the patch first:

```bash
sh prepare.sh
```

Then build:

```bash
mkdir -p build
cd build
cmake ..
make -j 8
cd ..
```

:::caution[Possible Errors]
You may encounter installation errors related to `boost-locale` and `boost-interprocess`: `vcpkg-src/packages/boost-locale_x64-linux/include: No such     file or directory`, `/vcpkg-src/packages/boost-interprocess_x64-linux/include: No such file or directory`.
Reinstall these packages with `vcpkg`:
```bash
/path/to/vcpkg remove boost-locale boost-interprocess
/path/to/vcpkg install boost-locale:x64-linux
/path/to/vcpkg install boost-interprocess:x64-linux
```

:::
### Run PD Disaggregation

Start etcd:

```bash
./etcd-download-test/etcd --listen-peer-urls 'http://localhost:2390'  --listen-client-urls 'http://localhost:2389' --advertise-client-urls  'http://localhost:2391'
```

For cross-machine configuration, refer to the following etcd command:

```bash
/tmp/etcd-download-test/etcd --listen-peer-urls 'http://0.0.0.0:3390' --listen-client-urls 'http://0.0.0.0:3389' --advertise-client-urls 'http://11.87.191.82:3389'
```

Start xllm service:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```

For cross-machine configuration, start xllm service with:

```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ../xllm-service/build/xllm_service/xllm_master_serving --etcd_addr="11.87.191.82:3389" --http_server_port 38888 --rpc_server_port 38889 --tokenizer_path=/export/home/models/GLM-5-W8A8/
```
- Start the Prefill instance
```bash
  BATCH_SIZE=256
  # Maximum inference batch size
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  # Inference entry binary path, which is the build artifact from the previous step
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  # Model path, here using the int-quantized GLM-5
  DRAFT_MODEL_PATH=/export/home/models/GLM-5-MTP/
  
  MASTER_NODE_ADDR="11.87.49.110:10015"
  LOCAL_HOST="11.87.49.110"
  # Service port
  START_PORT=18994
  START_DEVICE=0
  LOG_DIR="logs"
  NNODES=16
  
  for (( i=0; i<$NNODES; i++ ))
  do
    PORT=$((START_PORT + i))
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="$LOG_DIR/node_$i.log"
    nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
      --model $MODEL_PATH  --model_id glmmoe \
      --host $LOCAL_HOST \
      --port $PORT \
      --devices="npu:$DEVICE" \
      --master_node_addr=$MASTER_NODE_ADDR \
      --nnodes=$NNODES \
      --node_rank=$i \
      --max_memory_utilization=0.86 \
      --max_tokens_per_batch=5000 \
      --max_seqs_per_batch=$BATCH_SIZE \
      --communication_backend=hccl \
      --enable_schedule_overlap=true \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_graph=true \
      --draft_model $DRAFT_MODEL_PATH \
      --draft_devices="npu:$DEVICE" \
      --num_speculative_tokens 1 \
      --enable_disagg_pd=true \
      --instance_role=PREFILL \
      --etcd_addr=$LOCAL_HOST:3389 \
      --transfer_listen_port=$((36100 + i)) \
      --disagg_pd_port=8877 \
      > $LOG_FILE 2>&1 &
  done
  
  # --etcd_addr=$LOCAL_HOST:3389  Refer to the advertise-client-urls configuration in etcd.
  # --instance_role=DECODE        PD configuration: DECODE or PREFILL.
  ```

- Start the Decode instance
  
  ```bash
    BATCH_SIZE=256
  # Maximum inference batch size
  XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
  # Inference entry binary path, which is the build artifact from the previous step
  MODEL_PATH=/export/home/models/GLM-5-w8a8/
  # Model path, here using the int-quantized GLM-5
  DRAFT_MODEL_PATH=/export/home/models/GLM-5-MTP/
  
  MASTER_NODE_ADDR="11.87.49.110:10015"
  LOCAL_HOST="11.87.49.110"
  # Service port
  START_PORT=18994
  START_DEVICE=0
  LOG_DIR="logs"
  NNODES=16
  
  for (( i=0; i<$NNODES; i++ ))
  do
    PORT=$((START_PORT + i))
    DEVICE=$((START_DEVICE + i))
    LOG_FILE="$LOG_DIR/node_$i.log"
    nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
      --model $MODEL_PATH  --model_id glmmoe \
      --host $LOCAL_HOST \
      --port $PORT \
      --devices="npu:$DEVICE" \
      --master_node_addr=$MASTER_NODE_ADDR \
      --nnodes=$NNODES \
      --node_rank=$i \
      --max_memory_utilization=0.86 \
      --max_tokens_per_batch=5000 \
      --max_seqs_per_batch=$BATCH_SIZE \
      --communication_backend=hccl \
      --enable_schedule_overlap=true \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_graph=true \
      --draft_model $DRAFT_MODEL_PATH \
      --draft_devices="npu:$DEVICE" \
      --num_speculative_tokens 1 \
      --enable_disagg_pd=true \
      --instance_role=DECODE \
      --etcd_addr=$LOCAL_HOST:3389 \
      --transfer_listen_port=$((36100 + i)) \
      --disagg_pd_port=8877 \
      > $LOG_FILE 2>&1 &
  done
  
  # --etcd_addr=$LOCAL_HOST:3389  Refer to the advertise-client-urls configuration in etcd.
  # --instance_role=DECODE        PD configuration: DECODE or PREFILL.
  ```
  
  Notes:

- PD disaggregation needs to read `/etc/hccn.conf`. Make sure this file on the physical machine is mounted into the container.

- `etcd_addr` must be the same as the `etcd_addr` used by `xllm_service`.
  The test command is similar to the one above. Note that for `curl http://localhost:{PORT}/v1/chat/completions ...`, `PORT` should be the `http_server_port` used to start xLLM service.

- When deploying P or Q across multiple machines, such as deploying two P instances, add `--rank_tablefile` to complete communication.
