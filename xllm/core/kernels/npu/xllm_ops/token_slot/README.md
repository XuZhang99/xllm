<!-- Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Token slot 地址翻译

输入为同一 NPU 上的连续 `torch.int32` 张量 `slots[N]` 和 `M[B]` 或 `M[L,B]`，允许非零
`storage_offset`。属性 `block_size` 是区间 `[1, 2147483647]` 内的整数。负 slot 输出
`-1`。对非负 slot `x`，写入地址为 `M[layer, x // block_size] * block_size + x % block_size`。
输出形状分别为 `[N]` 和 `[L,N]`。算子返回地址，调用方负责写入 KV Cache。

```python
from solution import prepare, run

prepared = prepare([slots, mapping], {"block_size": 128})
physical = prepared.run()
physical_once = run([slots, mapping], {"block_size": 128})
```

`prepare` 校验形状、布局、设备、元素数和属性。数值检查用 NPU 归约，汇总为一次 CPU 回读。
最大非负 slot 必须小于 `B * block_size`。M 的最小值必须非负，
最大值须不超过 `(INT32_MAX - block_size + 1) // block_size`，因此物理页最后一个地址仍在
int32 范围内。形状、设备、类型或页大小变化后须重新 prepare。合法值更新和保持布局的同形状
`set_` 重绑定由后续调用读取当前内容。调用方负责遵守 PyTorch 的跨流依赖和输入存储生命周期。

## 设备实现

一个 Triton program 负责一层中的一段连续 token。页长为二次幂时通过移位与掩码分解
逻辑地址。其余页长令 `R = floor(2**32 / block_size)`，以 int64 计算
`q0 = (x * R) >> 32`。由于 `0 <= x < 2**31`，近似商与精确实数商的误差小于
`1/2`，因此 `q0` 最多低估 1。用 `r0 = x - q0 * block_size` 判断是否给商加 1、
余数减去一个页长。校验保证物理地址合成可在 int32 中完成。

每层的映射行最多 4096 个元素时整行读入片上统一缓冲区，再按逻辑页索引执行
gather。Triton Ascend 3.2.2 的片上 gather 接收 float32，调用前后按位重解释以保留全部
int32 位。更长的映射行直接按索引读全局内存。内核用掩码保护无效 token 和尾块。
输出为空时只分配空张量，其他情况每次分配独立连续张量并发射一次内核。

`prepare` 编译并保留与形状、页长和设备匹配的启动器。关闭指针对齐特化，保证合法的非零
offset 和存储重绑定仍使用同一启动器。每次执行从 Triton Ascend 驱动取得输入所在设备的当前 stream，并向启动器传入当前 Tensor 和 stream。prepared 对象只持有输入引用、输出形状和启动器，设备上仅分配
结果张量。后续调用可以保留前一次返回的输出。

## 复现

在 Atlas A3、CANN 9.1.0 上将一张空闲逻辑 NPU 暴露给进程。xLLM A3 容器所用的 Python
3.12、PyTorch 2.10.0+cpu、torch-npu 2.10.0.post4 和 triton-ascend 3.2.2
是一套可用组合。已有这些依赖的容器可以直接运行。需要配置独立 Python 环境时，使用与平台
和 CANN 匹配的 ARM64 安装包。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-npu==2.10.0.post4
python -m pip install triton==3.5.0
python -m pip install --no-deps triton-ascend==3.2.2
```

从 xLLM 仓库根目录运行。将 `FIXTURES` 设为题包中原样保留的 `case.json`、`op.json`、
`golden.py` 和 `generate_data.py` 所在目录。内核首次 prepare 时由 Triton 编译。
验证使用题包的 CPU 标杆及 NPU 基线，单独运行此工程即可。

```bash
FIXTURES=/absolute/path/to/token_slot
OUTPUT=/absolute/path/to/token_slot_results
mkdir -p "$OUTPUT"
npu-smi info | tee "$OUTPUT/device-before.log"
# Confirm the selected logical device is idle before running the next command.
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$OUTPUT/triton-cache" \
  python -u xllm/core/kernels/npu/xllm_ops/token_slot/verify.py \
  --fixtures "$FIXTURES" --output-dir "$OUTPUT" 2>&1 | tee "$OUTPUT/run.log"
status=$?
npu-smi info | tee "$OUTPUT/device-after.log"
exit "$status"
```

`result.json` 包含精度、工程、主性能结果和正式验收状态，`generalization.json` 包含路由
及宽页用例和两组补充配对计时。`environment.json` 记录软件版本、源码提交及哈希。
两个计时侧分别预热 50 次，交替测量 6 轮，每轮 200 次完整调用并同步。峰值内存为
预热后 PyTorch NPU 分配器可见的单次新增活跃字节，计入输出和临时张量。
在 910B2 上开发复测时可加 `--development`，正式验收状态保留为 `NOT_RUN`。
