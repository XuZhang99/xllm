<!-- Copyright 2026 The xLLM Authors.
SPDX-License-Identifier: Apache-2.0
-->

# DenseBlockTableTranslate

## 接口

`prepare([X, M], {})` 返回可以重复执行的对象。每次 `prepared.run()` 读取原张量的当前值并返回新的 ND 连续 `torch.int32` 张量。`run([X, M], {})` 执行两步。

`X` 为 `[R,C]`，`M` 为 `[B]` 或 `[L,B]`。两者须是同设备、连续的 NPU INT32 张量，可带非零 storage offset。输出分别为 `[R,C]` 或 `[L,R,C]`，因此二维 `M` 的首维为 1 时，输出仍保留层维。

`prepare` 验证输入数量、属性、格式、形状、设备以及 INT32 元素上限。值域通过 NPU 归约和标量回读验证，要求 `min(M) >= 0`，并在 `X` 非空时要求 `max(X) < B`。此后调用方可原地更新 `X` 和 `M`，并保持其形状、类型、布局、设备和值域约束。

## 实现

令 `N = R*C`，`i = r*C + c`。当 `X[i] >= 0` 时，层 `l` 的物理页号为 `M[l*B + X[i]]`，写入 `output[l*N + i]`。当 `X[i] < 0` 时写入 `-1`。逻辑地址和物理页号均以 INT32 存储，查得的物理页号按原值写回，支持 `2147483647`。

- 每层映射最多 4,096 项、层数不超过 40、输出不超过 1,048,576 项且 `N` 为 8 的倍数时，Triton 将每层映射加载到片上缓冲区，同一程序处理该层的多个 tile 时复用映射。为兼容 Ascend Triton 的数据类型限制，`tl.gather` 查表时用 bitcast 传递完整的 32 位页号。层边界按 32 字节对齐，且尾部有写掩码。额外的全局张量只有输出。
- 其他非空形状由 Triton 把负地址暂映射到 0，同时生成原生 Gather 接收的 INT64 索引。C++ 扩展调用 torch_npu 的原生 `index_select` 沿 `M` 的页维查表。Triton 在连续的输出上按 `i = offset % N` 读取逻辑地址，原地把负地址对应值恢复为 `-1`。新增张量为输出及每调用的长度 `N` 的 INT64 安全索引，PyTorch/CANN 还可分配算子工作区。
- `R=0` 或 `C=0` 时返回新的空输出，形状仍遵循接口约定。

Prepared 对象持有两项输入、整数元数据和已初始化的 Triton 启动入口。Triton 在 prepare 阶段完成编译，稳态调用由 C++ 扩展分配输出并向输入设备的当前 stream 提交设备工作，同时将输入存储记录到该 stream。调用方在跨 stream 更新和消费输入时使用普通的 PyTorch stream 依赖。输出及临时张量按次分配，调用方可同时持有任意多份输出。

## 构建和验证

准备与 CANN 兼容的 PyTorch、torch_npu、triton-ascend，以及 C++ 编译器、Ninja 和 setuptools。进入本目录构建扩展。

```sh
MAX_JOBS=2 python setup.py build_ext --inplace
```

在主机上用 `npu-smi info` 确认一张物理卡空闲。设 `CARD` 为该物理卡号，`FIXTURES` 为任务发布包目录，其中包含原始的 `case.json`、`op.json`、`generate_data.py` 和 `golden.py`。以下命令使用容器内逻辑卡 0。输出目录须为新目录。

```sh
ASCEND_RT_VISIBLE_DEVICES="$CARD" python verify.py \
  --fixtures "$FIXTURES" --output-dir /path/to/new-validation-run
```

`functional.json` 和 `functional.log` 保存原始验收脚本输出。它覆盖 14 个用例、动态更新及内存检查。外层验证器在首个性能用例补跑 7 项非法输入检查，逐例验证直接 `run`，并对提交实现与基线分别验证非默认 stream、六份独立输出及回收。主性能记录写入 `result.json` 的 `benchmarks[0:2]`，补充诊断结果写入 `generalization.json`。每轮 200 次调用，计时包括轮末同步，基线与实现按轮交替先后次序。报告包含全部 6 轮、均值、中位数、配对 bootstrap 区间及逐项判定。校验错误或主性能判据失败时验证器返回非零退出码。

正式验收环境为 910C/A3、CANN 9.1.0。复核时以 `result.json` 中记录的 NPU 型号、版本、commit 和源码校验和为准。
