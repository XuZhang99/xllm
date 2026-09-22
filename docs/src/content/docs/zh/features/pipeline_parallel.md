---
title: 流水线并行
---

<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

## 实现范围

Python/NPU GLM 路径（`model_type=glm_moe_dsa`）新增实验性 PP × TP 支持。
`pp_size` 默认值为 `1`；16 个 worker 设置 `--pp_size=2` 即 PP2 × TP8，
stage 0 使用 rank 0–7，stage 1 使用 rank 8–15。

stage `s` 持有层区间
`[floor(n_layers*s/pp_size), floor(n_layers*(s+1)/pp_size))`。
每个 stage 只加载本地层权重、分配本地 KV/index cache；首级加载 embedding，
末级加载最终归一化和 LM head。权重名称保留全局层号，缓存使用本地层号。
公共 block 数按占用最大的 stage 计算，并考虑各 stage 实际使用 indexer 的层数。

相同 TP lane 通过 HCCL PP 子组传输 hidden states 和 residual。
当下一级从共享索引层开始时，同时传输上一索引器产生的 int32 top-k。
末级 TP rank 0 采样，engine 将输出写回原有 sequence。

## micro-batch 调度

engine 在 sequence 边界将当前调度批次拆为至多 `pp_size` 个非空子批次，
保留每条 sequence 已分配的 chunk 预算。各 stage 按固定顺序执行子批次，
等待本级所有 TP worker 完成后再准备下一份输入；不同 stage 可同时处理
不同子批次。全部 stage 成功后才应用输出。

单条已调度 sequence 只有一个子批次，无法填满流水线。
当前尚未将同一请求的连续 prefill chunks 交错执行，也未按预测的 stage
耗时分配子批次。因此这仍不是完整的 Dynamic CPP 调度实现。

## 启动选项

在每个 rank 的既有 GLM 启动命令中加入：

```bash
--model_impl=python \
--pp_size=2 \
--dp_size=1 --ep_size=1 --cp_size=1 \
--enable_graph=false \
--enable_schedule_overlap=false \
--enable_shm=false \
--enable_chunked_prefill=true
```

TP 大小由 `nnodes / pp_size` 推导。`pp_size` 必须整除 world size，
且不能超过模型层数。命令行和 JSON 配置均支持该选项。
可叠加 `--enable_dynamic_chunking=true`，但当前启动 profiling 测量整个
流水线的端到端耗时，尚未分别拟合各 stage 的成本。

当前要求在线 NPU Python GLM worker、eager 执行、DP=EP=CP=1，
不启用 layerwise/KV split。明确拒绝 graph、投机解码、EPLB、PD、
主机缓存卸载、外部 KV 存储、XTensor、原有 multi-stream 和 beam-search kernel。

## 验证与后续工作

CPU 测试覆盖层所有权、residual/top-k 跨级一致性、rank 映射、真实四进程传输、
stage 内执行顺序和不均匀分层缓存预算。生产使用前仍须完成 NPU 全模型精度与
性能验证；CPU 通信测试不能证明 HCCL 路径正确。

后续包括逐 stage profiling、按耗时分配子批次、连续 prefill chunk 交错调度、
各 stage 的图捕获，以及更多并行组合和模型适配。

### 验证记录（2026-09-16）

- 容器内 `python setup.py build` 成功。
- Python GLM parallel/CP/indexer 与 collective 回归：67 项通过。
- C++ 调度器 1 项、缓存估算 24 项、配置 26 项，全部通过。
- 已尝试 PP2 × TP8 GLM-5.3 + 动态 chunk 测试，但启动检查发现其他 vLLM
  worker 后停止，尚未启动本次服务。没有本 PP 实现的 NPU 推理、精度或吞吐结果。
- 开发机脚本为 `/home/xu/scripts/pipeline_case.py` 和
  `/home/xu/scripts/run_pipeline_smoke.sh`；支持 `--pp-size`、`--dynamic`、
  `--smoke-only`、`--accuracy`。性能对比应使用 eager TP16 作为对照。
