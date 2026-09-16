---
title: 动态 Chunked Prefill
---

<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

## 范围

动态 chunked prefill 根据启动 profiling 的结果和请求已缓存的前缀长度，
调整每轮 prefill 的 token 数量。默认关闭。

本实现迁移了 vLLM-Ascend Dynamic Chunked Pipeline Parallel 的启动测量、
二次延迟模型和动态 chunk 选择，并接入 xLLM 的 C++ 调度器。
xLLM 当前没有 LLM 流水线并行（PP）执行链路，本功能不提供 PP，
也不代表已经实现完整的 Dynamic CPP。

参考实现：

- [vLLM-Ascend predictor](https://github.com/vllm-project/vllm-ascend/blob/0526083dd4aae02a02afeb0de680c119d8e5e732/vllm_ascend/core/profiling_chunk_predictor.py)
- [SGLang DynamicChunkSizer](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_components/dynamic_chunk_sizer.py)

## 使用

在原有 LLM 启动命令上添加：

```bash
--enable_chunked_prefill=true \
--enable_dynamic_chunking=true \
--max_tokens_per_chunk_for_prefill=4096 \
--dynamic_chunk_min_tokens=256 \
--dynamic_chunk_smooth_factor=1.0 \
--dynamic_chunk_profile_samples=16
```

也可以在配置 JSON 中设置同名字段。`--help` 会列出这些选项，配置导出会
保留非默认值。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `enable_dynamic_chunking` | `false` | 启用启动 profiling 和动态 chunk |
| `dynamic_chunk_min_tokens` | `256` | 对齐及资源约束前的最小预测长度，必须大于 0 且不超过基准 chunk |
| `dynamic_chunk_smooth_factor` | `1.0` | `(0, 1]`；越小越接近固定 chunk |
| `dynamic_chunk_profile_samples` | `16` | 测量长度数量，范围 `[8, 64]` |

基准 chunk 取 `max_tokens_per_chunk_for_prefill` 和 `max_tokens_per_batch`
的较小值。KV/CP 对齐粒度为 `lcm(64, block_size * kv_split_size_effective)`。
至少需要 8 个不同的对齐长度才能拟合；不足时记录警告并使用固定 chunk。
启动 profiling 需要容纳一个基准 chunk 的 KV 和模型上下文空间。

## 算法和调度

1. 复用 `ProfileManager::run_request` 执行真实的 synthetic prefill。
   每个长度先 warmup，再运行 3 次取延迟中位数。请求的 KV 会释放且不写入
   prefix cache。计时沿用 xLLM 的同步结果返回路径，包含执行及调度通信开销。
2. 用非负最小二乘拟合 `f(L) = a*L² + b*L + c`，长度先归一化以改善长序列
   下的数值条件。允许 `a=0` 的线性模型；数据退化、非有限值或无有效长度相关
   成本时使用固定 chunk，不捕获或隐藏模型执行错误。
3. 令目标增量延迟 `T = f(base) - f(0)`，对缓存历史 `H` 求解
   `a*x² + (2*a*H+b)*x = T`。使用稳定的正根公式，再做平滑、最小长度和对齐处理。
4. `PrefillFirstPolicy`、`DecodeFirstPolicy` 和 `UnifiedPolicy` 使用预测上限。
   prefix cache 匹配后按有效缓存长度预测；实际分配仍受全局/DP token 预算、
   KV 空间和既有 SLO 策略限制。混合调度的预算重分配不能突破动态上限。
   短尾块可以不对齐；无法容纳一个对齐块的剩余预算留给下一轮。

这个上限针对每个请求的 chunk，并不保证多请求整个 batch 的耗时固定。
现有延迟感知调度器仍负责其 batch 延迟预算。

## 支持边界

- 接入 `ContinuousScheduler` 及复用该调度路径的 `DisaggPDScheduler`。
  Decode-only 实例跳过动态 prefill profiling。
- `ZeroEvictionScheduler`、`PDOOCScheduler` 不支持此配置；会明确拒绝。
- 含线性 attention 层且启用 prefix cache 时，固定 chunk 边界用于状态检查点，
  暂不支持动态边界，会明确拒绝。
- 本次只在启动时拟合；未迁移 vLLM-Ascend 的在线 history-aware 再校准。
- 不保证长上下文外推的预测精度或吞吐提升。固定/动态模式需要同设备、同模型、
  同负载的 A/B 对比；没有 PP 时不能直接引用原 CPP 的流水线收益。

## 测试和后续工作

相关测试：`dynamic_chunk_predictor_test`、`DynamicChunkConfigTest`，以及
`scheduler_test` 中的动态 chunk 调度回归测试。

后续：

1. 在空闲 NPU 上验证长输入、并发、prefix cache、CP 和 decode graph 的组合，
   比较输出、TTFT、吞吐及实际 chunk 序列。
2. 增加带历史长度的测量和有界在线再校准，明确样本归属和计时同步成本。
3. 独立实现 PP 层分片、rank 通信、多个在途 batch 调度和输出路由，再将本预测器
   接入 PP 的 chunk 调度，验证真实的流水线空闲时间改善。

### 本次迁移的验证记录（2026-09-15）

- 基线：`main` / `beabafad07953629e6af0fa54125126a1349ebc3`。
- 分支：`feat/dynamic-chunked-prefill`；本地及远端改动文件逐一校验一致。
- 在 `zx-xllm-npu` 容器中执行 `python setup.py build`，服务、导出模块和
  全部测试目标编译成功。未改动的 TileLang kernel 缓存经源码依赖、编译器指纹
  和二进制 SHA-256 校验后复用。
- 25 项配置测试、5 项预测器测试和 1 项调度回归测试通过；调度测试覆盖三种
  policy 在正常预算及不足一个对齐块预算下的行为。
- NPU 验证脚本位于开发机 `/home/xu/scripts/validate_dynamic_chunk_npu.py`。
  尝试启动前检查发现设备被其他任务占用，脚本未启动任何服务。因此尚无
  实际 NPU profiling、长输入、prefix cache 或并发推理的验证结果，也没有吞吐收益结论。

### NPU 对比验证（2026-09-16）

在同一 16-NPU Ascend 910C 设备上，用同一二进制和 GLM-5.3-w8a8 权重完成
固定 A1 / 动态 B / 固定 A2 对比。TP=16，DP/EP=1，base chunk 和 batch token
预算均为 8192，最大并发序列数 4，block=128，KV cache 8 GiB，关闭 prefix cache
和 schedule overlap，开启 graph。每轮排除预热后测量 72 个请求，三轮共 216 个
请求全部成功；输入长度 512/8192/16384，输出 128 token，并发 1/4。

- GSM8K 固定 200 题、4-shot、temperature=0、max_tokens=1024：固定与动态均为
  198/200（99%），逐题对错一致。两轮均有同一道题达到输出上限，回答文本并非
  逐字一致。这是子集初筛，不是完整 GSM8K 或所有长上下文的精度证明。
- 每轮的 5 个功能检查通过，包含 15440-token 文本取回及并发请求。
  动态 predictor 成功拟合；该长输入实际 chunk 为 8192/3584/2816/848。
- 相对 A1/A2 吞吐均值，动态变化分别为：512-token 并发 1/4 为 -0.1%/+0.5%，
  8192-token 为 -0.3%/+1.5%，16384-token 为 -4.1%/+0.2%。
- 16384-token 单请求 TTFT 中位数：固定约 5.96 秒，动态约 6.47 秒（+8.5%）；
  两次固定基线基本一致。当前配置未显示明确的整体吞吐收益，应保持默认关闭。
- 测试脚本已补充清理脱离父进程组的本次 worker，解决轮次切换时的显存残留。
  此次没有修改模型或调度实现；尚未验证 prefix cache、CP 及 PP 组合。

完整请求、profiling、逐题评分及 A/B/A 报告位于开发机
`/home/xu/scripts/dynamic-chunk-comparison-20260916-133244/`。
