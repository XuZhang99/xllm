---
name: xllm-npu-profiler
description: 采集和分析 xLLM 昇腾 NPU profiling，导出可在 https://ui.perfetto.dev 打开的 timeline，定位 prefill/decode、算子、通信重叠和 host 下发瓶颈。用于 profiling、性能时间线采集、已有 Ascend trace 的 Perfetto 可视化分析；不用于调度器时延预测数据的采样。
---

<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# xLLM NPU Profiling 与 Perfetto

完成链路：预热 → 有限窗口采集 → 导出 timeline → 拉回本地 → 在
[Perfetto](https://ui.perfetto.dev) 打开 → 根据具体时间区间给出诊断。
用户已提供 trace 时直接从格式检查和可视化开始，不必重新启动服务。

## 采集前

- 遵循仓库 `AGENTS.md`。默认本地 `/Users/xu/xllm`，远程
  `ssh xu@198.186.3.2`，宿主机仓库 `/home/xu/xllm`，容器
  `zx-xllm-npu`。以当前用户指示覆盖默认值。
- 在本地修改，需要执行的脚本放在开发机 `scripts` 目录。先查明容器挂载、
  仓库和 scripts 的实际路径，再同步本次文件；核对两端 remote、branch、HEAD
  和相关未提交文件内容。保留无关修改，不 reset、clean 或覆盖整个工作区。
- 从启动脚本 `start_xllm.sh` 和请求脚本 `run_xllm.sh` 读取实际模型、端口、
  设备和 workload。`eval_scope.sh` 用于精度验证，不作为默认 trace workload。
  必须构建时在容器仓库执行 `python setup.py build`，记录实际运行 binary 路径。
- 检查 `npu-smi info`、已有服务和输出盘空间；不要停止他人的作业。
  记录设备型号、CANN/msprof 版本、TP/DP/EP/CP、graph、输入输出 token 数和并发。
  不从机器别名推断芯片型号。
- 为每次采集新建独立 run 目录，不清空旧数据。确认服务 readiness、成功请求和
  graph/编译预热已完成，随后开始正式采集。

## 选择采集路径

首选 [NPU 采集与导出步骤](references/capture.md)：服务启动前设置
`PROFILING_MODE=dynamic`，在同一容器 PID 命名空间 attach 到服务父 PID，
用 `start/stop/quit` 控制 msprof，采集器退出并刷盘后导出每个 `PROF_*`。
环境变量只在客户端或采集器设置不能追溯改变已启动服务。

每次先检查 `xllm/core/runtime/worker_impl.cpp` 的
`WorkerImpl::start_profile/stop_profile` 平台分支。
本 skill 创建时 NPU 分支不支持在线 HTTP profiler；不能照搬 CUDA 文档中的
`/start_profile`、`/stop_profile` 或假设 `--profile_dir` 会生成 NPU trace。
`enable_profile_step_time` 和 `ProfileManager` 的时延表也不是设备 timeline。

采短窗口并保留实际部署配置。根据问题分别采 prefill-focused（长输入短输出）
或 decode-focused（足够多的稳定 decode steps）；后者仍包含 prefill，分析时
必须选出 decode 区间。仅为算子映射需要时补采 eager trace，不能用它代替正式 graph trace。

## 导出与可视化

1. 按 [capture.md](references/capture.md) 导出，枚举所有 rank/device 的文件。
   保留原始 `PROF_*` 和日志，优先选择完整的 `msprof_*.json`；已有
   `trace_view.json`、`*.pt.trace.json` 则检查其真实事件格式。
2. 检查文件非空、JSON 可解析且有带时间戳的事件。CSV、数据库和仅有
   metadata 的 JSON 不能当作可用 timeline。记录文件大小、SHA-256 和 rank/device。
3. 拉回本地，按 [Perfetto 分析步骤](references/perfetto.md) **实际打开文件**，
   检查时间范围、Host API、device streams、kernel 和通信轨道。
   有浏览器控制工具时主动完成网站操作；使用工具支持的本地文件选择，或文档中的
   native trace processor 方式。工具无法导入时给出准确本地文件路径与手动步骤，
   并标记“已导出，尚未完成 UI 验证”，不要把打开空网站算作完成。
4. 对代表性 prefill 和稳定 decode 区间分别截图和测量。每个结论附
   trace 文件、rank/device、track、起止时间、单位及相关事件名。

## 诊断规则

- 先核对采集覆盖范围。缺少 CPU、HCCL 或某 rank 轨道时写“未采到”，不能判定
  该活动不存在；无法区分阶段时不要编造 prefill/decode 标签。
- 按 kernel 名聚合调用数、总时长、均值，并结合所在 stream 和阶段解释热点。
  CPU scope、runtime API 和 device kernel 不可一起相加当成设备耗时。
- 通信重叠以同一时钟、同一区间内 compute/comm 事件交集为依据。
  多 stream 总时长可能超过墙钟时间；以区间并集计算 busy/idle，注明分母。
- host bubble 要同时看前一 device task 结束、下一 task 开始、中间 Host API、
  同步/拷贝/graph replay 和其他 streams。空白区或单一阈值不足以证明 hostbound。
- rank skew 需要匹配同一次请求/step 并核对时钟。不能仅凭单 rank trace
  判断整个 TP/EP 组；不要直接拼接独立 JSON 导致 pid/tid 或时钟冲突。
- timeline 不能单独证明 KV 碎片、HBM 带宽利用率或融合机会；补充对应指标、
  算子形状和当前源码后再下结论。事实、推测和验证方案分开写。
- Profiling 用于解释瓶颈；用户可见性能提升需要同配置、关闭 profiling 的
  before/after 实测，不能从算子累计耗时直接推导吞吐提升。

## 交付

将以下内容放在同一 run 目录，最后返回可点击的本地 timeline 和报告路径：

```text
<run_id>/
  manifest.md           # commit/branch、binary、环境、启动与请求参数、父 PID
  capture.log           # start/stop/quit、错误、退出/刷盘证据
  workload.log          # readiness、预热、正式请求成功与实际 token 数
  export.log
  PROF_*/               # 原始数据和导出产物；可留远程，manifest 写明路径
  timelines/            # 本地副本，保留 rank/device 子目录
  timeline_notes.md     # 瓶颈、具体区间、证据、候选优化与验证方法
  screenshots/          # Perfetto 总览及关键区间；实际完成 UI 分析时保存
```

明确区分“采集完成”“timeline 导出完成”“Perfetto 加载与分析完成”。失败请求、
空 trace、导出错误、预热混入或缺失关键轨道时，保留产物并说明诊断限制；
不能仅凭退出码 0 宣布 profiling 成功。只清理本次创建的采集器和临时资源，
记录本次启动的服务最终状态。

## 参考来源

流程参考 `xllm-workflow` 的 `skills/xllm-npu-profiler` 和 profiling artifact
约定，已改为仓库内自包含说明；不依赖相邻仓库安装或其其他 skills。
具体命令与 Perfetto 官方资料见两个按需读取的参考文件。
