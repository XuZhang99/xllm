<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# 在 Perfetto 中查看与分析

## 打开真实文件

1. 用浏览器控制工具打开 `https://ui.perfetto.dev`，通过 **Open trace file**
   选择拉回本地的 timeline，或使用工具支持的文件拖放。
2. 等待解析完成，确认存在非空时间轴、进程/线程或 device stream 轨道以及
   可点击的事件。记录加载告警；只出现网站欢迎页不算已打开 trace。
3. 先看总览，再定位稳定请求/step；搜索实际出现的 kernel、HCCL、runtime API
   或 MSTX 标记。固定所需 tracks、放大区间，点击 slice 检查 start/duration/args。
   保存总览和关键区间截图，文件名包含 rank、阶段和区间。
4. 保留本地原始 trace，不使用 Share/公开上传作为打开文件的必要步骤。
   网站 URL 本身不包含本地 trace，交付时需附真实本地文件路径。

优先查看一个代表性 rank；需要解释通信拖尾或负载不均时再查看其他 rank。
不能靠手工拼接 JSON 制造多 rank 全局时间轴。

## 大文件或文件选择工具不可用

Perfetto 支持本地 native Trace Processor 供网页连接。查当前工具的帮助和
[大 trace 官方说明](https://perfetto.dev/docs/visualization/large-traces)，在
**运行浏览器的本地机器**启动，例如：

```bash
# 在独立工具目录下载；先检查已有安装，避免覆盖现有文件
curl -fL https://get.perfetto.dev/trace_processor -o trace_processor
chmod +x trace_processor
./trace_processor --httpd /absolute/path/to/msprof_timestamp.json
```

官方当前也提供 `trace_processor server http <trace>` 写法；以安装版本帮助为准。
打开 Perfetto 并选择检测到的本地 accelerator，确认它加载的正是目标 trace。
默认连接本机 `127.0.0.1:9001`，不要把服务绑定到公网来绕过文件选择。
浏览器工具如果在另一机器执行，先确认其 localhost 与文件所在机器是否一致；
没有可达连接时明确报告限制。任务结束后停止本次创建的 trace processor。

## 用 SQL 辅助复核

在 Perfetto Query/SQL 面板先检查 `slice` 与 `track` 的实际内容。下例仅列出
按 track/name 聚合的热点，不能直接把结果当作 kernel-only 占比：

```sql
SELECT s.track_id, t.name AS track, s.name,
       COUNT(*) AS calls,
       SUM(s.dur) / 1e6 AS total_ms,
       AVG(s.dur) / 1e3 AS avg_us
FROM slice AS s
LEFT JOIN track AS t ON t.id = s.track_id
WHERE s.dur > 0
GROUP BY s.track_id, t.name, s.name
ORDER BY total_ms DESC
LIMIT 40;
```

Perfetto SQL 的 `ts/dur` 单位为 ns；源 Chrome JSON 常用 us，不能混用。
选择具体 rank/device/stream 和阶段后再过滤 `track_id`、`ts` 范围。
跨区间边界 slice 的统计需裁剪到选定窗口；嵌套 scope 与并发 stream 不能直接
求和推导 busy time。看不到 slice 时检查导入告警和原始事件，不能输出空表结论。

Ascend JSON 可能触发 `slice_spill_overlapping_complete_event`：同一 thread 的
完整事件存在不能嵌套的重叠，Perfetto 会将其放入 overflow tracks。记录实际
导入器说明和计数，核对源事件与导入 slice 数；不要通过删事件消除告警，也不要
将显示层的 overflow 直接解释为实际新增 stream 或额外并行度。

## 记录可复查的证据

每个瓶颈使用以下结构记录到 `timeline_notes.md`：

```text
Trace / SHA-256 / rank / device:
阶段与判定依据:
选定 tracks:
窗口 [start, end] 与单位、时间原点:
观察: 事件名、调用次数、duration、gap 或 overlap
截图路径 / SQL 与过滤条件:
解释: 已证实的事实、候选原因、尚缺证据
下一步: 涉及源码位置、可验证改动、无 profiling 对照方案
```

对 decode gap，要列出前后 device task、这段时间其他 streams 是否忙、host
在做什么以及是否存在同步等待。对 HCCL，区分总通信时间与未被计算覆盖的
通信时间；对 graph replay，区分首次 capture/编译与稳定 replay。
无法测量的字段写未覆盖，不用固定百分比阈值代替因果判断。

界面操作参考：[Perfetto UI 官方文档](https://perfetto.dev/docs/visualization/perfetto-ui)。
