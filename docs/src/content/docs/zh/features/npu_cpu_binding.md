---
title: NPU CPU 绑核
---

xLLM 支持为 Ascend 独立服务进程及其运行时线程分配专用 CPU 集合：

```bash
xllm --model /path/to/model --enable_cpu_binding=true ...
```

开关默认关闭；JSON 配置同样支持 `enable_cpu_binding`，显式命令行参数优先。性能基线使用同一二进制，并设置 `--enable_cpu_binding=false`。

## 分配策略

实现参考 [vLLM Ascend CPU binding 设计](https://docs.vllm.ai/projects/ascend/en/v0.23.0/developer_guide/Design_Documents/cpu_binding.html)。

- **A3：全局切分。** 根据 `npu-smi info -m` 的完整逻辑 NPU 清单，切分启动时允许使用的 CPU。不可见 NPU 仍保留自己的份额，因此使用相同 cpuset、各自仅暴露一张不同 NPU 的进程不会分配到相同 CPU。
- **A2：拓扑亲和性。** 将 NPU 的 CPU 亲和性与启动 cpuset 取交集；如果只涉及一个 NUMA 节点，扩展到 cpuset 内的下一个节点，再对共享相同 CPU 集合的所有 NPU 均分。没有拓扑时使用全局切分；拓扑不完整或分组部分重叠时跳过绑核。
- **设备编号。** 保留 `ASCEND_RT_VISIBLE_DEVICES` 的原始顺序，例如 `12,3` 下的运行时设备 `0` 对应全局逻辑 NPU `12`。
- **线程角色。** 普通线程使用 CPU 池中除最后两个 CPU 外的所有 CPU；`acl_thread` 使用倒数第二个，`release_thread` 使用最后一个。新线程继承创建者的亲和性。运行时初始化、权重加载和首次模型 forward 后重新应用亲和性，覆盖延迟创建的运行时线程。

在 640 核、16 张 NPU 的 A3 上，NPU `i` 分配 `40*i` 至 `40*i+39`：38 个普通线程 CPU，1 个 ACL CPU，1 个 release CPU。每张 NPU 至少需要 3 个允许使用的 CPU，支持非连续 cpuset 和余数分配，不扩张启动 cpuset。

在加载模型和 pinned host buffer 前，启动线程设置 NUMA 首选内存节点，并尝试迁移已有内存页，后续创建的线程继承该策略。权限不足时记录日志，CPU 绑核仍可执行。首选内存策略允许跨节点分配，并非严格内存绑定。

## 可选 IRQ 绑定

```bash
xllm --model /path/to/model \
  --enable_cpu_binding=true --enable_npu_irq_binding=true ...
```

`enable_npu_irq_binding` 默认关闭，依赖 `enable_cpu_binding`。开启后 CPU 池前两个 CPU 用于当前 NPU 的 SQ/CQ IRQ，普通线程使用 `pool[2:-2]`，每张 NPU 至少需要 5 个 CPU。需要可解析的 PCI/MSI 中断信息以及可写的 `/proc/irq/*/smp_affinity_list`。前置条件不足时记录日志并跳过 IRQ 修改，保留线程绑核。xLLM 不停止 `irqbalance`；IRQ 亲和性是主机设置，进程退出后仍然保留。

## 验证

日志输出全局逻辑 NPU ID、分配策略、普通线程 CPU 集合、ACL/release CPU、线程数量以及内存策略结果。发送预热请求后检查实际掩码：

```bash
# PID 为待检查的 xLLM rank 进程。
taskset -apc "$PID"
cat /proc/"$PID"/task/"$TID"/comm
cat /proc/"$PID"/task/"$TID"/status
```

性能对比采用预热后的相同负载，保持模型、并行配置、图模式、缓存、输入和并发一致。建议按关闭/开启/关闭顺序测试并报告基线漂移。共享开发机启动测试前同时检查 NPU 进程归属、利用率和 HBM。

## 支持范围

- Linux aarch64 上 A2/A3 独立服务，每个进程一个 NPU worker；覆盖 native 和 Python 模型执行。离线嵌入与单进程多设备模式不初始化本策略。
- CPU 掩码上限为 `CPU_SETSIZE`（通常 1024）；无法读取的超大掩码会跳过。Ascend 950 需要独立的 cluster/UVB 策略，当前明确跳过。
- `npu-smi`、`lscpu` 查询有超时限制。CPU 不足、设备映射失败、亲和性系统调用不可用时记录警告；线程修改失败时尝试恢复原掩码。
- CPU 编号或 cpuset 跨 NUMA 时，全局切分的 CPU 池仍可能跨 NUMA。不同且相互重叠的 cpuset 之间没有全局协调。
