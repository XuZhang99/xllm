<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# 等待所有 NPU 空闲

`wait_npu_idle.py` 只依赖 Python 标准库和 `npu-smi`，可以单独复制使用。
每次查询输出一行 JSON；全部芯片满足条件后立即返回，不要求连续空闲。
默认每 10 秒查询一次。

空闲条件是：健康状态为 `OK`、AICore 利用率为 0、没有运行进程，且每个
芯片的 HBM 占用不超过 10%。HBM 阈值保留了驱动驻留空间，也避免把
“进程列表已空、显存仍未回收”误判为空闲。阈值可通过
`--max-hbm-used-ratio` 调整。

```bash
# 等到 16 个芯片全部空闲后立即运行测试
python3 scripts/wait_npu_idle.py --expected-devices 16 && bash run_tests.sh

# 只检查一次；适合在启动服务前复核
python3 scripts/wait_npu_idle.py --expected-devices 16 --once

# 每 5 秒检查，最多等待一小时，并保留原始快照
python3 scripts/wait_npu_idle.py --expected-devices 16 \
  --interval 5 --timeout 3600 --snapshot-dir /tmp/npu-idle-snapshots
```

在开发容器内运行已同步的独立脚本：

```bash
ssh dev 'sudo docker exec zx-xllm-npu python3 /home/xu/scripts/wait_npu_idle.py --expected-devices 16'
```

## 后台运行

在开发容器中执行下面的命令。`nohup`、标准输入重定向和日志重定向使它
在退出终端或断开 SSH 后继续运行。锁文件防止重复启动同一个监控任务，
PID 文件记录实际 Python 监控进程；检测到全部空闲后进程立即退出。

```bash
cd /home/xu/scripts
nohup bash -c '
  exec 9>wait_npu_idle.lock
  flock -n 9 || exit 75
  echo "$$" > wait_npu_idle.pid
  exec python3 -u wait_npu_idle.py --expected-devices 16
' </dev/null >>wait_npu_idle.log 2>&1 &
```

查看状态和日志：

```bash
cd /home/xu/scripts
ps -p "$(cat wait_npu_idle.pid)" -o pid,etime,args
tail -n 5 wait_npu_idle.log
```

如需提前停止，先用上面的 `ps` 确认该 PID 仍是本次
`python3 -u wait_npu_idle.py` 进程，再执行
`kill "$(cat /home/xu/scripts/wait_npu_idle.pid)"`。正常检测到空闲后无需停止；
PID 文件只用于记录，不能单独证明进程仍在运行。

需要后台等待后自动测试时，应把等待和测试放在同一个后台任务中：

```bash
cd /home/xu/scripts
nohup bash -c '
  exec 9>npu_test_queue.lock
  flock -n 9 || exit 75
  python3 -u wait_npu_idle.py --expected-devices 16 && bash run_tests.sh
' </dev/null >>npu_test_queue.log 2>&1 &
```

将 `run_tests.sh` 替换为实际测试入口；这与单独的监控任务二选一即可。
后台任务的退出码不能通过启动时的 `$?` 判断，应检查日志中的最终状态和
测试结果。锁只防止重复启动当前任务，不会预约 NPU。

返回码：`0` 全部空闲；`1` 单次检查忙碌或等待超时；`2` 单次查询不可用、
快照不完整或缺少命令；`130` 用户中断。等待模式会在查询超时或失败后继续
重试，但不会将失败视为空闲。检测到设备数量或物理 ID 变化也不会放行。

默认监控首次完整快照中的全部芯片；指定 `--expected-devices` 可进一步
防止不完整设备列表被当作完整清单。脚本不终止任何进程，也不占用或锁定 NPU；
它返回的是查询时的状态，共享机器上的服务仍可能同时启动。
