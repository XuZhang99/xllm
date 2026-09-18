<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# NPU 采集、导出与拉回

以下命令中的 PID、scripts 路径和 run 目录需要从现场确认后赋值。
命令在注明的环境执行；不要混用宿主机 PID 和容器 PID。

## 1. 确认执行环境

```bash
# 本地
ssh xu@198.186.3.2
# 远程宿主机：确认挂载，再进入容器
sudo docker inspect zx-xllm-npu --format '{{json .Mounts}}'
sudo docker exec -it zx-xllm-npu bash
# 容器内
command -v msprof
msprof --help
ps -eo pid,ppid,args
npu-smi info
```

不要复制文档里的示例 PID。结合进程树、启动日志、命令行及 `/proc/<pid>/exe`
确认目标 xLLM 父进程，不能直接取 `npu-smi` 中的 worker PID。
本地和远程分别检查 `git remote -v`、`git branch --show-current`、`git rev-parse HEAD`
及 `git status --short`；本次相关 dirty 文件需要内容校验，HEAD 相同并不代表代码相同。

查明远程 scripts 在容器内的路径后，阅读 `start_xllm.sh`、`run_xllm.sh`。
旧脚本只作为配置线索：核对 binary、`python_model_path` 和当前分支是否匹配，
并检查缓存 dtype 等参数的平台支持。例如部分 NPU 版本会拒绝
`kv_cache_dtype=int8`，不能因为旧脚本写了此参数就直接沿用。验证 profiling
链路时优先使用当前版本支持的默认配置；复用预编译 binary 时记录其来源与
哈希，明确这不等于验证了 skill 分支的重新构建结果。
服务启动环境需包含 `export PROFILING_MODE=dynamic`。通过既有启动脚本启动，
并确认它没有清掉该变量；可检查目标进程 `/proc/<pid>/environ` 中此变量，
不要输出整个 environ。已启动且没有该变量的服务需要按任务授权安排重启。
仅启动 profiler 不能弥补此前的启动配置。

Python 模型路径可能在首次请求时才创建额外的 HCCL 通信组。服务端口已就绪
不代表通信组可用；固定 `HCCL_IF_BASE_PORT` 可能在此时触发绑定冲突。
沿用已验证配置，若日志报告 `Communication_Error_Bind_IP_Port`，核对具体
IP/端口与进程，检查是否应使用 HCCL 自动选端口，不终止其他任务来释放端口。

## 2. 预热后开启有限窗口

先使用 `run_xllm.sh` 或当前任务的 workload 发成功请求，完成预热。
核对脚本能检测 HTTP/服务错误，不能把 curl 返回了错误 JSON 视为请求成功。
记录实际输入/输出长度、并发、prefix-cache 命中条件与是否提前 EOS。

在容器交互终端 A 中：

```bash
# 填写已确认的数值和容器内独立输出目录
XLLM_PARENT_PID=12345
PROFILE_RUN=/path/to/profiling/run_YYYYMMDD_HHMMSS
mkdir -p "$PROFILE_RUN"
set -o pipefail
msprof --dynamic=on --pid="$XLLM_PARENT_PID" \
  --output="$PROFILE_RUN" --model-execution=on \
  --runtime-api=on --aicpu=on 2>&1 | tee "$PROFILE_RUN/capture.log"
```

以上为 `xllm-workflow` 的动态采集参数；先用已安装版本的 `msprof --help`
确认支持，不支持时查对应版本工具文档，不能静默删除关键参数继续声称采集完整。

等待 attach ready 后，在终端 A 输入 `start`。确认采集已开始，在同容器终端 B
执行正式 workload，保存输出和退出码到 `workload.log`。正式请求全部完成后，
在 A 输入 `stop`，确认停止，再输入 `quit`，等 msprof 退出并完成数据刷盘。
输入命令及其时间也要记入 capture.log/manifest（终端输入未必被 tee 记录）。
不要依赖固定 sleep 代替 readiness。请求失败也要结束本次采集并保留失败证据。

用工具自动化时保留交互终端 session，逐步写入上述控制命令；不要把它们一次性
通过管道全发完。若采用 FIFO 脚本，脚本放远程 scripts 中，确保异常退出时停止
本次采集器、关闭 FIFO 并等待退出后再 export，不直接照搬先 export 后清理的顺序。

## 3. 显式导出 timeline

枚举本次 run 下每个 `PROF_*`，逐个执行并记录退出状态：

```bash
msprof --export=on --output="$PROFILE_RUN/PROF_actual_name" \
  > "$PROFILE_RUN/export-PROF_actual_name.log" 2>&1
```

不要仅选“最新”目录而丢掉其他 rank。完成后可用：

```bash
find "$PROFILE_RUN" -type f \( -name 'msprof_*.json' \
  -o -name 'trace_view.json' -o -name '*.pt.trace.json' \)
```

常见位置为 `PROF_*/mindstudio_profiler_output/msprof_*.json`，旧版本也可能
位于 `timeline/`。按实际文件识别，不硬编码目录存在即成功；有 CSV 没有
timeline 时检查 export 日志和版本支持的 timeline 导出选项。
`step_trace_*.json` 仅有 step 信息时不能代替完整 kernel timeline。

保留原始 JSON，不随意缩放时间戳。Chrome Trace 常见结构是事件数组或含
`traceEvents` 数组的对象；检查有 `ph`、`ts`、`pid`、`tid` 的时间事件，
完整 slice 通常为 `ph=X` 且有 `dur`，也可能使用 `B/E` 配对。
仅 JSON 解析成功还不够，下一步需在 Perfetto 中验证真实 tracks/slices。

## 4. 拉回本地

先在 manifest 中列出容器路径、宿主机对应路径和 rank/device。bind mount 内
的产物可直接 scp；不在挂载内则在远程宿主机用
`sudo docker cp zx-xllm-npu:/container/path /host/staging/path`
暂存本次产物，并使当前用户可读，不批量改原始数据权限。

```bash
# 本地：示例中路径替换为已确认的宿主机导出目录
mkdir -p /Users/xu/xllm-profile-artifacts/run_YYYYMMDD_HHMMSS/timelines
scp -r xu@198.186.3.2:/host/path/to/exported_device_directory \
  /Users/xu/xllm-profile-artifacts/run_YYYYMMDD_HHMMSS/timelines/
```

保留每个 rank/device 的目录，避免同名文件互相覆盖。远程用 `sha256sum`，
macOS 本地用 `shasum -a 256` 校验所选 timeline，记录字节数和哈希。
原始 PROF 可保留在远程，不必为了查看一个 rank 拉取全部大文件。

导出命令与格式参考：[Ascend msprof 文档](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0005.html)。
