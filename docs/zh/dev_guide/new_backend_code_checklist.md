# xLLM 新后端代码改造清单

本文是对 `docs/zh/dev_guide/adapt_new_chip.md` 的补充，目标不是解释原理，而是提供一份更偏工程实施的改造清单。建议把你的目标后端先命名为 `XPU`，实际落地时再替换成具体芯片名。

建议推进方式：

- 先做 “能编译、能识别设备、能单卡生成”
- 再做 “补关键算子、补模型、补性能”
- 最后做 “多卡、图模式、KV 和高级特性”

---

## 0. 改造目标定义

首版建议只追一个里程碑：

**单机单卡、单模型、离线生成功能可用**

更具体一点：

- 能构建 `xllm`
- 能识别 `--devices="xpu:0"`
- 能加载一个 decoder-only 模型，例如 `qwen3`
- 能完成 prefill + decode
- 能返回合理 token

不要一开始把目标设成：

- 多机多卡
- MoE
- VLM
- DiT
- 图模式
- speculative / MTP

---

## 1. 构建层改造

### 1.1 根 CMake 开关

文件：

- `CMakeLists.txt`

待办：

- 新增 `option(USE_XPU "Enable XPU support" OFF)`
- 在 `project(...)` 前后接入你的编译器、toolchain、runtime、include、lib
- 如果需要单独语言前端或特殊编译 flags，参考 `USE_MUSA`
- 如果只是普通 C++ runtime，参考 `USE_NPU` / `USE_MLU`

验收标准：

- `cmake -DUSE_XPU=ON ...` 能通过配置
- 不会影响现有 `USE_NPU/USE_CUDA/USE_MLU/USE_MUSA/USE_ILU`

### 1.2 子模块 CMake

文件：

- `xllm/core/kernels/CMakeLists.txt`
- `xllm/core/layers/CMakeLists.txt`
- `xllm/core/platform/CMakeLists.txt`

待办：

- 在 `core/kernels/CMakeLists.txt` 中为 `USE_XPU` 增加 `add_subdirectory(xpu)`
- 把 `xpu_kernels` 链接到 `kernels` 总目标
- 在 `core/layers/CMakeLists.txt` 中为 `USE_XPU` 增加 `add_subdirectory(xpu)`
- 把 `xpu_layers` 链接到 `layers` 总目标
- 在 `core/platform/CMakeLists.txt` 中给 `platform` 目标增加 XPU runtime 依赖

验收标准：

- `platform` 能链接 XPU runtime
- `kernels` 能链接 XPU kernel 库
- `layers` 能链接 XPU layer 库

### 1.3 新增后端目录

建议新增目录：

```text
xllm/core/kernels/xpu/
xllm/core/layers/xpu/
```

```text
xllm/models/llm/xpu/
```

如果需要单独平台扩展，也可以新增：

```text
xllm/core/platform/xpu/
```

验收标准：

- 新目录能被 CMake 接入
- 空实现状态下先能通过基础编译

---

## 2. 设备与平台层改造

### 2.1 设备识别

文件：

- `xllm/core/platform/device.h`
- `xllm/core/platform/device.cpp`
- `xllm/core/util/device_name_utils.cpp`

待办：

- 在 `device.cpp` 增加 `#elif defined(USE_XPU)` 分支
- 接入：
  - `Device::device_count()`
  - `Device::type_str()`
  - `Device::type_torch()`
  - `Device::set_device()`
  - `Device::init_device_context()`
  - `Device::empty_cache()`
  - `Device::synchronize_default_stream()`
  - `Device::current_stream()`
  - `Device::get_device_mem()`

关键决策：

- 设备字符串是否用 `xpu`
- PyTorch 对应设备类型是 `torch::kPrivateUse1`、专有类型，还是复用别的类型

注意事项：

- `DeviceNameUtils::parse_devices(...)` 会校验设备前缀是否等于 `Device::type_str()`
- 一次构建产物默认只支持一种主设备类型

验收标准：

- `--devices="xpu:0"` 能解析
- `Device::device_count()` 返回正确设备数
- `torch::Tensor(...).to(torch::Device("xpu:0"))` 可用

### 2.2 Stream 抽象

文件：

- `xllm/core/platform/stream.h`
- `xllm/core/platform/stream.cpp`

待办：

- 为 `Stream::Stream(...)` 接入 XPU stream pool
- 为 `Stream::synchronize()` 接入 XPU stream 同步
- 为 `set_stream_guard()` 接入你的 c10 stream 包装
- 为 `wait_stream(...)` 确认 event record/block 行为在 XPU 后端是否成立

注意事项：

- `stream.cpp` 当前对 `CUDA/ILU/MUSA` 和 `NPU/MLU` 的封装方式不一样
- 如果你的后端没有完整 c10 stream 包装，可能需要补 wrapper

验收标准：

- 可以创建 stream
- 可以切换 stream
- 可以等待 stream

### 2.3 内存与 VMM 能力

文件：

- `xllm/core/platform/vmm_api.h`
- `xllm/core/platform/vmm_api.cpp`
- `xllm/core/platform/shared_vmm_allocator.h`
- `xllm/core/platform/shared_vmm_allocator.cpp`

首版建议：

- 如果芯片暂不支持 VMM，先不要强做全量适配
- 先确认哪些路径会强依赖 VMM，再通过 feature flag 关闭

验收标准：

- 单卡基础路径不因为 VMM 缺失而阻塞

---

## 3. 启动入口与参数层改造

### 3.1 主入口启动

文件：

- `xllm/xllm.cpp`

需要确认的点：

- `FLAGS_devices` 能否传入 `xpu:*`
- `Options` 里的设备和 backend 信息能否透传到模型创建和 runtime

通常不需要大改，但要检查：

- 某些 flag 是否写死只支持 NPU
- 某些特性是否需要对 XPU 强制关闭

建议首版显式关闭：

- `enable_graph`
- `enable_prefill_piecewise_graph`
- `enable_graph_vmm_pool`
- `enable_prefix_cache`
- `enable_multi_stream_parallel`
- `enable_xtensor`
- `enable_manual_loader`
- `enable_rolling_load`

验收标准：

- 启动时不会因为 feature 校验直接 `LOG(FATAL)`

### 3.2 全局 flags 与条件逻辑

文件：

- `xllm/core/common/global_flags.h`
- `xllm/core/common/global_flags.cpp`

待办：

- 检查是否有对设备后端写死的默认值
- 检查是否需要增加 XPU 专属 flag
- 检查某些 flag 的帮助文本是否要补充 XPU

验收标准：

- 帮助信息与实际后端能力一致

---

## 4. kernel 分发层改造

### 4.1 新增 XPU kernel API

文件：

- `xllm/core/kernels/ops_api.h`
- `xllm/core/kernels/ops_api.cpp`
- `xllm/core/kernels/xpu/xpu_ops_api.h`

待办：

- 在 `ops_api.cpp` 中加入 `#elif defined(USE_XPU)`
- 将统一 API 分发到 `xpu::xxx(...)`

典型形式：

```cpp
#elif defined(USE_XPU)
#include "xpu/xpu_ops_api.h"
```

以及：

```cpp
#elif defined(USE_XPU)
  xpu::fused_layernorm(...);
```

验收标准：

- `ops_api.cpp` 编译通过
- 未实现接口可以先 `NOT_IMPLEMENTED()`，但最小模型路径不能走到它们

### 4.2 第一批必做算子

先围绕一个简单 LLM 路径实现，不要追求一次覆盖全部 `ops_api.h`。

第一批建议：

- `apply_rotary`
- `fused_layernorm`
- `matmul`
- `reshape_paged_cache`
- attention 相关核心接口
- `apply_top_k_top_p`
- `random_sample`

如果你的第一目标模型不走某些 kernel，可以延后。

验收标准：

- 最小模型推理链路不会掉进 `NOT_IMPLEMENTED()`

### 4.3 第二批算子

当你准备补性能和更复杂模型时再做：

- `group_gemm`
- `scaled_quantize`
- `scaled_matmul`
- `rejection_sample`
- `gather_split`
- `fused_mla_q`
- `fused_mla_kv`
- `fused_indexer_q`
- `fused_indexer_k`

### 4.4 MoE 和 all2all 系列

以下接口不要放进第一阶段：

- `moe_active_topk`
- `moe_gen_idx`
- `moe_expand_input`
- `moe_combine_result`
- `moe_all2all_*`
- `moe_init_routing_v2`

只有当你明确要支持 MoE 模型时再做。

---

## 5. 模型层改造

### 5.1 模型接入方式选择

文件：

- `xllm/models/models.h`

你需要先做一个决定：

- 复用通用模型实现
- 还是单独新增 `models/llm/xpu/*`

建议：

- 如果芯片行为接近 CUDA/MLU，先尽量复用通用模型
- 如果 attention、cache layout、rope 处理差异大，再单独做 `xpu` 目录

### 5.2 第一模型建议

推荐首个目标模型：

- `qwen3`

参考文件：

- `xllm/models/llm/qwen3.h`
- `xllm/models/llm/musa/qwen3.h`
- `xllm/models/llm/npu/qwen3.h`

原因：

- 路径清晰
- 仓库里已有多个后端实现可对照
- MUSA 版本接近“新后端最小样板”

### 5.3 模型注册

文件：

- `xllm/models/models.h`
- `xllm/models/model_registry.h`
- `xllm/models/model_registry.cpp`

待办：

- 在 `models.h` 中加 `#elif defined(USE_XPU)` 的 include 分支
- 在模型头文件中加：
  - `REGISTER_CAUSAL_MODEL(...)`
  - `REGISTER_MODEL_ARGS(...)`

必要时检查：

- `ModelRegistry::get_model_backend(...)`
- `resolve_model_registration(...)`

验收标准：

- 模型类型能被识别
- 不会报 “Model is not supported currently”

### 5.4 ModelContext 与 tensor options

文件：

- `xllm/core/framework/model_context.cpp`
- `xllm/core/framework/model/*.h`

待办：

- 检查 XPU 下 dtype、tensor options、设备迁移是否正确
- 检查是否有针对某些 backend 的特殊分支需要扩展

验收标准：

- 模型参数能正确 load 到 XPU
- 前向过程中的中间张量不会错误落到 CPU 或别的设备

---

## 6. Layer 与 attention 路径改造

### 6.1 先看模型真实调用链

建议你在动手前做一次调用链梳理：

- 选定 `qwen3`
- 找到模型调用的 decoder layer
- 找到 decoder layer 用到的 kernel 和 attention 实现

推荐搜索：

```bash
rg "kernel::" xllm/core/layers xllm/models
```

### 6.2 attention 相关重点

重点目录：

- `xllm/core/layers/`
- `xllm/core/kernels/`

需要核对：

- prefill 路径
- decode 路径
- paged kv cache 的布局
- rope 输入格式
- position ids 的 dtype
- block size 假设

这部分通常是新后端最容易出错的地方。

验收标准：

- 单 batch 短文本输出正常
- 多轮 decode 不崩
- 长 prompt 不出现明显 cache 错误

---

## 7. 运行时与分布式层改造

### 7.1 单卡阶段基本不需要大改

文件：

- `xllm/core/distributed_runtime/master.cpp`
- `xllm/core/distributed_runtime/master.h`

对新后端首版来说，通常不需要改 `create_master(...)` 的 backend 分类，因为这里分的是任务类型：

- `llm`
- `vlm`
- `dit`
- `rec`

不是设备类型。

### 7.2 多卡通信后做

等单卡稳定后，再看：

- `xllm/core/framework/parallel_state/collective_communicator.*`
- `xllm/core/distributed_runtime/worker_server.cpp`
- `xllm/core/distributed_runtime/dist_manager.cpp`

待办：

- 评估是否要接入 XPU 通信库
- 评估是否支持 allreduce / allgather / all2all
- 评估 rank table、拓扑发现、进程组初始化

首版建议：

- 不做

---

## 8. 测试与验证清单

### 8.1 编译验证

- `USE_XPU=ON` 能完整编译
- 其余后端不被破坏

### 8.2 设备验证

- `device_count()` 正确
- `--devices="xpu:0"` 正确
- 能在 XPU 上构造 tensor

### 8.3 kernel 单测

建议至少对以下算子补最小验证：

- matmul
- layernorm
- rotary
- paged cache write/read
- sampling

### 8.4 端到端验证

建议先跑：

- `examples/generate.py`
- 简单 prompt
- 短文本 decode

再跑：

- 长 prompt
- batch > 1
- 多轮会话

### 8.5 稳定性验证

至少验证：

- 连续多次加载/释放模型
- 显存回收是否正常
- stream 同步是否有随机错误
- decode 多轮是否 cache 污染

---

## 9. 推荐的提交拆分

为了减少回归风险，建议按下面顺序拆 commit 或 PR。

### PR 1：构建骨架

- 根 `CMakeLists.txt`
- `core/kernels/CMakeLists.txt`
- `core/platform/CMakeLists.txt`
- 空的 `xpu/` 目录和基础头文件

### PR 2：设备与 stream

- `device.cpp`
- `stream.cpp`
- 设备识别与最小 runtime 接入

### PR 3：kernel 最小链路

- `ops_api.cpp`
- `xpu_ops_api.h`
- 最小必要算子

### PR 4：qwen3 单模型接入

- `models.h`
- `models/llm/xpu/qwen3.h` 或通用模型修正

### PR 5：端到端验证与修正

- 修 bug
- 补最小测试
- 修参数和设备迁移问题

### PR 6：性能和高级特性

- attention 优化
- graph
- multi-card
- MoE

---

## 10. 首版必须避免的坑

- 不要第一版同时做新后端和多卡通信
- 不要第一版同时做 MoE/VLM/DiT
- 不要默认把 NPU 的特性全搬过来
- 不要在 `ops_api.h` 里声明一堆接口却没有根据实际调用链安排优先级
- 不要忽视 `Device::type_str()` 和 PyTorch 设备命名的一致性
- 不要忽视 paged KV cache 的 layout 差异

---

## 11. 一份可打勾的最小清单

- [ ] 根 `CMakeLists.txt` 加 `USE_XPU`
- [ ] `core/kernels/CMakeLists.txt` 接入 `xpu_kernels`
- [ ] `core/platform/CMakeLists.txt` 接入 XPU runtime
- [ ] 新增 `core/kernels/xpu/`
- [ ] `device.cpp` 增加 `USE_XPU`
- [ ] `stream.cpp` 增加 `USE_XPU`
- [ ] `DeviceNameUtils` 能解析 `xpu:0`
- [ ] `ops_api.cpp` 接入 `xpu::...`
- [ ] 实现 `matmul`
- [ ] 实现 `fused_layernorm`
- [ ] 实现 `apply_rotary`
- [ ] 实现 `reshape_paged_cache`
- [ ] 实现最小 attention 路径
- [ ] 实现 `apply_top_k_top_p`
- [ ] 实现 `random_sample`
- [ ] 接入 `qwen3`
- [ ] 在 `models.h` 注册 XPU 模型
- [ ] 单卡 `generate.py` 跑通
- [ ] 长 prompt 跑通
- [ ] 多轮 decode 跑通

如果这份清单都完成了，基本就算把 xLLM 的一个“新后端首版”做出来了。
