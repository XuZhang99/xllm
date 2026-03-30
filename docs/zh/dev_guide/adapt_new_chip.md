# xLLM 适配新国产芯片教程

本文基于对 `xllm` 仓库的一次粗读整理，目标不是解释所有实现细节，而是给出一条可执行的适配路线，帮助你把 xLLM 跑到一款新的国产芯片上。

适用对象：

- 你已经拿到了该芯片的编译工具链、运行时库和 PyTorch/Libtorch 后端
- 你希望先让 xLLM 单卡可跑，再逐步补齐性能和分布式能力

不适用对象：

- 你还没有 PyTorch 后端，或者芯片侧没有基本的 Tensor/Stream/Allocator 能力
- 你希望一步到位复刻 NPU 分支的全部特性

## 1. 先说结论：xLLM 的适配点在哪里

从仓库结构看，xLLM 的“芯片适配”主要分成 5 层：

1. 构建层：决定是否启用某个设备后端
2. 设备层：决定 xLLM 识别什么设备、如何切卡、如何同步、如何取显存信息
3. 算子层：把统一的 `kernel::xxx(...)` 分发到你的芯片实现
4. 模型层：为目标设备选择一套模型实现
5. 通信/图模式/高级特性层：补齐多卡、图执行、KV 管理、优化特性

可以把它理解成：

`xllm.cpp` -> `Options` -> `Device/Stream` -> `kernel::ops_api` -> `models/*/<backend>` -> runtime/scheduler/distributed

对新芯片来说，最重要的不是一开始追求“全量特性”，而是先打通下面这条最小链路：

`编译通过 -> 识别设备 -> 单卡加载模型 -> 能执行一轮 prefill/decode -> 能返回 token`

## 2. 我对代码的粗读结论

### 2.1 启动入口

入口在 `xllm/xllm.cpp`。

这里负责做几件事：

- 解析模型路径和 `--backend`
- 调 `DeviceNameUtils::parse_devices(...)` 解析 `--devices`
- 把各种运行参数组装成 `Options`
- 调 `create_master(...)` 拉起推理主流程

这意味着：**只要设备名解析、平台能力和模型构造没打通，服务就起不来。**

### 2.2 设备抽象

设备抽象在：

- `xllm/core/util/device_name_utils.cpp`
- `xllm/core/platform/device.h`
- `xllm/core/platform/device.cpp`
- `xllm/core/platform/stream.h`
- `xllm/core/platform/stream.cpp`

这里统一封装了：

- `device_count()`
- `type_str()`
- `type_torch()`
- `set_device()`
- `init_device_context()`
- `empty_cache()`
- `synchronize_default_stream()`
- `current_stream()`

这层决定了 xLLM 编译产物究竟把你的芯片当成 `npu`、`mlu`、`musa` 还是别的名字来识别。

### 2.3 算子统一入口

算子总入口在：

- `xllm/core/kernels/ops_api.h`
- `xllm/core/kernels/ops_api.cpp`

这个文件非常关键。上层模型和 layer 基本都调用统一的 `kernel::matmul(...)`、`kernel::apply_rotary(...)`、`kernel::fused_layernorm(...)`、`kernel::group_gemm(...)` 等接口，然后在 `ops_api.cpp` 里按编译宏分发到具体后端：

- `npu::*`
- `mlu::*`
- `cuda::*`
- `ilu::*`
- `musa::*`

所以新芯片适配的核心工作，通常就是：

1. 新增一套 `core/kernels/<your_backend>/`
2. 在 `ops_api.cpp` 里接入分发
3. 逐步补齐模型实际会调用到的算子

### 2.4 模型组织方式

模型注册和工厂在：

- `xllm/models/model_registry.h`
- `xllm/models/models.h`

`models.h` 会按编译宏选择不同模型集合。例如：

- `USE_NPU` 下包含大量 `llm/npu/*`
- `USE_MUSA` 下目前只包含 `llm/musa/qwen3.h`

这说明 xLLM 的思路不是所有设备共享完全同一份模型实现，而是允许按后端维护不同版本。

如果你的芯片能力和 CUDA/MLU/MUSA 接近，最好先复用通用模型；
如果你的芯片需要不同 attention/cache/layout/算子序，最好单独建 `models/llm/<your_backend>/`。

### 2.5 现有后端成熟度并不一致

从代码量看：

- `npu` 分支最完整，包含模型、算子、平台、分布式相关能力
- `cuda`/`mlu`/`ilu` 也有比较完整的 kernel 适配


## 3. 建议的适配策略

推荐分四阶段推进：

### 阶段 A：先做最小后端骨架

目标：

- 能编译
- 能识别 `--devices="<your_device>:0"`
- 能单卡完成最小推理

### 阶段 B：补齐 LLM 主路径

目标：

- 至少跑通一个主流模型，例如 `qwen3`
- 支持 prefill + decode
- 支持 KV cache 基本读写

### 阶段 C：补性能关键算子

目标：

- attention
- rotary
- norm
- matmul/group gemm
- sampling

### 阶段 D：补高级特性

目标：

- 多卡通信
- 图模式
- prefix cache / global kvcache
- MTP / speculative / VLM / MoE 优化

## 4. 落地步骤

## 4.1 新增编译开关

先在根目录 `CMakeLists.txt` 增加一个开关，例如：

```cmake
option(USE_XPU "Enable XPU support" OFF)
```

然后仿照现有分支，把你的工具链、运行时和头文件接进来。

参考位置：

- 根目录 `CMakeLists.txt`
- `xllm/core/kernels/CMakeLists.txt`
- `xllm/core/platform/CMakeLists.txt`

建议做法：

- 如果芯片有独立编译语言，按 `USE_MUSA` 的方式接入
- 如果只是普通 C++ 扩展，按 `USE_NPU` / `USE_MLU` 方式接入

这一步完成后，至少要保证：

- `cmake` 能识别开关
- 对应后端目录能被 `add_subdirectory(...)`
- 最终 `platform` 和 `kernels` 能链接到你的 runtime 库

## 4.2 接入设备抽象

重点文件：

- `xllm/core/platform/device.cpp`
- `xllm/core/util/device_name_utils.cpp`

你至少要实现下面这些分支：

- `Device::device_count()`
- `Device::type_str()`
- `Device::type_torch()`
- `Device::set_device()`
- `Device::init_device_context()`
- `Device::empty_cache()`
- `Device::synchronize_default_stream()`
- `Device::current_stream()`
- `Device::get_device_mem()`

这里有两个关键判断。

### 判断 1：你的 PyTorch 后端对应什么 `torch::DeviceType`

现有代码里：

- NPU/MLU 走 `torch::kPrivateUse1`
- CUDA/ILU 走 `torch::kCUDA`
- MUSA 走 `torch::kMUSA`

如果你的芯片 PyTorch 后端也是 `PrivateUse1`，那你要特别注意：

- 设备字符串要和你注册给 PyTorch 的名字一致
- Tensor `.to(device)`、`torch::Device(...)` 构造是否兼容

### 判断 2：xLLM 对设备名称的要求是“编译期唯一后端”

`DeviceNameUtils::parse_devices(...)` 里会检查：

- `parts[0] == Device::type_str()`

也就是说，一次构建产物默认只认一种设备类型。  
如果你新增的是 `xpu`，那就要保证：

- `--devices="xpu:0"`
- `Device::type_str()` 返回 `"xpu"`

## 4.3 接入 Stream 和基础平台能力

除了 `device.cpp`，还要检查：

- `xllm/core/platform/stream.cpp`
- `xllm/core/platform/vmm_api.*`
- `xllm/core/platform/shared_vmm_allocator.*`

最低要求：

- 至少能拿到当前 stream
- 至少能做默认 stream 同步

如果你的芯片暂时不支持 VMM 或统一虚拟内存，建议策略是：

- 第一阶段先保证能编译并关闭相关高级特性
- 不要一开始强行补齐 `shared_vmm_allocator`

因为 xLLM 的“先跑起来”并不强依赖所有高级内存能力。

## 4.4 新增 kernel 后端目录

建议新建：

```text
xllm/core/kernels/xpu/
```

最少包含：

- `CMakeLists.txt`
- `xpu_ops_api.h`
- 若干算子实现文件

然后在：

- `xllm/core/kernels/CMakeLists.txt`
- `xllm/core/kernels/ops_api.cpp`

中接入你的后端。

推荐模式：

```cpp
#elif defined(USE_XPU)
#include "xpu/xpu_ops_api.h"
```

以及：

```cpp
#elif defined(USE_XPU)
  xpu::apply_rotary(...);
```

## 4.5 第一批必须补的算子

不要试图一口气实现 `ops_api.h` 的所有接口。  
先以一个具体模型为目标，按“被真实调用到”的顺序补。

如果你先适配 `qwen3`，通常优先级如下：

1. `apply_rotary`
2. `fused_layernorm`
3. `matmul`
4. `reshape_paged_cache`
5. attention 相关实现
6. `apply_top_k_top_p`
7. `random_sample`

如果要支持 MoE，再追加：

1. `group_gemm`
2. `moe_active_topk`
3. `moe_gen_idx`
4. `moe_expand_input`
5. `moe_combine_result`
6. all2all 相关接口

一个很实用的方法是：

- 先选中一个目标模型
- 用 `rg "kernel::"` 搜对应 layer / model
- 记录真实调用链
- 只实现这条链路必需的算子

## 4.6 先选一个最小模型做样板

建议先选：

- `qwen3`

建议策略：

- 如果你的芯片张量语义和现有通用实现接近，优先复用 `models/llm/qwen3.h`
- 如果 cache layout、rope、attention metadata 处理和现有后端差异大，就新建 `models/llm/xpu/qwen3.h`

## 4.7 把模型注册进来

新增模型实现后，要确保它能被注册并选中。

关键位置：

- `xllm/models/models.h`
- `xllm/models/model_registry.h`

你至少要做两件事：

1. 在 `models.h` 的 `#elif defined(USE_XPU)` 分支里 include 你的模型头文件
2. 在模型头文件里使用现有注册宏，例如：

```cpp
REGISTER_CAUSAL_MODEL(qwen3, QWen3ForCausalLM);
REGISTER_MODEL_ARGS(qwen3, [&] {
  ...
});
```

如果你不做这一步，就算编译通过，运行时也会出现“模型类型不支持”。

## 4.8 决定是“新后端专属模型”还是“复用通用模型”

这是适配里最关键的架构选择。

### 方案 A：尽量复用通用模型

适用于：

- 你的芯片 PyTorch 后端语义和 CUDA/MLU 接近
- 大部分 layer 代码不需要特殊分支
- 差异主要在算子实现

优点：

- 代码少
- 后续跟进上游更容易

缺点：

- 某些性能优化难以下沉

### 方案 B：维护 `models/llm/xpu/*`

适用于：

- attention / cache / rope / graph capture 差异很大
- 需要做后端专属张量布局
- 某些层必须调用芯片厂商 fused op

优点：

- 性能空间大
- 适配逻辑更直接

缺点：

- 后续维护成本更高

如果你没有明确证据表明必须 fork 模型实现，我建议先走方案 A。

## 4.9 先把高级特性关掉

新芯片首版适配时，建议先关闭或暂缓以下能力：

- graph mode
- prefix cache
- global kvcache
- disagg PD
- eplb
- multi stream parallel
- speculative / mtp
- 多机通信

原因很简单：这些能力会把问题从“算子和运行时是否正确”放大成“调度、通信、内存、图捕获是否协同正确”。

首版目标应该是单机单卡稳定推理。

## 4.10 通信后端单独做，不要和单卡适配绑死

`xllm/xllm.cpp` 里有：

- `communication_backend`
- `rank_tablefile`
- `nnodes`
- `dp_size`
- `ep_size`

这说明 xLLM 的多卡能力是独立层。

建议顺序：

1. 单卡打通
2. 单机多卡打通
3. 再补多机

如果你的芯片厂商提供了类似 HCCL/NCCL/LCCL 的通信库，再去补：

- collective service
- allreduce / all2all 相关算子
- rank table / 拓扑配置

不要在第一版里同时调试“单卡算子错误”和“多卡通信错误”。

## 5. 一个建议的最小实施方案

如果让我从零开始做一个 `XPU` 后端，我会按下面顺序：

1. 在根 `CMakeLists.txt` 加 `USE_XPU`
2. 在 `xllm/core/platform/device.cpp` 增加 `USE_XPU` 分支
3. 在 `xllm/core/kernels/xpu/` 下新增最小 kernel 目录
4. 在 `xllm/core/kernels/ops_api.cpp` 接入 `USE_XPU`
5. 先让 `matmul`、`fused_layernorm`、`apply_rotary`、`reshape_paged_cache`、采样相关接口可用
6. 新建或复用 `qwen3` 模型实现
7. 在 `xllm/models/models.h` 接入 `USE_XPU`
8. 用单卡跑最小生成样例
9. 补 attention 和性能关键路径
10. 最后再做多卡、图模式和高级优化

## 6. 验证顺序

推荐按下面顺序做验证。

### 6.1 编译验证

先验证：

- `cmake` 能通过
- `platform` 和 `kernels` 能链接
- `xllm` 主二进制能产出

### 6.2 设备验证

验证：

- `--devices="xpu:0"` 能被识别
- `Device::device_count()` 返回正常
- `set_device()` 后能成功创建 tensor

### 6.3 最小张量验证

建议单独写几个小测试：

- `torch::randn(...).to("xpu:0")`
- matmul
- layernorm
- rotary
- cache 写入/读取

### 6.4 单模型验证

先用最简单的离线生成样例验证：

- `examples/generate.py`
- 或直接启动 `xllm/xllm.cpp`

目标不是性能，而是：

- 不崩
- 有 token 输出
- 输出不明显异常

### 6.5 回归验证

在单卡稳定后，再补：

- 长 prompt
- batch 场景
- 多轮 decode
- 多卡
- MoE
- VLM

## 7. 常见坑

### 7.1 设备类型和 PyTorch 后端类型不一致

如果：

- `Device::type_str()` 返回的是 `xpu`
- 但 `torch::DeviceType` 或 PyTorch 注册名不是这个

通常会出现 `.to(device)` 或 `torch::Device("xpu:0")` 失败。

### 7.2 `PrivateUse1` 后端的设备名没对齐

NPU/MLU 这类后端经常走 `PrivateUse1`。  
如果你也走这条路，必须确认：

- Python 侧设备名
- C++ 侧 `torch::Device(...)`
- xLLM 的 `Device::type_str()`

三者一致。

### 7.3 不是所有模型都适合作为第一阶段目标

不要一上来就适配：

- DeepSeek MoE
- VLM
- MTP
- DiT

优先选普通 decoder-only 模型，例如 `qwen3`。

### 7.4 不要默认照搬 NPU 的全部路径

NPU 分支很完整，但它依赖很多专有能力：

- `torch_npu`
- Ascend runtime
- ATB / 自定义算子
- 通信和图模式能力

如果你的芯片没有这些能力，机械照搬会把适配复杂度放大很多。

### 7.5 首版不要过早追图模式

图模式在 xLLM 里是高级优化能力，不是启动前提。  
新后端首版建议先保证 eager path 正确，再考虑 graph capture。

## 8. 我建议优先参考哪些文件

如果你要开始动手，优先读这些文件：

### 构建和入口

- `CMakeLists.txt`
- `xllm/xllm.cpp`

### 设备与平台

- `xllm/core/platform/device.h`
- `xllm/core/platform/device.cpp`
- `xllm/core/platform/stream.h`
- `xllm/core/platform/stream.cpp`
- `xllm/core/util/device_name_utils.cpp`

### 算子分发

- `xllm/core/kernels/ops_api.h`
- `xllm/core/kernels/ops_api.cpp`
- `xllm/core/kernels/CMakeLists.txt`

### 模型注册

- `xllm/models/model_registry.h`
- `xllm/models/models.h`

### 参考后端

- 模型参考：`xllm/models/llm/qwen3.h`
- 算子参考：`xllm/core/kernels/cuda/*`
- 平台参考：`xllm/core/platform/device.cpp`

## 9. 一个现实的结论

如果你的目标是“把 xLLM 适配到一款新国产芯片”，真正的工作量通常不在“改几个宏”，而在下面两个问题：

1. 你的芯片 PyTorch 后端是否足够完整
2. 你的芯片是否已经具备 attention/norm/matmul/cache/通信 的基础高性能实现

如果这两点都具备，那么 xLLM 的代码结构是适合接入新后端的。  
如果这两点不具备，那么 xLLM 适配会变成“顺便补一套深度学习运行时能力”，工作量会很大。

## 10. 推荐的项目推进顺序

最后给一个更实用的推进顺序：

1. 先决定设备命名、PyTorch device type、编译宏
2. 打通 `platform/device.cpp`
3. 补 kernel 分发层
4. 选 `qwen3` 做第一模型
5. 跑通单卡生成
6. 做 correctness 校验
7. 补 attention / KV / sampling 性能
8. 再做多卡通信
9. 最后补图模式、MoE、VLM、投机推理

如果你后续真的要动手改代码，建议第一版目标只定成：

**“单机单卡 Qwen3 可稳定生成”**

这是最合理的里程碑。
