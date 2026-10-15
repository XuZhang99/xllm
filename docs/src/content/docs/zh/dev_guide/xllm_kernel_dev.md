---
title: "xllm-kernel 算子开发方案"
sidebar:
  order: 6
---

`xllm-kernel/` 负责 Python 模型使用的算子接口、实现注册与准备，以及独立的数值验证和性能测量。
本文给出新增算子、增加已有算子的实现、接入模型时的开发流程。安装和公共 API 见
[组件 README](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/README.md)。

## 1. 职责与目录

| 位置 | 负责内容 | 新增算子时的处理 |
| --- | --- | --- |
| `python/xllm_kernel/ops/<operation>/` | 公共算子语义、各 solution 的实现与注册、运行时绑定 | 定义接口或增加实现 |
| `python/xllm_kernel/registry.py` | `OpContract`、`KernelSpec`、`PreparedKernel` 与准备阶段选择 | 复用；不增加按算子名分支 |
| `python/xllm_kernel/fake/` | 本包拥有的 Torch operator 的 fake 实现 | 按实际需要增加 |
| `csrc/<solution>/` | 可选本地扩展、Torch bridge 和设备 launcher 适配 | Python/已有 Torch op 不够时再增加 |
| `python/xllm_kernel/numerics/<operation>.py` | 该算子的参数、输入、reference、容差、支持域和 `OperationSuite` | 每个算子提供自己的 suite |
| `python/xllm_kernel/numerics/catalog.py` | 显式发现内置 suite | 增加模块入口；不放算子逻辑 |
| `python/xllm_kernel/numerics/{arguments,suite,verify,report}.py` | 公共 CLI 参数、用例接口、验证生命周期、数值报告 | 通过接口复用 |
| `python/xllm_kernel/benchmark/` | 公共性能 CLI、NPU 计时与报告 | 复用已准备的用例，不解析算子 tensor 参数 |
| `test/`、`test/npu/` | 公共基础设施测试、独立设备测试与 fixture | 增加对应 pytest 用例 |

以上路径均相对于 `xllm-kernel/`。模型结构、batch metadata、KV cache 管理和请求调度仍属于
xLLM runtime。算子接收准备好的输入，不在 solution 内另建请求调度或 cache 生命周期。

公共文件不能写死具体算子的报告标题、输入角色、shape、容差、reference 或某个实现的支持域。
公共 API 的重导出、suite catalog 的模块名单和构建脚本中的扩展源文件名单属于显式接入点，
可以列出具体算子；计算语义和条件判断仍应留在对应算子模块。

## 2. 先定义契约，再实现 kernel

开发前明确以下内容，并让代码、测试和 README 使用同一套定义：

- 输入/输出角色、shape、dtype、设备、布局、stride、允许的空维度和标量参数范围。
- 是否修改输入、输出是否允许与输入 alias，是否有 residual、out-buffer 或量化语义。
- 数值 reference、累积精度、输出精度、容差规则和非有限值处理。
- 支持 eager、ACLGraph 中的哪些模式，以及 workspace、stream、设备上下文和生命周期要求。
- 依赖的 Torch/torch_npu、设备 SDK、第三方库与构建产物。

[`OpContract`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/registry.py) 当前表达 operation、version、mutates、
aliases_inputs 和 numerical_mode。`KernelSpec` 另外声明实现名、device、execution_modes、
input_domain、solution 和 priority。`input_domain` 当前是描述字符串，注册表不会据此自动验证
shape 或布局；真实实现仍须在 launch 前检查参数。

给同一个算子新增 solution 时，只有语义一致的实现才能使用相同的 `OpContract`。
改变 mutation、alias、量化或输出语义时，需要明确区分契约，不能靠实现名字掩盖差异。
实现间的数值等价可以按声明的容差判定，不应默认承诺逐位一致。

## 3. 实现、注册与模型接入

### 3.1 实现与注册

在 `ops/<operation>/<solution>.py` 中提供 `register(registry)`：先解析依赖并检查运行条件，
再把 `KernelSpec` 和 callable 注册进去。对照现有
[native adapter](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/ops/rms_norm/npu.py) 和
[xlite adapter](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/ops/rms_norm/xlite.py) 阅读。

复用宿主已有 Torch op 时，保留原有 schema、实现和 fake 的所有权。
新增本包拥有的 Torch op 时，使用本包 namespace，同时提供真实设备 bridge 和相应 fake。
构建产物及版本检查应在准备阶段完成，不能在请求执行或 graph replay 时下载依赖、编译或重选实现。

NPU 实现使用当前 PyTorch stream，并遵循 torch_npu 的提交队列和 Tensor 生命周期。
禁止在算子内部为方便实现而创建私有 stream 或执行设备同步；跨 stream、设备切换和异步资源释放
必须有对应测试。FakeTensor 负责可检查的逻辑 shape/dtype/device 语义，真实设备实现仍负责
物理格式等设备条件。

### 3.2 准备阶段与执行阶段

```text
加载并检查依赖
    -> register(KernelSpec, callable)
    -> registry.freeze()
    -> registry.prepare(contract, device, execution_mode, implementation)
    -> PreparedKernel
    -> 模型保留 callable
    -> eager 调用或 graph capture/replay
```

注册表冻结后不再增加实现。`prepare` 按契约、设备、执行模式和显式实现名过滤，再按优先级及
稳定名字顺序选择。执行路径直接使用已绑定 callable；准备失败或执行错误向上传播，不通过
自动 fallback 隐藏错误。改变实现策略需要重新创建 worker 和 graph。

当前包顶层 `initialize` 重导出 RMSNorm 的初始化入口，宿主通过
`XLLM_KERNEL_RMS_NORM_IMPL` 绑定 RMSNorm。它还不是多算子初始化管理器。新增算子时，需要
明确实现该算子的准备入口，并在宿主模型导入/capture 前接入；新增已有算子的 solution，
则需要同时检查该算子的初始化选择、离线 suite 和宿主配置入口。

模型接入只替换目标调用点。公共 API、输入布局和 cache view 先保持语义一致，再验证模型层的
实际路径；单算子精度通过不能替代模型接入测试。

## 4. 提供算子专属的离线 suite

公共接口定义在 [`numerics/suite.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/suite.py)。
数值 CLI 和性能 CLI 使用同一份 suite，避免各自维护不同的 shape、输入或 reference。

### 4.1 `OperationSuite`

每个 `numerics/<operation>.py` 导出一个 `SUITE`：

| 字段 | 约定 |
| --- | --- |
| `operation` | 与已准备实现的 `OpContract.operation` 一致 |
| `implementations` | 该 suite 支持的完整实现名；catalog 中不可重复 |
| `add_arguments(parser)` | 注册该算子的专属 CLI 参数 |
| `cases(args)` | 校验专属参数并返回 `KernelCase` 用例；此时不分配设备输入 |
| `prepare(device, mode, implementation)` | 加载所选实现并返回 `PreparedKernel` |

在 `numerics/catalog.py` 的 `_BUILTINS` 中增加模块入口。catalog 导入 suite 时只能使用
Python 元数据；Torch、设备 SDK 和 native extension 的导入留到真正准备/执行时，保证
`--help` 在无设备环境可用。

公共 CLI 处理 device、执行模式、显式实现名、warmup、revision、output；benchmark 另外处理
iterations 和 repeats。`--operation` 可显式指定，也可由 `--implementation` 推导。
shape、dtype 组合等算子参数由 `add_arguments` 和 `cases` 负责，不能塞进公共 CLI 的条件分支。

### 4.2 `KernelCase` 与 `PreparedCase`

每个逻辑用例实现以下接口：

| 接口 | 返回内容与职责 |
| --- | --- |
| `case_id` | 可复现、能区分用例参数和 seed 的标识 |
| `parameters` | 可写入 JSON 的参数字典 |
| `unsupported_reason(plan)` | 不适用时返回原因，适用时返回 `None`；不得通过执行失败来猜支持域 |
| `prepare(plan)` | 生成输入，返回 `PreparedCase` |

`PreparedCase` 的三个字段是：

- `function()`：绑定输入、标量参数和已准备实现的零参数 callable，返回值可以是 tensor、tuple
  或其他由该算子定义的结构。
- `validate(function)`：调用传入的零参数 callable，按该算子的 reference 和容差返回包含
  `status` 的验证结果；校验输出结构、数值、mutation 和 alias 等契约。
- `metadata`：输入 stride、格式等可序列化的专属记录。

验证器必须检查传入 callable 的结果，不能无视它而重新调用原始实现。ACLGraph 验证传入的是
捕获输出的读取函数；重新执行 eager 算子会掩盖错误的 graph 输出。验证结果和 metadata
也不应覆盖公共 case ID、operation、implementation 等身份字段。

默认输入用固定 seed 在 CPU 生成，再按实现需要搬到设备，保证候选实现接收相同逻辑数据。
reference 应独立于被测实现。支持域过滤必须和真实实现的参数检查一致；边界仍要由直接设备
测试覆盖，不能通过全部标为 `not_applicable` 规避验证。

现有 [`numerics/rms_norm.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/rms_norm.py) 提供完整实例。
公共 CLI 测试另外使用参数为 size/scale、返回字典的测试算子，验证 runner 不依赖 RMSNorm 的
input/weight/eps 结构。这个测试算子只是公共流程的单测，不代表新增设备算子支持。

## 5. 公共验证、计时和报告流程

```text
解析公共参数 + suite 专属参数
    -> suite.prepare
    -> 用例支持域检查
    -> case.prepare
    -> validate(function)
    -> warmup
    -> 可选 graph capture/replay + validate(captured_output)
    -> benchmark 计时
    -> JSON / Markdown 报告
```

`numerics/verify.py` 只管理准备、状态和报告生命周期；不解释 tensor 角色，也不选择某个算子的
reference。`numerics/report.py` 和 `benchmark/report.py` 使用通用标题，逐行记录 operation，
不通过名字猜测算子类型或读取某个算子的 shape 字段。

失败状态需保持可见：不适用记录原因；数值失败不计时；capture/replay 失败直接报告并抛出；
没有任何通过的用例不能产生成功报告。对设备执行错误不得切换到 eager 或其他 solution 后
继续给出有效性能结果。

当前计时器支持 NPU eager 和 ACLGraph，graph benchmark 每次 capture 包含 64 次独立调用。
这要求用例能安全地反复调用且性能期间输入状态可复用。对会修改输入、消费状态或依赖随机数的
新算子，应先设计显式 reset 和计时边界，再扩展公共用例接口；不要直接套用当前循环。

准备、warmup 和 capture 不进入稳态计时。保持输入、case、设备、执行模式和计时设置一致，
报告原始样本与离散程度，并记录资源占用。eager 的设备事件间隔包含主机提交间隙，不能简单
等同于单个 kernel 的纯设备执行时间。性能结论还应区分单算子和端到端结果。

## 6. 测试与验证

包测试通过 pytest 独立运行，不接入宿主 `tests/CMakeLists.txt`、CTest 或 `all_tests`。
以下命令从仓库根目录执行，先按组件 README 安装包及依赖。

```bash
# 公共基础设施、数值报告和 CLI 的 CPU 测试
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu --ignore=xllm-kernel/test/distributed

# 设备集成测试：先构建匹配的 host/bridge，在空闲设备上顺序运行
XLLM_KERNEL_RMS_NORM_IMPL=npu.xllm_native.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_rms_norm_npu.py
XLLM_KERNEL_RMS_NORM_IMPL=npu.xlite.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_xlite_rms_norm_npu.py
```

设备 fixture 在 `test/npu/conftest.py` 中加载宿主并初始化 Python kernels；pytest 不负责构建。
同一进程的实现绑定不可切换，因此 native/xlite 使用不同进程。直接 pytest 不获得 CTest 的
设备资源锁，需要自行确保设备空闲；设备切换测试需要两张可见 NPU。

离线工具示例使用当前已支持的算子。新增算子接入后，替换 operation、implementation 和专属参数：

```bash
python -m xllm_kernel.numerics --operation rms_norm --device npu --mode eager \
  --implementation npu.xllm_native.rms_norm --rows 1 16 --widths 6144 \
  --revision "$(git rev-parse HEAD)" --output numerics.json
python -m xllm_kernel.benchmark --operation rms_norm --device npu --mode aclgraph \
  --implementation npu.xllm_native.rms_norm --rows 1 16 --widths 6144 \
  --iterations 50 --revision "$(git rev-parse HEAD)" --output benchmark.json
```

新增算子的测试按以下顺序展开，范围以实际契约为准：

1. SDK-free import、注册/准备、独立安装和 CLI 参数测试。
2. CPU reference、容差、错误输出、非有限值、mutation/alias 检测。
3. 真设备 dtype/shape/layout/stride/空输入/边界与非法参数测试。
4. 非默认 stream、设备 guard、异步对象生命周期、重复调用和内存稳定性。
5. eager/ACLGraph 输出一致性，以及改变输入后的交错 replay。
6. 独占设备上的单算子性能比较，最后按模型接入范围做集成验证。

CPU double 和 FakeTensor 只证明对应逻辑，不能作为 NPU 精度或性能证据。

## 7. 交付检查项

- 契约、真实参数检查、fake 检查、suite 支持域和 README 一致。
- 公共 runner、report、registry 中没有为新算子增加名字、shape、输入角色或容差分支。
- suite 导入不加载设备依赖，未知实现和不匹配的 operation 在准备执行前失败。
- 新 solution 的宿主选择入口、离线 suite 和设备测试同时接入。
- 单测仍位于独立包 `test/`，不修改宿主的测试目标来自动运行它们。
- 检查格式、安装包内容、CLI 示例和报告；提交说明区分源码检查、CPU、构建、NPU 和模型证据。
- 如需 JIT、状态 reset、多设备或新平台服务，先补齐显式接口及生命周期，再增加实现。

## 8. GLM-5.3 算子清单与显式后端接入

本轮扩展对照 `xllm/python/models/glm5_2.py`、共享的 `deepseek_v32.py` 路径、
`layers/embedding.py`、`layers/linear.py`，以及
`kernels_npu/{normalization,activation,quantization,rotary_embedding,moe,mla,sparse_attention}.py`。
这里指完整的 GLM-5.3 DSA 模型，不是 GLM-5.3 Flash。开发基线为
`feat/glm53-pytorch-kernel-package` 的 `26c4d550`。

| 算子 | NPU 实现 |
| --- | --- |
| `add` | `torch`, `xlite` |
| `all_gather` | `xllm_native`, `cann` |
| `all_reduce` | `xllm_native`, `cann` |
| `all_to_all_single` | `cann` |
| `atb_matmul_ein_sum` | `xllm_native`, `torch`, `xlite` |
| `atb_matmul_ein_sum_out` | `xllm_native` |
| `dequant_swiglu_quant` | `xllm_native`, `cann` |
| `dynamic_quant` | `xllm_native`, `cann`, `xlite` |
| `embedding` | `torch`, `xlite` |
| `fused_add_rms_norm` | `xllm_native`, `cann`, `xlite` |
| `fused_add_rms_norm_dynamic_quant` | `cann` |
| `grouped_matmul` | `cann` |
| `grouped_matmul_out` | `xllm_native` |
| `inplace_partial_rotary_mul` | `xllm_native`, `cann` |
| `interleaved_rotary_embedding` | `cann`, `xlite` |
| `layer_norm` | `torch`, `xlite` |
| `lightning_indexer` | `xlite`, `xllm_native`, `torch` |
| `lightning_indexer_out` | `xlite`, `xllm_native` |
| `linear` | `torch`, `xlite` |
| `moe_finalize_routing` | `cann` |
| `moe_gate_routing` | `xlite`, `cann` |
| `moe_gmm1` | `cann`, `xllm_native` |
| `moe_token_dispatch` | `cann`, `xllm_native` |
| `moe_token_unpermute` | `cann` |
| `quant_matmul` | `xllm_native`, `cann` |
| `quant_matmul_out` | `xllm_native` |
| `quantize_per_tensor` | `xllm_native` |
| `reduce_scatter` | `xllm_native`, `cann` |
| `reshape_paged_cache` | `xllm_native`, `torch` |
| `rms_norm` | `xllm_native`, `cann`, `xlite` |
| `rms_norm_dynamic_quant` | `xllm_native`, `cann` |
| `router_matmul` | `torch` |
| `scatter_nd_update` | `xllm_native`, `torch` |
| `sfa_dcp_remap` | `tilelang`, `torch` |
| `silu_and_mul` | `xllm_native`, `cann`, `xlite` |
| `sparse_flash_attention` | `xllm_native`, `torch` |
| `sparse_flash_attention_out` | `xlite`, `xllm_native` |

`torch` 表示显式的 PyTorch NPU 组合实现，`cann` 调用 TorchNPU 算子，
`xllm_native` 复用宿主已有 schema，`tilelang` 解析已有 AOT 算子。
xlite 使用固定 GVirt 版本的 launcher，并提交到当前 stream。准备阶段仅加载指定后端。

新增算子模块声明 `OPERATOR` 并导出 `prepare`：

```python
from xllm_kernel.ops.dynamic_quant import prepare
plan = prepare("npu", "aclgraph", "npu.cann.dynamic_quant")
quantized, scale = plan.function(hidden)
```

loader 在 capture 前返回 callable；`Operator` 负责注册、冻结和准备。
已有集成仍可使用显式的 `register(registry)`。增加 adapter 不会自动改变模型后端策略；
已有模型调用以及 MLA、MoE 等组合调度继续由宿主管理。

新增精度 suite 通过 `tensor_suite` 组织算子自己的形状矩阵。`Example` 绑定输入 tensor tree、
CPU 参考、声明的输入修改、容差及可选的语义归一化比较方法。`TensorCase` 只携带元信息，
发现 suite 时不加载设备 SDK。`PreparedCase.reset` 在计时 event 区间外恢复可变输入；
有输入修改的算子每张图只捕获一次调用。各算子的测试文件复用
`test/npu/operator_checks.py` 的执行流程，不接入 `tests/CMakeLists.txt`。

源码核对与覆盖边界：

- 模型默认 hidden size 6144、词表 154880、256 个专家、top-8 路由。
  Embedding 按 hidden 维进行 TP 切分，TP=16 时单卡宽度为 384。
  用例包含 decode/prefill 行数；本地 head 数、expert 数通过 CLI 显式指定，
  不把一组单算子形状当作完整部署配置。
- xlite 的 RMSNorm 后接 BF16 量化、经 FP16 中间结果的量化 GEMM、bitmap dispatch，
  不能直接替代融合 FP32/NZ 契约。不通过相同后端名称隐藏额外舍入或布局差异。
- xlite indexer adapter 负责 head weight 打包及 scratch 分配、清零。稀疏 attention 的 `_out`
  接口支持单 query 的 BF16 decode，不返回 LSE；填充索引压缩、KV gather 和 workspace 开销均计入计时。
  融合 MLA preparation 继续由宿主管理。已有 TileLang 半分布局 RoPE 不能冒充交错 RoPE。
- Native 稀疏 attention 保留独立的 RoPE ACLNN 入口，不替换成同名旧 CANN ABI。
  独立 CPU 参考核对输出和 log-sum-exp，包含因果 mask、打乱的物理页和 -1 稀疏索引填充。
- Lightning Indexer 核对索引唯一性、因果边界和所选 CPU 分数，分数相对容差为 1.5%，
  允许数值并列，不声称索引逐元素完全相同。
- MegaMoE、可选 FP8 KV/INT8 index cache、固定形状 MLAPO v2、MTP 调度元数据需要独立状态及布局契约。
  普通 ATen 张量操作保留在宿主。本轮算子清单覆盖 BF16/W8A8 路径。

通信算子通过 `torchrun` 和通用 `--distributed` CLI 流程测试：显式创建默认 HCCL group，
各 rank 独立生成报告，并记录每轮各 rank 的最大耗时。CANN ProcessGroup 后端仅声明 eager；
native 当前 stream 的 all-reduce/gather/reduce-scatter 提供 ACLGraph 入口，运行前设置
`HCCL_OP_EXPANSION_MODE=AIV`。All-to-all 仅支持等长分片和 eager。
具体命令及输入域见包 README。

**交付状态：** 仅完成源码和格式检查。本轮未编译、未执行单测、精度、benchmark、设备调用或端到端测试。
新增用例是后续验证规范，不是通过结果。

路由及 SwiGLU 语义同时参考官方
[MoeInitRoutingV2 API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_moe_init_routing_v2.md) 和
[DequantSwigluQuant API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_dequant_swiglu_quant.md)。
安装的 runtime 必须提供所选接口；缺失时准备或执行明确报错。文档不能替代二进制 ABI 验证。
