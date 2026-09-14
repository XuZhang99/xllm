---
title: GLM-5.3 HiSparse 实现状态与后续计划
---

本文记录 `feat/glm53-pytorch-hisparse` 分支截至 2026-09-14 的实现状态、未完成项和后续思路。分支基于 main 的 `f00248070ad41a2a97598a9f305d7b1e9acc7d84`，面向 NPU 上基于 PyTorch 组图的 GLM-5.3，复用 GLM-5.2 的 `glm_moe_dsa` 模型实现。

参考 [SGLang HiSparse](https://lmsysorg.mintlify.app/docs/advanced_features/hisparse_guide) 的分层缓存思路。当前版本已打通 Host KV、HBM 热缓存和 ACLGraph 推理，但尚未实现完整的 LRU 和跨层预取，也没有证明端到端性能收益。参数和使用方法见[英文功能说明](../../../en/features/hisparse/)。

## 已完成

- 完整 BF16 MLA KV 存储在注册并映射到 NPU 的 Host 内存；DSA index cache 全量保留在 HBM，支持 INT8 index cache。
- 每层维护全 worker 共享的有界 HBM 热缓存、物理槽位反向映射和有效性标签。新 KV 写入会使旧热缓存映射失效。
- TileLang store/gather 内核支持 Host 写入、HBM 命中读取、Host 缺失读取以及无效槽位跳过。
- Decode 将逻辑 Top-K 转为物理槽位，收集到 HBM 紧凑 KV 工作区后执行 sparse attention；图内没有 CPU Top-K 回读或 Python Host 回调。
- 支持普通及 chunked prefill、eager/ACLGraph decode，单独核算 Host 全量 KV 和 HBM index/map/hot/scratch 容量。
- 适配实际 GLM-5.3 checkpoint 的 W8A8_DYNAMIC attention projection 权重，并修复 QuantLightningIndexer metadata 在图捕获与回放之间的生产和生命周期问题。

当前热缓存策略是按当前选择顺序刷新最多 `hisparse_device_buffer_size` 个 token，容量按每层、每 worker 计算，所有请求共享；它不是 LRU。层间及不同 graph bucket 共享一个 selected-KV 工作区，因此当前依赖顺序执行。

## 优先后续工作

### 1. LRU 或近似 LRU 替换

**未完成：** 当前刷新策略不依据访问历史选 victim，多请求场景可能偏向排在前面的请求，重复选择也可能降低有效容量。

**思路：** 在 device 上维护访问状态，先完成物理槽位转换、有效性检查和有效 miss 去重，再保留 hit、选择 victim、填充 miss，按顺序更新 KV、tag 和反向映射。比较精确 LRU 与 clock/近似 LRU 的维护开销，先用 profile 决定策略。索引和值的更新必须避免并行读写竞争，不引入逐步 CPU 同步，保持固定形状及图地址稳定。

**验收：** 覆盖选择量超过热容量、重复 Top-K、请求重排、padding、槽位回收及 graph bucket 切换；与全 Host 参考结果一致，并记录命中率、替换开销和公平性。

### 2. IndexShare 跨层预取与传输重叠

**未完成：** 保留了模型的 IndexShare 选择语义，但未利用共享索引提前读取后续层 KV。

**思路：** 共享的是 token 选择，KV 内容仍属于各自层。确认索引产生层与复用层边界后，在独立 stream 上提前 gather 后续层的 Host KV；通过双缓冲或按在途层分配工作区，避免覆盖当前层 attention 的输入。使用 event 保证读取前完成、复用前消费完毕，并满足 ACLGraph 捕获时的 stream join 和稳定地址要求。需要先改变当前单工作区顺序执行的假设，重新计入 HBM 预算。

**验收：** 覆盖 IndexShare 边界、变长 batch、不同图桶及循环回放；检查无竞争、结果一致，并测量实际传输重叠比例和新增 HBM 成本。

### 3. 完整精度与性能评测

**未完成：** 未运行完整精度数据集，也未做隔离设备上的 HiSparse 开关 A/B 吞吐基准。

**思路：** 在相同代码、模型、量化方式、TP、输入和采样参数下分别关闭/开启 HiSparse。覆盖 2K/8K/32K/64K 上下文、不同并发、冷热缓存；首先确认模型和测试配置允许相应长度。使用固定种子和温度 0 的基础对照，再运行完整 GSM8K 等精度集及长上下文检索任务。短样本正确不能替代完整精度评测。

**验收：** 同时报告可容纳上下文/请求数、Host/HBM 峰值、TTFT、TPOT、吞吐、命中率、Host 读取量和 gather/prefill 耗时，明确图桶覆盖及 eager fallback。设备需无其他任务干扰；在结果出来前不宣称加速。

### 4. Prefill 与运行观测

**未完成：** Prefill attention 直接访问 Host KV；尚未提供面向调优的完整缓存统计。

**思路：** 先定位 Host 带宽及 prefill 瓶颈，再评估 chunk 级 HBM staging、复用和写回重叠，保持因果 mask 与位置语义。添加 device 计数器，异步采样命中、缺失、填充字节、替换和注册容量，避免图内逐步同步。结合统计评估热容量及多请求分配策略。

**验收：** 在有界 scratch 内保持输出一致，量化 TTFT 与容量的取舍；确认统计开关的额外开销。

## 后续兼容性扩展（当前不支持）

这些能力不是本次已交付范围；目前通过配置检查拒绝不支持的组合。应逐项设计、测试后再解除限制。

| 能力 | 大致思路及依赖 |
| --- | --- |
| CP、KV/layer split | 明确分片后的物理槽位所有权、局部/远端 KV 获取方式和映射失效规则，适配通信与图捕获，并重新计算每 rank 预算。 |
| Prefix sharing | 将热映射生命周期与共享物理页引用计数绑定，避免单请求释放或更新破坏其他请求。 |
| MTP | 支持每请求多个 query、候选 token 写入及回滚，重新定义选中行数上限、去重和工作区大小。 |
| PD 分离 | 设计 Host 接收缓存注册、完整 KV 传输及请求交接协议；不能直接复用 SGLang CUDA 的 direct-to-host 协议。 |
| Schedule overlap、XTensor、sleep、spawned offline workers | 分别验证并发访问、内存所有权、Host 注销/重注册及跨进程生命周期；不能仅删除配置限制。 |
| INT8/FP8 MLA KV | 定义含 scale 的 Host 布局，选择 gather 后反量化或兼容 packed KV 的 attention 路径；index cache dtype 与 MLA KV dtype 分开处理。 |

当前内核仅支持 BF16 MLA latent=512、RoPE=64。Top-K 需能被 block size 整除，逻辑槽位与 selected-KV 行数还受内核 int32 地址计算上限约束；扩展布局或容量需同时修改容量检查、AOT 特化和边界测试。

## 已完成验证与证据边界

- 定向测试共 **152 个不重复用例通过**：126 个 Python 用例（含实际 NPU 内核/图回放）和 26 个 C++ 配置及容量测试；完整 NPU xllm 可执行文件构建通过。
- 实际 GLM-5.3 W8A8 模型在 **16 NPU、TP16、CP1** 上启动，8192-token prefill warmup 和 ACLGraph **16/8/4/2/1** 图桶捕获通过。
- 短请求、2439-token 长请求检索，以及三路并发请求完成；后者在足够输出预算下均正常结束。成功运行日志未见 NPU 错误或 eager fallback。
- 上述结果证明本次配置的功能闭环，**不等同于完整数据集精度、所有组合兼容或性能收益**。

本次开发机证据保存在 `/home/xu/scripts/`：

- `glm53_hisparse_npu_tests.log`、`glm53_hisparse_indexer_graph_long.log`：内核及图回放验证。
- `glm53_hisparse_regression.log`、`glm53_hisparse_cpp_tests.log`：Python 回归及 C++ 测试。
- `glm53-hisparse-20260914-164910/`：成功的全模型启动及各 rank 日志。
- `glm53_hisparse_smoke.json`、`glm53_hisparse_concurrency.json`：请求结果。
- `start_glm53_hisparse.sh`：本次专用启动脚本。

这些是开发环境证据路径，不是仓库分发内容。

## 环境与生命周期仍需确认

本次环境的自定义 AICPU 包配置与实际动态库名称存在不一致，通过独立 OPP 副本及匹配的环境变量完成验证，没有修改系统安装包。后续应统一 xllm_ops 包配置、动态库和启动环境，在标准部署环境重跑 metadata eager/graph 与全模型测试，避免依赖开发机私有路径。

测试服务停止时未在预期时间内优雅退出，最终仅对本次专用二进制进程强制终止。尚不能把退出行为归因于 HiSparse；需要先对照关闭 HiSparse 的基线，再验证 stream 排空、Host 注销顺序和进程退出后的资源释放。

建议推进顺序：建立精度/性能基线及统计 → LRU/近似 LRU → 多工作区与跨层预取 → prefill 优化 → 按需求逐项扩展兼容性。每一步保留独立开关或可比较基线，并补齐对应 NPU/ACLGraph 验证。
