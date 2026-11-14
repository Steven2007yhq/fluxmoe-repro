# 实现范围与论文对应关系

依据是用户提供的 [FluxMoE v3 论文](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe.pdf)，其 arXiv 版本为2604.02715v3。作者未提供源码。本项目保留第一阶段 CPU 参考与模拟器，新增真实 CUDA 异步分页及 vLLM 0.29 MoERunner 接入。文件格式、接口和优化 kernel 为独立设计，不声称恢复作者内部实现。

## 论文到源码

| 论文位置 | 对应源码 | 实现及覆盖范围 |
| --- | --- | --- |
| §3 专家分页 | [model.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/model.py)、[runtime.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/runtime.py) | 同构 MoE 层，整层全部本地专家物化，计算前就绪；不借用未来路由预测 |
| §4.1 PagedTensor / Algorithm 1 | [paging.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/paging.py)、[events.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/events.py) | CPU 固定虚拟标识、真实双 bytearray 别名、RAW/WAR 依赖；普通层执行不重新映射 |
| §4.1 CUDA VMM 与 Tensor | [cuda_vmm.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/cuda_vmm.py)、[paged_arena.cpp](/run/media/steve/Lenovo/FluxMoE论文复现/native/paged_arena.cpp) | 真实 CUDA VA、两个物理 allocation handle、零拷贝 PyTorch view 和 storage lease；已接入真实异步 pager |
| §4.2 压缩 GPU backend | [codec.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/codec.py) | BF16 指数的 canonical Huffman 编解码；符号和尾数逐位保存；自定义格式带验证信息 |
| §4.2 原生 GPU 解码 | [cuda_codec.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/cuda_codec.py)、[huffman_decode.cu](/run/media/steve/Lenovo/FluxMoE论文复现/native/huffman_decode.cu) | 初始化时 CPU 校验与构建索引，持久 GPU 压缩源，在调用方 stream 上直接写入 VMM 地址 |
| §4.2 CPU/GPU 后端分配 | [planner.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/planner.py)、[runtime.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/runtime.py) | 带宽比例的整张量取整、down 优先、CPU 字节预算；H2D 是带宽约束的事件模型 |
| §3–4.2 真实异步流水 | [cuda_pager.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/cuda_pager.py) | 四个加载 stream，解码直写 VMM、pinned H2D、真实 RAW/WAR 事件与双缓冲预取；单 rank 固定后端放置 |
| §4.3 / Algorithm 2 | [planner.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/planner.py) | `rho`、CI、AG、恢复阈值；逐迭代元数据快照、旧版本读引用和回收条件；层重映射要求引擎边界 |
| §5 集成与执行 | [runtime.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/runtime.py)、[integrations/vllm.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/integrations/vllm.py) | 保留 CPU 参考流水；新增实际 vLLM 0.29 MoERunner 参数绑定与 eager hooks，已验证真实 TritonExperts 输出 |
| §6 容量条件 | [capacity.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/capacity.py) | 三个原模型的静态容量筛选，详见 [CAPACITY.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CAPACITY.md) |

## 实际执行与数值判定

`ToyMoE` 用固定随机种子生成 BF16 权重。每层执行 router、top-k、fused gate/up、SwiGLU、down 和残差；每次加载完整本地专家集合，即使部分专家没有收到 token。CPU 模拟器通过 `pool.read` 取得实际物化的字节，不使用一个替代函数直接返回参考输出。

每轮结束后，模拟结果与全部原始 BF16 权重常驻的 Python 参考路径对比，要求完全一致，并生成输出 SHA256。两侧都用 Python float 运算，因此这项相等性验证的是存储和调度路径，不等于 BF16 CUDA GEMM 的数值保证。

[torch_reference.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/torch_reference.py) 另用独立 PyTorch 矩阵运算实现相同模型，权重位模式先转为 BF16，再用 float64 计算。其小模型绝对误差容限为 `1e-10`，用于覆盖归约次序差异；这不是原模型困惑度或通用低精度误差阈值。CUDA 选项实际使用 GPU，设备不可用时明确失败。本机6层、batch 8的 CUDA 交叉验证已通过，最大绝对误差 `1.1102230246251565e-16`，原始记录见 [torch_cuda_reference.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/torch_cuda_reference.json)。

BF16 编解码测试覆盖全部65536种位模式、正负零、非有限值、单符号码表及损坏数据。NaN 位模式按整数存储位比较，避免浮点 NaN 比较规则掩盖位级错误。合成小张量可能因头部开销而膨胀，本项目不预设一定达到论文压缩率。

## CPU 模拟器的双缓冲与事件依赖

CPU `PagedTensorPool` 为 gate/up 和 down 各分配两个真实 bytearray 物理块；pageable 层按排序后的 ordinal 对2取模绑定。每张量有独立且稳定的整数虚拟标识，驻留层有独立物理块。写入会真正改变共享存储，并更新逐张量 owner；读取未物化或被其他层覆盖的别名会抛 `HazardError`。

事件调度器以完成时间执行回调。每 stream 串行；所有 H2D 流共用一个排他 `h2d_link`，两个解压流共用一个排他 `decoder`，不能因增加流数而获得无限聚合带宽。两个资源可互相重叠，也可与计算重叠；这不包含 GPU SM、DRAM、PCIe KV 交换等资源竞争模型。

加载复用某物理块前依赖其最后读事件，保证 WAR；计算依赖本层所有加载事件，保证 RAW。CPU 数值回调在整层 compute 事件完成时同时读取两个 projection，因此二者的 last-read 都延续到该整层事件结束。这是保守的验证实现，没有模拟 gate/up 与 down kernel 之间更细的释放窗口。

开启预取时，最初提交前两层加载，此后计算完层 i 才允许向复用块提交 i+2 的加载。关闭预取是关闭未来层预取；当前层加载仍可以与当前层的 nonexpert 时间窗口重叠。`layer_materialization_ms` 从请求发出计到所有本地后端完成，包含共享资源排队；它不是单纯传输服务时间，也不是 exposed stall。

`tests/test_runtime.py` 覆盖1/2/3/6层、五种策略和预取开关，共40种组合；检查与 resident 输出一致、trace 因果依赖、资源串行，以及删除 RAW/WAR 依赖后能否触发错误。它证明已测小模型条件下的路径正确性，不证明真实 GPU 调度性能。

## CPU 模拟器的五种策略与公平性边界

| strategy | 放置方式 | 对照用途 |
| --- | --- | --- |
| `resident` | 全部专家 BF16 常驻 | 数值参考和容量对照 |
| `compressed` | 全部 pageable 专家在逻辑 GPU 后端压缩存储 | 仅压缩、双缓冲和预取的影响 |
| `hybrid` | 层内部分整张量 CPU 后备，其余压缩后备 | 分层混合后端机制 |
| `whole_layer_cpu` | 预算内整层 CPU 后备，其余层 BF16 常驻 | vLLM-O 思路的简化机制对照 |
| `whole_layer_hybrid` | 预算内整层 CPU 后备，其余层压缩后备 | FluxMoE-H 思路的简化机制对照 |

这些名称不表示实际运行了 vLLM-O、FluxMoE-H 或 KTransformers。整层策略向下取整后可能没有用满 CPU 预算；报告保留各自实际 host/device 字节，不能仅凭相同预算上限就声称实际占用完全相同。`toy.json` 用8层，使12.5%预算恰能容纳一个层。

## 驻留规划与 KV 边界

控制器使用 `rho = (remaining_batch / active_batch) * forward_ms / sum(load_ms)`，按 CI 检查，按 AG 选择整张量。满足 batch pressure 且 `rho>1` 时尝试回收，`rho<0.9` 时恢复最近一次回收；down 张量优先，恢复受初始状态和历史约束。元数据变更先 stage，在下一迭代边界 publish，已有读引用完成后旧版本才可回收。

模拟运行可以切换 compressed/CPU 元数据，权重副本来自已有参考字典。**新副本准备的实际传输/解压时间和新旧副本同时存在的峰值容量尚未建模。**因此稳态内存账本和元数据生命周期测试，不能被解读为完整动态迁移性能。

层级 resident/pageable 改变被标记为需要引擎重启；本次 runtime 不在单次 launch 中更改 PagedTensor 映射。KV 是固定的字节保留量，不模拟 KV 内容、swap、请求调度或 active batch 的自动变化。`remaining_batch` 是外部控制输入，不是被仿真的请求队列。规划器释放专家存储后不会自动扩大 KV，也不会凭空增加吞吐。

`planner.json` 用 CI=1 和 CPU fraction=0.5 专门触发控制路径，不能拿它与论文 CI=16、CPU专家预算12.5%的主实验直接比较。论文 Exp#6 的41次引擎重启模拟协议尚未集成成本项目运行入口。

当前真实 `CudaMoEPager` 采用构造时固定的后端放置，所有层均 pageable；它尚未消费上述 planner 的在线调整决策。默认 CPU 放置按预算选择整张量，down 优先并按层平衡；也可以显式传入 `cpu_keys`。这与 CPU 模拟器的带宽比例初始化是两个独立入口，不能混称为完整 GPU 驻留控制器。

## 原生 CUDA 组件与真实异步分页

Huffman 解码模块与 CPU 格式共用位级语义，但 GPU chunk 索引、树表示和 kernel 分工为本项目独立设计。第一阶段 `decode_cuda_with_metrics` 同步入口仍可用：先在 CPU 完整验证/解码并建立索引，再启动 GPU kernel、拷回结果；原始 [RTX 4060 smoke 记录](/run/media/steve/Lenovo/FluxMoE论文复现/results/cuda_codec_smoke.json) 保留。新 `CudaCompressedTensor` 将验证、索引构建及压缩源上传移到 setup，源在 GPU 持久保存；`decode_into` 将调用方 stream 与目标 VMM 地址交给原生 kernel。它在逐层 hot path 不进行 CPU 解码、D2H、设备同步或解码输出的临时分配。源的错误检查与最终释放依赖完成事件，在显式边界执行。构建与接口见 [CUDA_CODEC.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_CODEC.md)。

VMM 模块调用 `cuMemAddressReserve`、`cuMemCreate`、`cuMemMap` 和 `cuMemSetAccess`。一个 arena 维护两个按设备 granularity 对齐的物理块，多层保留独立 VA 并映射到交替物理块。它查询设备 VMM 支持，保留/释放自身的 primary context 引用，不重置共享 context。同步 copy 接口继续用于独立检查；新增 `arena.tensor()` 将真实 VA 暴露为零拷贝 PyTorch Tensor，派生 view 和 Parameter 引用通过 storage lease 保持 arena 存活，存在外部 Tensor 时显式 close 会拒绝释放内存。详见 [VMM_TORCH.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VMM_TORCH.md)。

VMM 本机 GPU 验证已经通过：[cuda_vmm_smoke.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/cuda_vmm_smoke.json) 记录6个真实且稳定的 VA、两个物理块以及9项通过的检查，包括偶数/奇数层别名、跨块隔离、覆盖传播和关闭。该设备 allocation granularity 为2MiB，因此6层各1031字节的例子实际分配4MiB，不能将此小例子作为显存节省证据。编译、依赖与 API 见 [CUDA_VMM.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_VMM.md)。

`CudaMoEPager` 为 gate/up 与 down 分别创建解码和 H2D stream，共四个加载 stream。CPU 后备张量使用真实 pinned memory，并以 `copy_(..., non_blocking=True)` 写入对应 VMM slice；其余张量调用 `decode_into` 直接写入相同目标布局。全部本地专家都会物化，不依赖未来路由。

执行协议为 `begin_iteration → acquire(layer) → 提交本层计算 → release(layer) → … → end_iteration`。`acquire` 在计算 stream 上等待各加载完成事件以满足 RAW；`release` 在最后一项权重读取之后记录事件，后续覆盖同一物理块的加载 stream 等待该事件以满足 WAR。前两层先预取，释放层 i 后预取 i+2；跨迭代仍保留最后读依赖。调用方必须将全部权重读者提交到 acquire 对应的 stream，或先显式 join 外部 side stream。

`begin/acquire/release/end` 不做 host/device/stream 同步。结果验证、`trace()` 和 `close()` 是显式同步边界。当前协议拒绝乱序 acquire、未完成 release、超过两层窗口的预取和 CUDA Graph 捕获。trace 记录真实 CUDA events，默认 `max_trace_events=10000`；超过上限的记录计数为 `trace_events_dropped`，不无限累积时间戳事件。内存报告计入对齐后的 VMM 物理块、压缩设备源及 pinned 专家，仍不代表完整引擎峰值。

## 已验证的真实 vLLM 模块接入

本机 `/home/steve/vllm_env/bin/python` 中的 vLLM 0.29.0、PyTorch 2.13.0+cu130 已运行实际 `MoERunner → UnquantizedFusedMoEMethod → TritonExperts`。4层、每层4专家、H=64、I=128、batch=16，在非默认 stream 上连续3轮输出与同一 vLLM 常驻 BF16 对照逐位一致，最大误差0，8个绑定后的真实 VMM 地址稳定。证据见 [vllm_paging_smoke.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/vllm_paging_smoke.json) 和 [执行日志](/run/media/steve/Lenovo/FluxMoE论文复现/results/vllm_paging_smoke.log)。

适配器在权重 `process_weights_after_loading` 完成后绑定 `runner.routed_experts.w13_weight/w2_weight`，保留 Parameter 身份，以 eager forward hooks 调用 acquire/release。它保留 CPU 恢复副本，不保留原 GPU 参数 storage；验证 bootstrap 仍经历初始全驻留 GPU 峰值。量化、TP/EP、EPLB、DBO、CUDA Graph 等未覆盖模式会被拒绝。完整说明见 [VLLM_INTEGRATION.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VLLM_INTEGRATION.md)。

这证明实际 vLLM MoE 算子读取分页后的 VMM 权重；尚未启动完整 LLM 引擎、注意力/KV 调度或真实检查点评测。当前 CUDA 验证发生在本机 4060，没有 A40 性能证据。

原生模块测试需要 `FLUXMOE_TEST_CUDA=1` 显式启用，可用 `/home/steve/torch_env/bin/python` 运行。真实 vLLM 测试需要另设 `FLUXMOE_TEST_VLLM=1`，使用 `/home/steve/vllm_env/bin/python`。源码存在、编译成功和 GPU 执行通过是不同证据，应分别查看构建记录与测试原始日志；本次没有改动两个已有环境。

## 证据与重算

单次模拟的 `result.json` 包含配置、输入生成种子、模型及输出指纹、事件原始记录、逐轮时间、放置版本、内存账本、planner 决策和状态。`manifest.json` 保存解释器/平台信息及源码与配置 SHA256。逻辑 GPU 账本包含物化块、压缩头与后备数据、router、KV及workspace；不包括作为测试夹具保留的完整 Python oracle，也不是 RSS 或 `nvidia-smi` 读数。

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'
python3 -m fluxmoe validate --output results/validation
python3 -m fluxmoe simulate --config configs/toy.json --output results/demo
python3 -m fluxmoe sweep --config configs/toy.json --output results/sweep
python3 -m fluxmoe report --input results/sweep/runs.jsonl --output results/sweep/report.md

/home/steve/torch_env/bin/python -m fluxmoe.async_validation --output results/async_paging
/home/steve/vllm_env/bin/python -m fluxmoe.integrations.vllm --output results/vllm_paging_smoke.json
```

`trace.json`/各配置 `.trace.json` 是由相同原始事件导出的 Chrome Trace 数据，单位转换写在 [report.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/report.py)；`report` 从保存的 JSON/JSONL 重建表格。OOM 保留为失败状态和缺失指标，不参加零值加速比。配置和模型确定时重复得到相同模拟输出，不构成硬件重复测量或统计置信区间。

## 推进 A40 与完整 vLLM 引擎尚缺的工作

1. 保持版本差异可追溯：论文 vLLM 0.23.0 与当前0.29 adapter不同；运行其他版本前重新检查布局、postprocess 和执行路径。
2. 在 checkpoint loader 阶段准备 CPU/压缩后备并注册分页 storage，避免当前先加载全驻留模型再绑定的初始化峰值；接入完整 LLM 引擎。
3. 将驻留 planner 接入真实 GPU 后端，补齐后端变更的新副本准备、临时双份峰值及延迟释放；当前GPU配置固定，在线调整仅在CPU参考路径验证。
4. 在目标 A40 拓扑下验证 TP/EP 分片、KV头复制、GPU间通信与 NUMA/H2D 带宽；本项目现阶段是单 rank，没有 collective。
5. 接入实际注意力、KV allocator 和请求调度器；从真实 scheduler 读取 batch pressure，区分固定 KV 的单 launch 实验和论文41次重启的模拟协议。
6. 用原始模型权重进行逐层数值与 WikiText-2 质量验证；补齐 vLLM、vLLM-O、FluxMoE-H、KTransformers 的实际对照和相同资源预算。
7. 固定 ShareGPT 请求轨迹、精度、输出长度、计时边界与重复方式，采集未经 profiler 干扰的性能和单独的机制诊断，执行正文与附录矩阵。

截至本交付，三个原始大模型与上述系统性能实验均未运行。现有实现为这些后续工作提供已执行的正确性参考、测试和记录格式，不构成整篇论文已经复现的结论。
