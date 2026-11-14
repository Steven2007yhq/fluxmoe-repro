# FluxMoE 独立实现：CPU 模拟与异步 GPU 分页

本项目根据 FluxMoE v3（arXiv:2604.02715v3）独立实现专家分页的核心机制。当前同时提供 CPU 离散事件模拟和真实 CUDA 异步执行路径：GPU 解码直接写入 VMM 双缓冲，pinned-host 权重通过非阻塞 H2D 加载，计算与预取使用 RAW/WAR 事件衔接；已接入本机 vLLM 0.29 的实际 MoERunner。作者未公开源码，因此本项目的文件格式、接口和未明确的调度选择都是独立设计。

**当前已在本机 RTX 4060 验证单卡异步分页、vLLM MoE 模块及完整 `LLM.generate` 引擎流程。**随机初始化 tiny Mixtral 的全驻留/分页对照已[通过](/run/media/steve/Lenovo/FluxMoE论文复现/results/engine_validation/comparison.json)；原论文三个大模型与论文性能复现尚未完成。CPU 模拟时间与真实 CUDA 时间线分别保存，本机 RTX 4060 的正确性结果不代表 A40、L40S 或 H20 的吞吐。

真实分页的调用顺序、资源生命周期和验证入口见 [异步分页说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/ASYNC_PAGING.md)。

## 真实 GPU 分页与 vLLM 验证

以下命令使用已有环境，不安装或修改依赖。首次使用需按 [CUDA 解码说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_CODEC.md) 和 [VMM 构建说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_VMM.md) 构建当前源码对应的原生库。

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'

/home/steve/torch_env/bin/python -m fluxmoe.async_validation --output results/async_paging

/home/steve/vllm_env/bin/python -m fluxmoe.integrations.vllm --output results/vllm_paging_smoke.json
```

[真实 vLLM 运行记录](/run/media/steve/Lenovo/FluxMoE论文复现/results/vllm_paging_smoke.json) 已验证4层实际 `MoERunner → UnquantizedFusedMoEMethod → TritonExperts`：batch 16、连续3轮非默认 stream 执行，与全驻留 BF16 对照逐位一致，最大误差0，绑定后的8个 VMM 参数地址稳定。该入口使用合成权重，初始化必要的单 rank 分布式组与 worker workspace，没有启动完整 LLM 服务。接入方式和限制见 [VLLM_INTEGRATION.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VLLM_INTEGRATION.md)。

## 完整 vLLM 引擎验证

新增入口构建本地真实 BF16 safetensors 检查点与 tokenizer，再分别运行全驻留和分页 `LLM.generate`，检查权重、KV 写入/历史读取、生成 token 和 logprob。tiny Mixtral 是随机初始化的流程 fixture，不是预训练语言模型，不评估语言质量。

本机[完整对照结果](/run/media/steve/Lenovo/FluxMoE论文复现/results/engine_validation/comparison.json)已通过：11 个请求、176 个生成 token 一致，176 个生成及 200 个 prompt logprob 最大误差均为 0；每侧完成 454 次 KV 逐 bit 写入检查和 165 次历史 attention 对照。移除 KV 诊断 hooks 后，另外 3 个请求、48 个生成 token 仍与各自之前的结果一致。权重、缓存覆盖、容差和原始记录见 [本机结果](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VLLM_ENGINE.md)。

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'

/home/steve/vllm_env/bin/python -m fluxmoe.engine_validation \
  --output results/engine_validation_run01

/home/steve/vllm_env/bin/python -m fluxmoe.engine_validation \
  --output results/engine_validation_run01 --compare-only
```

每次完整运行使用新目录，原始 JSON、日志和比较报告保留在该目录。初次加载仍有完整模型常驻 GPU 的峰值；固定 1 MiB KV 池使两侧预算一致。KV 诊断主动同步，本入口不提供性能结论。Loader、worker extension、API 和适用边界见 [VLLM_ENGINE.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VLLM_ENGINE.md)。

## A40 验收与真实权重微实验

用户手动在 A40 上运行后，已拷回的原始 JSON/日志通过了[本机独立核验](/run/media/steve/Lenovo/FluxMoE论文复现/results/a40_server/a40-verify.IhdPv5/local_audit.md)：146项测试通过，同机全驻留/分页生成与KV核验通过。这仍是随机tiny Mixtral的单卡正确性结果。

新增 [表3/图11手动实验入口](/run/media/steve/Lenovo/FluxMoE论文复现/docs/REAL_WEIGHT_EXPERIMENTS.md)：固定官方checkpoint revision，按所需层下载经过SHA256校验的分片，分析原始BF16 exponent/mantissa，使用计时外C++准备器和双CUDA stream稳态解码，并从原始样本生成CSV、PNG及PDF。Mixtral第0层的A40真实权重pilot已完成，旧tree解码约4.07GB/s，图11分布和原始计时已独立核验；Qwen、GLM及完整网格尚未运行。

[CUDA评审改进与A40手动验证](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_DECODER_REVIEW.md)新增`buffered/packed/vectorized`对照，默认`packed`在原bulk基础上使用uint32打包读写、34列shared tile和尾部回退；位流缓存单独保留作消融。`tree/lookup/tiled/bulk`仍可显式选用。优化后的A40性能须由用户重新编译并实测，本机结果单独记录。上一轮原始4GB/s核验及优化见[历史说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_DECODER_OPTIMIZATION.md)。

## 独立 CPU 模拟

CPU 主路径只需要 Python 3.10+ 标准库，不需要 `pip install`。以下命令都从本项目目录执行：

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'

python3 -m fluxmoe validate --output results/validation

python3 -m fluxmoe simulate --config configs/toy.json --output results/demo

python3 -m fluxmoe sweep --config configs/toy.json --output results/sweep

python3 -m fluxmoe report --input results/sweep/runs.jsonl --output results/sweep/report.md

python3 -m fluxmoe simulate --config configs/planner.json --output results/planner

python3 -m fluxmoe capacity --model mixtral --gpu a40 --gpus 2 --kv-gb 8 --workspace-gb 2
```

`validate` 保存测试原始日志。`simulate` 运行同一组随机种子生成的 BF16 权重与输入，并逐轮核对常驻参考输出。`sweep` 比较五种策略、开启/关闭预取及默认 batch 2/4，共20个配置点；它关闭动态 planner，以免同时改变多个因素。OOM 记录为 `status=oom`、`metrics=null`，不会填入零吞吐或删除失败点。单次 `simulate` OOM 返回退出码2；`sweep` 即使存在 OOM 也会继续保存整个矩阵，需检查每条记录的状态。

静态 `capacity` 命令不读取 GPU、不下载模型。其零 KV/workspace 默认值仅给出权重及双缓冲下界；完整解释见 [容量估算说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CAPACITY.md)。

## 已实现路径

| 路径 | 实际执行的工作 | 当前边界 |
| --- | --- | --- |
| CPU MoE | top-k 路由、fused gate/up、SwiGLU、down、残差计算 | 合成小模型；权重 BF16 存储，Python float 累加 |
| BF16 编解码 | 指数 Huffman 编码，符号与尾数保持逐位不变 | 自定义版本化格式，不是作者未公开的格式 |
| CPU PagedTensor | 真实 bytearray 双缓冲、固定虚拟标识、跨层别名与读写检查 | 标识不是真实 CUDA VA |
| 离散事件 | 流依赖、RAW/WAR、预取、共享 H2D/decoder 资源排队 | 带宽与计算时间是显式模型参数 |
| 驻留 planner | CI/AG、下投影优先、元数据版本发布、旧副本回收条件 | 单 rank；KV 保留量固定；迁移成本尚未建模 |
| 独立 PyTorch oracle | 用独立矩阵运算核对同一小型 MoE | float64 累加，用于数值交叉验证 |
| 原生 CUDA Huffman | 持久 GPU 压缩源，指定 stream 直接解码到 VMM 地址 | setup 时 CPU 校验/建立索引；解码 hot path 无 D2H、CPU 解码或设备同步 |
| 原生 CUDA VMM / Torch view | CUDA Driver API 固定 VA 双缓冲，零拷贝 PyTorch Tensor 和 storage lease | close 拒绝释放仍有外部 Tensor 引用的存储 |
| 真实异步分页 | 四个加载 stream、pinned H2D、解码、预取、RAW/WAR CUDA events | 单 rank、固定放置、全 pageable；GPU 路径尚未接在线驻留调整 |
| vLLM MoE adapter | 真实 MoERunner 参数绑定和前向 hooks，使用实际 TritonExperts 算子 | vLLM 0.29、单 rank、BF16、eager；不是完整 LLM 引擎 |
| vLLM 完整引擎入口 | 磁盘权重加载、真实 attention/KV/request scheduler、tokenizer 与 `LLM.generate`，全驻留/分页对照 | 随机 tiny Mixtral；单 rank eager BF16 Triton，固定 KV；首次加载仍需完整 GPU 权重 |

论文到源码的映射、同步语义和剩余工作见 [实现说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/IMPLEMENTATION.md)。

## 可选 PyTorch / CUDA 验证

当前机器已有 `/home/steve/torch_env/bin/python`（PyTorch 2.14，cu126）以及 RTX 4060 Laptop 8GB。以下命令明确使用该解释器，不修改现有环境：

```bash
/home/steve/torch_env/bin/python -m fluxmoe torch-check --device cpu --output results/torch_cpu.json

/home/steve/torch_env/bin/python -m fluxmoe torch-check --device cuda --output results/torch_cuda.json

FLUXMOE_TEST_CUDA=1 /home/steve/torch_env/bin/python -m unittest discover -s tests -v

FLUXMOE_TEST_VLLM=1 /home/steve/vllm_env/bin/python -m unittest tests.test_vllm_adapter -v
```

不设置 `FLUXMOE_TEST_CUDA=1` 时，原生 CUDA 测试会跳过；vLLM 测试另需 `FLUXMOE_TEST_VLLM=1`，PyTorch 未安装时其独立参考测试也会跳过。跳过不等于已验证。显式请求 CUDA 而设备不可用时会报错，不会静默转为 CPU。

独立 PyTorch CUDA oracle 已在本机通过：6层、batch 8，最大绝对误差约 `1.11e-16`，详见 [torch_cuda_reference.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/torch_cuda_reference.json)。这验证小模型数值路径，不是 GPU 性能测试。

原生 Huffman 模块使用独立 C ABI 共享库。第一阶段的同步 smoke 记录仍保存在 [cuda_codec_smoke.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/cuda_codec_smoke.json)，其端到端时间包含 CPU 校验、chunk 索引和 D2H。新异步路径使用 `CudaCompressedTensor`，仅在初始化验证和上传后备数据，随后 `decode_into` 在指定 stream 直接写入 VMM，验证检查与回收在边界完成。两类记录均不能直接当作论文约400GB/s解码性能的复现。构建与使用见 [CUDA 解码说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_CODEC.md)。

VMM 源码位于 [cuda_vmm.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/cuda_vmm.py) 和 [paged_arena.cpp](/run/media/steve/Lenovo/FluxMoE论文复现/native/paged_arena.cpp)。第一阶段6个真实 VA / 2物理块的记录保留于 [cuda_vmm_smoke.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/cuda_vmm_smoke.json)。新的 `arena.tensor()` 导出共享存储的 PyTorch view，配合 [cuda_pager.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/cuda_pager.py) 的 CUDA stream/event 执行异步分页；同步 `read/write` 原语继续用于独立检查。Tensor 与 arena 的生命周期规则见 [VMM_TORCH.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VMM_TORCH.md)。

本机 `vllm_env` 的 vLLM 0.29已完成上述 MoERunner 模块适配；论文使用 vLLM 0.23.0。项目没有为此降级或改动环境，当前结果按版本迁移验证报告。

## 配置和证据

[toy.json](/run/media/steve/Lenovo/FluxMoE论文复现/configs/toy.json) 使用8层，使12.5%的 CPU 权重预算能够容纳一个完整层，便于观察整层卸载路径。所有带宽默认值都刻意缩放到小张量量级，未按 A40 标定。

[planner.json](/run/media/steve/Lenovo/FluxMoE论文复现/configs/planner.json) 刻意使用 `control_interval=1`、`cpu_fraction=0.5`，在短运行中触发多次元数据变化。这是控制逻辑测试，不是论文12.5% CPU 专家预算下的性能配置。

单次运行保存：

- `result.json`：完整配置、种子、模型/输出指纹、状态、数值误差、内存账本、逐轮记录和原始事件。
- `trace.json`：Chrome Trace 格式的模拟时间线，显示流、加载、计算和依赖。
- `manifest.json`：执行解释器、平台、时间与源码/配置 SHA256。
- `report.md`：从原始记录生成的可读表格。

扫描运行的原始数据在 `runs.jsonl`，并为每个配置输出时间线。上面的 `report` 命令可以重新生成汇总；模拟器具有确定性，重复相同配置不会产生真实硬件噪声，不能据此构造硬件性能置信区间。

GPU pager 的 `trace()` 使用真实 CUDA events，读取时间线时会执行显式同步，不是 CPU 事件模拟。记录默认最多保留 `max_trace_events=10000` 条；超出部分计入 `trace_events_dropped`，因此长运行不能把截断 trace 当作完整性能窗口。

## 下一步：A40 与大模型验证

当前缺项包括：在 checkpoint loader 阶段避免初始全驻留 GPU 峰值；把驻留规划器接入真实 GPU 后端并覆盖切换期间的新旧副本峰值；支持 TP/EP 分片、collectives 和 CUDA Graph；将小模型的引擎验证扩展到预训练检查点、长期请求负载与动态 KV/专家预算；重建 vLLM-O/FluxMoE-H/KTransformers 公平对照；运行原始检查点、ShareGPT/WikiText-2 和论文实验矩阵。

这些步骤需要在目标 A40 服务器上验证。本机 RTX 4060 的真实异步分页和 MoE 模块正确性结果不能代替目标平台性能证据。

## 四卡长负载 KV 正确性

固定 ShareGPT 32×4,096，依次执行四卡 BF16 native host-KV 取证与同布局参考回放。通过条件包括同一解码块逐位一致、四 rank 完成与退出、长负载完整输出和抽样数值复核。运行、回收及离线复核命令见 [四卡长负载 KV 资格](docs/HOST_KV_LONG_QUALIFICATION.md)。
