# FluxMoE v3：Mixtral 图表、实验条件与当前复现边界

**范围（2026-11-14 起）：只复现论文中与 Mixtral-8×7B-Instruct 相关的图表实验。GLM-4.5 与 Qwen3-Next 不在范围内。** 范围调整前覆盖三模型的版本原样保留在 [PAPER_EXPERIMENT_MAP.three-models-20261031.md](PAPER_EXPERIMENT_MAP.three-models-20261031.md)。

本文件中的论文数值一律标为 paper-reported。论文设置、容量估算和计划项都不是本项目的性能结果。状态栏只引用已经回收并审计的证据。

## 依据与当前证据

- 论文：`fluxmoe.pdf`，arXiv:2604.02715v3，2026-09-10。[论文页面](https://arxiv.org/abs/2604.02715v3)。本地 PDF SHA256 为 `1ab3be9721c1e501f7c2171af919722b5271e284089bf76b4f39f00bf89975c3`。
- 模型：Mixtral-8x7B-Instruct-v0.1，revision `eba92302a2861cdc0098cc54bc9f17cb2c47eb61`，共 19 个 BF16 分片。目标机上已有原始分片、cache512 压缩容器和 native 产物。
- 当前框架为 vLLM 0.29.0（论文为 0.23.0）。目标机环境：Python 3.12.16、Torch 2.13.0+cu130、CUDA 13.2。
- **当前多 rank 运行时代码只存在于封存包中**，例如 [step373 包](../results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/step373-short-replay-prep-r2.f1Jf7uIn/package) 里的 `pp1_*.py` 和 `ref4_*.py`。主源码树 `fluxmoe/` 仍是 10/31 的单 rank 版本，其 `integrations/engine.py` 会先完整加载 GPU 权重，并拒绝多 rank。正式实验前应先把封存包代码合回主源码树。

已审计的 Mixtral 证据：

| 证据 | 结论 | 来源 |
| --- | --- | --- |
| 32×4096 长负载资格（2×A40，TP2/PP1/EP2，11/11） | H（FluxMoE-H）与 B（FluxMoE）各完成 32 个请求、131,072 个输出 token，两臂 greedy 输出逐位一致。17 个 host-KV 解码页在两 rank 全部 32 层完成换出、释放、回填和实际 attention 读取，位型逐位一致。运行时开着诊断采集，不能作为性能数据 | [H/B 资格](../results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/tp2pp1-long-hb-gpu01-h-prep.6AUlL7gG/HANDOFF.md) |
| 四卡独立常驻 BF16 参考（TP2/PP2/EP2） | 与 H/B 相比：748/131,072 个 greedy 预测不同，172/209 个 logits 行超出 `atol=rtol=1e-5` | [四卡参考](../results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/tp2pp2-resident-bf16-reference-prep.zCPozoMj/HANDOFF.md) |
| 偏差根因（11/14） | step 60 的分歧来自参考侧 EOS 掩码未对齐，已修正。step 395 的分歧追到 step 373：B 在 CPU 上构建模型，RoPE cos/sin 缓存在 CPU 上计算，参考在 GPU 上计算。`inv_freq[18]` 差 1 个 fp32 ulp，在位置 459 翻转了 BF16 舍入。与卸载、KV 存储和 host-KV 恢复无关。修正（B 在 GPU 上重建缓存）与修正后的全长复验均待执行 | [step373 报告](../results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/step373-short-replay-prep-r2.f1Jf7uIn/STEP373_LIVE_REPORT.md) |
| 四卡固定 batch ABBA（batch 3 × 8 token） | B 的耗时比 A 低 42.0%。只是每臂两个引擎的描述性结果，工作量很小 | [ABBA 审计](../results/static-abba-complete-review.irqmc4_x/REPORT.md) |
| A40 解码微基准 | K1 单投影 139.8 GB/s（第 0 层 expert0/1 gate_up，M=256 窗口），不是表 3 协议。>10-bit 长码只占 0.059%，不是主要瓶颈 | [K0/K1 报告](../results/decoder_review_a40/k0k1-priority.443m_gi2/deliverable/report.md) · [长码统计](../results/decoder-longcode-stats.klxgqxg2/REPORT.md) |
| 早期正确性基础 | RTX 4060 上随机 tiny Mixtral 的整引擎全驻留/分页对照逐位一致；A40 上 146 项测试通过 | [VLLM_ENGINE.md](VLLM_ENGINE.md) · [A40 核验](../results/a40_server/a40-verify.IhdPv5/local_audit.md) |

## 共同实验设置

### 模型、硬件与论文预算

| 项目 | 论文 Mixtral 条件 | 本项目 A40 移植条件 |
| --- | --- | --- |
| 模型 | 47B 总参数 / 45B routed（95.7%），32 个 MoE 层，每层 8 个专家，top-2，BF16 权重约 94 GB | 同一模型；固定 revision 和 tokenizer |
| 主实验部署 | 2×L40S，TP/EP 2/2；按表 2，90% 预算 86.9 GB 比权重 94 GB 少 7.1 GB，全驻留 vLLM 不可部署 | 2×A40，同一 NV4 配对，TP2/PP1/EP2/DP1 |
| 4 卡实验 | 图 1/2 用 4×L40S 全驻留 vLLM；表 7 用 TP/EP 4/4 | 4×A40：两对 NV4，对与对之间走 SYS |
| GPU | 48.3 GB（`nvidia-smi`），PCIe Gen4，H2D 理论峰值 32 GB/s，无 NVLink | 驱动可见容量 48.3052 GB，与 L40S 相同；PCIe Gen4 x16。BF16 稠密算力约为 L40S 的 1/2.4，显存带宽 696 对 864 GB/s |
| CPU / 主存 | 双路 Xeon Gold 5318Y，共 48 物理核（AVX-512），2 TB RAM | AMD Threadripper PRO 5995WX（64 核，Zen 3，**无 AVX-512**），约 541 GB RAM，8 个 NUMA 节点 |
| 预算 | 每卡 90% GPU 预算；512 GB 聚合 host-DRAM KV；使用 CPU 专家后备的系统统一为 routed 权重的 12.5% | 12.5% 不变；host-KV 必须按实际主存缩小（见"容量与预算"） |

论文没有给出完整的驱动、CUDA、软件 commit、请求抽样种子和 NUMA 绑定细节。这些都需要在目标环境固定下来，并标为差异。

### 系统与放置

| 名称 | 论文含义与公平条件 | 本项目实现状态 |
| --- | --- | --- |
| vLLM | 全部权重常驻 GPU。2×L40S 上不可部署，不能填 0 吞吐去算加速比 | 只用于 4 卡实验（图 1/2、表 7、迁移补充） |
| vLLM-O | 后部 MoE 层按预算放 CPU，前部未压缩常驻，逐层预取 | 未搭建。候选：vLLM 0.29 原生 prefetch 卸载（`offload_group_size`、`offload_num_in_group`、`offload_prefetch_step`），需先核实卸载范围是否严格为 12.5% 的 routed 专家；对不上就在 pager 中实现"未压缩常驻 + 尾层 CPU"策略 |
| FluxMoE-H | 前部层 GPU 压缩，后部层 CPU 后备，整层只用一个后端 | 已实现为静态放置 A/H：全局层 28–31 的本地专家投影放 CPU，每 rank 5.25 GiB |
| KTransformers | CPU/GPU 专家计算，只用 TP；论文为 48 线程、两个 NUMA 线程池 | 未搭建。需固定版本、线程数和 NUMA 设置。服务器 CPU 无 AVX-512，属于对 KT 不利的平台差异，必须标明 |
| FluxMoE | Mixtral 固定 12.5% 放置：每层一个 GPU 卸载一个 fused gate/up tensor，另一个 GPU 卸载一个 down tensor，共 64 个 tensor；不用默认 down-first 选择器代替 | 已实现为静态放置 B：TP0 的 local expert 0 gate/up 和 TP1 的 local expert 0 down 放 CPU，CPU 侧 7/3.5 GiB |
| FluxMoE-TP | 同样的机制、GPU 与驻留配置，关闭 EP，Mixtral 为 TP2 | 未实现 |

### 负载与指标

- 离线负载：ShareGPT prompt，所有请求初始即可用。论文中 Mixtral 平均输入 52.1 token，最大 1121。两组扫描：`Bcfg ∈ {32, 64, 128, 256}`、输出固定 4096；或输出 `∈ {1024, 2048, 3072, 4096}`、Bcfg 固定 256。两组合计 7 个不同的点。
- 论文报告 5 次运行的均值和 Student-t 95% CI。本项目必须保存固定请求清单、token ID、revision 和 seed。当前冻结的 32 请求清单为 ShareGPT 固定版本、分层抽样，平均输入 46.69 token，见 [首点方案](../results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/pp1-long-p32-l4096.kTmCeXpc/HANDOFF.md)。Bcfg 为 64/128/256 时需另外冻结清单。
- 指标：输出吞吐（tokens/s）、平均 TTFT、平均 TPOT；在线实验报告逐请求 E2E 延迟 CDF。TTFT/TPOT 包含各自定义范围内的排队与调度，不能用 GPU forward 时间代替，也不能从总吞吐反推。

## 图表对应表（Mixtral）

状态分为四类：**已完成**、**部分**、**前置就绪**、**未开始**。论文参照值只用于核对趋势，不与本项目测量混为同一来源。

| 论文位置 | 图形与指标 | 论文条件与参照值（paper-reported） | A40 移植条件与前置项 | 状态 |
| --- | --- | --- | --- | --- |
| 图 1，§2.1 | 堆叠容量柱图：计算所得 KV、权重、其他运行时；横轴为 Bcfg 或输出长度 | 4×L40S 全驻留 vLLM；权重 94 GB，预算 173.9 GB | 4×A40 全驻留 vLLM TP4。KV 按 config 和请求 token 算术计算；运行时显存实测，未测就标"未知" | 未开始，成本低 |
| 图 2，§2.2 | 吞吐与平均 Bact 分组柱图，横轴 Bcfg 32–256 | 4×L40S，输出 4096，vLLM 开 KV offload，只改聚合 GPU KV（20/40/60 GB）。Bcfg=32 时各配置差异 <0.3%；Bcfg=256 时 60 GB 比 20 GB 的 Bact 高 36.0%、吞吐高 70.0% | 4×A40，只用 vLLM。至少要有一个"整批可调度"点和一个"KV 受限"点；记录每步 Bact、抢占和实际 KV。可复用 [四卡 host-KV 资格](HOST_KV_LONG_QUALIFICATION.md) 中的 OffloadingConnector 配置 | 未开始 |
| 图 3，§4 | 系统结构示意 | — | 画实现架构，注明为独立设计，不填性能数据 | 未开始 |
| **图 5，Exp#2** | 吞吐、TTFT、TPOT 柱图；两组扫描 | 2×L40S；FluxMoE、FluxMoE-H、vLLM-O、KT 使用相同的 12.5% CPU 专家预算和 host-KV。Bcfg=256、输出 4096 时：FluxMoE 吞吐为 vLLM-O 的 8.7×、KT 的 4.3×；TTFT/TPOT 比 vLLM-O 低 19.8%/86.9%，比 KT 低 99.9%/29.1%；比 FluxMoE-H 吞吐高 25.3%，TTFT/TPOT 低 28.0%/21.1% | 2×A40 一个 NV4 配对。前置：RoPE 修正与数值验收、关闭诊断的计时入口、vLLM-O 和 KT 基线、Bcfg 64/128/256 请求清单、更大的 host-KV | 部分：首点（Bcfg=32、输出 4096）H/B 正确性资格通过，正式重复 0 次 |
| **图 6(b)，Exp#4** | 每个解码步的平均前向时间、平均 Bact，横轴 Bcfg | 输出 4096；FluxMoE、FluxMoE-H、vLLM-O（无 vLLM）。FluxMoE 对比 vLLM-O：Bact 7.8×，前向时间 +0.8%；对比 FluxMoE-H：前向时间 −20.0% | 与图 5 的 Bcfg 扫描共用运行。每步取各 rank 根 forward CUDA event 的最大值再平均；Bact 来自两 rank 对账后的真实 InputBatch | 未开始，口径已在 [首点方案](../results/codex-four-thread-handoff.TbHV6Yvp/work/04-figures/REPRESENTATIVE_POINT.md) 中定义 |
| **表 3，Exp#3（Mixtral 行）** | 每层每卡解码体积、存储节省、平均/P99 延迟、吞吐 | 单张 L40S；每层每卡 1.23 GB（GPU 后端的 87.5%）；节省 32.4%；3.07±0.14 ms，P99 3.13±0.14 ms；401.9±0.8 GB/s。共 32 层 × 2 卡 = 64 个组合，gate/up 与 down 在两个 stream 上并发 | 单张 A40；吞吐 = 未压缩输出字节 / 完整解码窗口；保留原 shape、dtype、码分布和双 stream | 部分：2/64 个组合；K1 单投影 139.8 GB/s，不是表 3 协议 |
| **表 4 + 表 5，Exp#4** | 每层物化分解：CPU 卸载、解压、E2E 物化、无 stall 窗口、平均 stall（ms） | 输出 4096；表 4 为 Bcfg 32/256，表 5 为 64/128；FluxMoE 与 FluxMoE-H（H 分 CPU 层 C 和 GPU 层 D）。Bcfg=32 时 FluxMoE：15.37 / 3.12 / 28.13 / 21.19 / 7.72；H：126.10 / 3.50 / C 200.74、D 3.51 / C 100.30、D 13.46 / 13.05。Bcfg=256 时 FluxMoE 的 stall 占窗口 33.9%，H 为 51.2%，stall 时长减少 43.3%。Bcfg 64/128 时减少 39.8–44.8% | 在加载流和推理流上为每层打 CUDA event，覆盖全部 rank 和 stream；与无 profiler 的性能运行分开采集 | 未开始 |
| 图 7 + 表 6，Exp#5 | 迭代→吞吐曲线：上行扫 CI，下行扫 AG；表 6 为释放的 down tensor 数和字节数 | Bcfg=32，输出 4096；对照为全部 routed 专家压缩驻留。CI ∈ {8,16,32,64}（AG=0.25），AG ∈ {0.25,0.5,0.75,1}（CI=16）。默认设置释放 24 个 down tensor，1.9 GB，为压缩 down 存储的 9.4%、压缩 routed 存储的 3.1%；AG 从 0.25 增到 1 时比例从 9.4% 升到 28.1%；CI 从 8 增到 64 时释放量减少 66.7% | 前置：真实 GPU 后端接入 planner 与安全回收，只做 down 阶段。2×A40 全压缩对照的静态占用为 35.23 GB/卡，余量 8.24 GB/卡 | 未开始；CPU 模拟 planner 仅验证过控制逻辑 |
| 图 8，Exp#6 | 迭代→吞吐曲线，有/无 planner；聚合吞吐 | Bcfg=256，输出 4096，41 次 engine launch 的多 launch 模拟。Mixtral 上 planner 迭代数更多、Bact 更小，但每步吞吐更高，聚合吞吐 +42.7% | 前置：planner 加多 launch 驱动（拼接已有输出重建 KV）；同时保存被排除的重建 prefill 和 relaunch 开销 | 未开始 |
| 图 9，Exp#7 | E2E 延迟 CDF（对数横轴），并发上限 256/512 | 10 req/s Poisson 到达，10 个预热 + 1000 个测量请求，ShareGPT 原始回复长度；FluxMoE、FluxMoE-H、vLLM-O、KT（无 vLLM）。FluxMoE 比 FluxMoE-H 的 P50 低 18.7%（并发 256）和 24.1%（并发 512） | 前置：在线服务路径（`vllm serve` 加载本项目 loader）和 Poisson 驱动；保存到达轨迹和逐请求时戳 | 未开始 |
| 图 14，附录 B | 同图 9，请求率 5/20/40 req/s | 5 req/s 时 FluxMoE 比 H 的 P95 低 27.5%/22.4%（256/512）；并发 256 时，P50 在 20 req/s 低 26.8%，在 40 req/s 低 25.5% | 同图 9。A40 更慢，同一 req/s 对应的负载位置不同，必须报告利用率、队列、Bact 以及是否饱和 | 未开始 |
| 图 12，附录 B | FluxMoE 对 FluxMoE-TP 的离线吞吐、TTFT、TPOT | TP2。三模型合计：吞吐差 ≤3.9%，TPOT 差 ≤2.6%，TTFT 最多低 14.9% | 前置：TP-only 变体；FluxMoE 侧复用图 5 的运行 | 未开始 |
| 图 13，附录 B | FluxMoE 对 FluxMoE-TP 的在线 CDF | 10 req/s；FluxMoE-TP 的 P50 低 13.6%（14.4%），P95 低 12.0%（6.7%），并发 256（512） | 图 12 的前置项加上图 9 的在线驱动 | 未开始 |
| 图 11，附录 A（Mixtral 面板） | exponent 符号 0…255、mantissa 符号 0…127 的分布 | 第 0 层预训练 BF16 routed-expert 权重 | 只需 CPU | **已完成**：[figure11.png](../results/real_weight_server/mixtral-offline.FqxVN6/figures/figure11.png) · [手册](REAL_WEIGHT_EXPERIMENTS.md) |
| 表 7，Exp#8 | WikiText-2 困惑度，vLLM 对 FluxMoE | WikiText-2-Raw-v1 test；LlamaTokenizerFast；334,659 token、651 个窗口；窗口 2048、步长 512，只计新 token；batch 1；TP/EP 4/4。两侧都是 3.6976，差值 0.0000 | 4×A40。vLLM 全驻留 TP4 放得下（23.5 GB/卡）。FluxMoE 侧需要 TP4/EP4，现有 4 卡路径是 TP2/PP2/EP2，否则需标为并行度偏差 | 未开始 |

### 非论文的迁移补充实验

论文中"压缩提高 Bact，从而超过全驻留 vLLM"这一结论（图 4，Exp#1）只在 GLM 和 Qwen 上做过。只做 Mixtral 时，这类结论没有直接证据。可选补充：4×A40 上 Mixtral FluxMoE（全部 routed 专家压缩驻留）对比全驻留 vLLM，与图 4 采用相同的两组扫描。该实验必须标为"迁移实验，非论文配置"，不能放入论文图号下。

### 图例与工作量注意事项

- 图 6(b) 和图 5 中，Mixtral 没有 vLLM 柱。正式绘图应从各方法的真实结果生成图例，不能为不可部署的 vLLM 虚构柱子。
- 表 4/5 只报告 FluxMoE 和 FluxMoE-H，H 分 C/D 两类层分别报告。
- 图 7 的 Mixtral 工作量是 Bcfg=32，图 8 是 Bcfg=256，不能为了复用一次运行而混用。
- 图 14 在论文中跨两页展示，是同一个 figure 号。

## 容量与预算

静态估计来自 `python3 -m fluxmoe capacity --model mixtral --gpu a40 --gpu-gb 48.3052`，使用十进制 GB，每卡预算 43.47 GB。估计中不包含 KV、workspace、压缩索引、加载峰值和碎片，详见 [CAPACITY.md](CAPACITY.md)。

| 配置 | 静态占用（GB/卡） | 余量（GB/卡） | 用途 |
| --- | ---: | ---: | --- |
| 2×A40，12.5% CPU 专家 | 31.43 | 12.04 | 图 5/6、表 4/5、图 9/12/13/14 |
| 2×A40，全部压缩驻留 | 35.23 | 8.24 | 图 7 的固定对照 |
| 4×A40，全驻留 BF16 | 23.50 | 19.97 | 图 1/2、表 7 的 vLLM 侧、迁移补充 |
| 4×A40，全部压缩驻留 | 17.62 | 25.86 | 表 7 的 FluxMoE 侧、迁移补充 |

首点运行的实际账本（每 rank 4 GiB GPU KV）：H 两 rank 的 GPU 总账本为 35.054/35.055 GiB，B 为 33.866/36.240 GiB；CPU 专家合计 10.5 GiB（H 每 rank 5.25/5.25，B 为 7/3.5）。

**KV 算术。** Mixtral BF16 KV 为 `2 × 32 层 × 8 KV heads × 128 × 2 B = 131,072 B/token`；TP2 下每卡 65,536 B/token。按每请求约 4,148 个 token（52 输入 + 4,096 输出）计：

| Bcfg | 全部请求 KV 合计 | 说明 |
| ---: | ---: | --- |
| 32 | 约 17.4 GB | 首点用 8 GiB GPU KV + 24 GiB host-KV 即可容纳 |
| 256 | 约 139 GB（129.6 GiB） | 论文的 512 GB host-KV 在这个负载下不会被填满 |

因此 Bcfg=256 时，聚合 host-KV 设为约 140 GiB 以上即不会成为约束，与论文 512 GB 的效果等价。这是一项有记录的预算偏差，必须同时证明运行中 host-KV 从未满。论文的 512 GB 在目标机放不下：512 GB 加 11.25 GB CPU 专家，超过可用主存。

### 目标服务器资源

2026-10-31 的只读审计原件：[hardware.json](../results/server_audit_20261031/hardware.json)、[software.json](../results/server_audit_20261031/software.json)、[extra.json](../results/server_audit_20261031/extra.json)、[driver.json](../results/server_audit_20261031/driver.json)。

| 项目 | 采样结果 | 对 Mixtral 实验的影响 |
| --- | --- | --- |
| GPU | 4×A40，SM 8.6，ECC On，每卡 46,068 MiB | 每卡 90% 预算 43.47 GB |
| 互联 | 两对 NV4，对与对之间为 SYS | 2 卡实验固定使用同一 NV4 配对；4 卡实验会跨 SYS |
| 设备编号 | 11/14 那次 boot 后，A40 的索引变为 1–4 | 设备一律按 UUID 和 PCI 地址固定，不能按索引 |
| RAM | 总量约 541 GB；8 个 NUMA 节点；11/11 只读记录可用 446.55 GiB | 两卡长跑实测累计 RSS 峰值约 143 GiB（host-KV 24 GiB 时）；共享映射可能被重复计入 |
| 磁盘 | 11/11 证据盘可用 196.40 GiB；`/data` 当时不可写 | Mixtral 原始分片和容器已就位，只做 Mixtral 时不存在磁盘约束 |
| 共享使用 | 其他用户的进程会间歇占用 GPU（11/08、11/11 两次准入因此失败） | 正式运行前必须实时复查空闲显存和计算进程 |

只读核验记录：[11/11 资源](../results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/checks/b1-acceptance-readonly.BKof0NZs/stdout.json)。

## 真实数据与交付图表规则

后续必须先完成目标平台的正确性收口，冻结可比较的配置，再采集正式性能。当前诊断 hooks 含 CPU 读回和同步，已有的引擎耗时不能作为性能柱图。

每个测量点至少保存：`experiment_id/run_id/figure_ref`；源码及依赖版本；checkpoint、tokenizer、数据 revision 与请求清单；模型 dtype；GPU UUID、每 rank 预算及 TP/EP；方法放置；host 专家与 KV 预算；原始计时与 token/请求计数；状态及失败原因。在线实验保存到达轨迹和逐请求时戳；物化实验保存各 rank/stream 的事件依赖与完成窗口；planner 保存逐步 Bact、驻留/释放字节、CI/AG 和 launch 边界。设备显存区分 allocated、reserved 和驱动占用，并计入加载峰值。

| 产物 | 可接受的数据来源 | 不可接受的替代 |
| --- | --- | --- |
| 容量柱图 | checkpoint/config/请求 token 算术，加上明确标注的设备实测运行态 | 把 `fits_static_budget` 当成实测可部署；把未知运行态当 0 而不说明 |
| 性能柱图/曲线/CDF | 关闭诊断 hooks 的正式真实运行和独立重复，保存原值、失败和计时边界 | CPU 模拟、理论缩放、把论文数字当作本项目测量、已有诊断运行的耗时 |
| 物化分解与时间线 | 覆盖全部相关 stream 和最慢 rank 的真实 CUDA event，以及暴露的 stall；另测无 profiler 的 E2E | 拿单条 stream 的 event 差当作全部操作完成；只展示重叠就宣称吞吐收益 |
| planner 曲线 | 真实 GPU 权重后备的变更和控制状态；多 launch 模拟明确标注排除项 | 把 CPU 模拟当 GPU 性能；把 41 次 launch 称作单引擎动态 KV 扩容 |
| 质量表 | 固定预训练 checkpoint，同 tokenizer、同评分窗的真实损失累积 | 用随机 tiny 输出相同或解压逐位正确来声称原模型困惑度已复现 |

所有最终图表都要附上对应的论文 figure/table 号、真实硬件/模型/版本、运行次数、误差条含义、原始数据路径和重新生成的入口。未运行、OOM、超时、功能不支持和无收益的点一律保留；OOM 不填 0 吞吐参与加速比计算。论文原值作为参照时单独标为 paper-reported。

## 当前结论

Mixtral 两卡 TP2/PP1/EP2 的加载、分页、host-KV 换出恢复和长负载生成已经在 A40 上取得正确性证据，两臂输出逐位一致。与独立参考的剩余偏差已定位到 RoPE 缓存构建设备，修正待执行。图 11 的 Mixtral 面板已完成，其余图表都还没有正式数据；vLLM-O、KTransformers、FluxMoE-TP、真实 GPU planner 和在线驱动都尚未实现。

## 范围外

以下论文内容在 2026-11-14 的范围调整后不再复现：图 4（只有 GLM/Qwen）、图 10（只有 GLM/Qwen，而且 Mixtral 每层 8 个专家、top-2 的情况下统计没有意义）、图 11 的 Qwen 面板，以及其余图表中 GLM 和 Qwen 的子图或表行。相关的分析和容量估算见 [三模型版本](PAPER_EXPERIMENT_MAP.three-models-20261031.md)。
