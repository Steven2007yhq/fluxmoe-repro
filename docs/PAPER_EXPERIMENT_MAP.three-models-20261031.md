# FluxMoE v3：论文图表、实验条件与当前复现边界

本文件的论文图表核对于 2026-10-31 在本地完成；服务器资源引用本轮 SSH 只读审计记录。未执行 benchmark、未下载检查点。下列数值中的论文设置和容量估算不是本项目性能结果。

## 依据与当前证据

- 论文：`fluxmoe.pdf`，arXiv:2604.02715v3，2026-09-10。[论文页面](https://arxiv.org/abs/2604.02715v3)
- 本地 PDF SHA256：`1ab3be9721c1e501f7c2171af919722b5271e284089bf76b4f39f00bf89975c3`。与 `/run/media/steve/Lenovo/papers/老师让读的/fluxmoe.pdf` 相同。
- 当前容量公式与限制：[CAPACITY.md](CAPACITY.md)。当前引擎限制：[VLLM_ENGINE.md](VLLM_ENGINE.md)。
- `results/engine_validation/comparison.json` 为 `passed`：单张 RTX 4060、随机 tiny Mixtral、真实 eager `LLM.generate`；11 请求、176 生成 token 与 176/200 个生成/prompt logprob 一致，最大 logprob 误差 0；另有每侧 3 请求、48 token 的移除 KV hooks 检查。
- 这些证据支持小模型加载、分页与数值正确性。tiny 不是预训练模型；其 VMM 物理分配 8 MiB 大于逻辑专家权重 1.5 MiB，不能用它证明节省显存或论文吞吐。
- 审计时源码 `fluxmoe/integrations/engine.py` 先调用 `DefaultModelLoader.load_model` 完成完整 CUDA 权重加载，再创建 pager/替换 VMM storage。因此**尚未具备超显存检查点的初次加载能力**。
- 同一源码显式拒绝 TP/PP/DP/EP、多 rank、KV offload 和非 Mixtral 架构；实际 GPU 路径尚未接入在线驻留调整。当前项目使用 vLLM 0.29，论文使用 0.23.0。不能只按下文静态预算启动原论文多卡矩阵。

## 共同实验设置

### 模型与硬件

`P/E` 为表 1 取整的总参数/routed-expert 参数；共享专家属于常驻非 routed 部分。`N/L` 为 MoE 层数/每层 routed 专家数。

| 代号 | 模型 | P/E | N/L | 主实验部署 | 非 KT 默认 TP/EP | 特例 |
| --- | --- | --- | --- | --- | --- | --- |
| G | GLM-4.5 | 355B/338B | 89/160 | 8×H20 | 8/8 | 表 7 同为 8/8 |
| Q | Qwen3-Next-80B-A3B-Instruct | 80B/77B | 48/512 | 4×L40S | 4/4 | 含共享专家及混合 attention/state |
| M | Mixtral-8×7B-Instruct | 47B/45B | 32/8 | 2×L40S | 2/2 | 图 1–2 与表 7 使用 4×L40S；表 7 为 4/4 |

论文 §6.1：

| 测试床 | GPU 与驱动可见容量 | CPU/主存 | 每卡 H2D 理论峰值 |
| --- | --- | --- | --- |
| H20 节点 | 8×H20；96GB 标称、102.6GB `nvidia-smi` | 双路 Xeon Platinum 8468，共 96 物理核；2TB RAM | PCIe Gen5，64 GB/s |
| L40S 节点 | 4×L40S；48GB 标称、48.3GB `nvidia-smi` | 双路 Xeon Gold 5318Y，共 48 物理核；2TB RAM | PCIe Gen4，32 GB/s |

每个系统默认 90% GPU-memory budget。所有系统有 **512GB aggregate host-DRAM KV tier**；使用 CPU 专家后备的系统统一专家预算为 routed 权重的 **12.5%**。KTransformers 使用 TP、无 EP，48/96 个推理线程，对应 L40S/H20 的物理核数；两个 NUMA 线程池。论文没有给出足够的驱动、CUDA、完整软件 commits、请求抽样种子和 GPU/NUMA 拓扑细节，需在目标环境另行固定并标为差异。

### 系统与放置

| 名称 | 论文含义与公平条件 |
| --- | --- |
| vLLM | 全部权重常驻 GPU；2×L40S 上 M 不可部署，不能填入 0 吞吐算加速比 |
| vLLM-O | 后部 MoE 层按预算放 CPU，前部未压缩常驻；逐层预取；须重建论文控制，而非任选当前版本一个近似参数 |
| FluxMoE-H | 前部层 GPU 压缩、后部层 CPU 后备；整层只能用一个后端 |
| KTransformers | CPU/GPU 专家计算，TP-only；版本、质量、线程、NUMA、offload/KV 预算均需固定 |
| FluxMoE | G/Q 主实验全部 routed 专家压缩驻 GPU；M 固定 12.5% CPU 放置：每层一个 GPU 卸载一个 fused gate/up tensor，另一个 GPU 卸载一个 down tensor，共 64 个 tensor；不是直接用默认 down-first 选择器代替 |
| FluxMoE-TP | 同样分页机制、GPU 和驻留配置，关闭 EP，TP 分别 8/4/2；用于检查与 KT 的差异是否来自 EP |

普通离线性能使用 ShareGPT prompts，全部请求最初可用。平均 tokenized prompt 长度 G/Q/M 为 44.9/45.6/52.1，最大 1121。两组扫描：`Bcfg={32,64,128,256}`、固定输出 4096；或输出 `{1024,2048,3072,4096}`、固定 Bcfg=256。论文报告五次运行的均值及 Student-t 95% CI。重建时保存固定请求列表、token IDs、revision/seed 和每次原始记录；该请求抽样本身未由论文完整公开。

指标是 output throughput（tokens/s）、平均 TTFT、平均 TPOT；在线为逐请求 E2E latency CDF。TTFT/TPOT 包含其定义范围内的排队/调度行为，不能用 GPU forward 时间替代，也不能从总吞吐反推平均 TPOT。

## 图表对应表

下列“缩小机制”仅说明可以设计的后续移植实验，不表示已经实现、运行或等价复现。A40 图必须写清设备、精度、模型和版本差异；真正与原图同配置的结果需要原硬件条件。图 3 是设计示意，不是性能实验。

| 论文位置 | 指标/图形与横轴 | 模型、条件、比较项 | 可缩小但保留的机制/当前前置项 |
| --- | --- | --- | --- |
| 图 1，§2.1 | 堆叠容量柱状图：calculated KV、weights、other runtime；横轴 Bcfg 或输出长度，另一维分别固定 4096/256；虚线 4×L40S 预算 | M，4×L40S，全驻留 vLLM；aggregate 权重 94GB、总预算 173.9GB | 可按真实 checkpoint config 和请求 token 数计算 KV；other runtime 若未测就标未知，不把理论和设备实测峰值混合。小模型须保持权重/运行态与预算的比例，不能拿 tiny VMM 粒度开销做原图结论 |
| 图 2，§2.2 | 吞吐、平均 active batch 的分组柱状图；横轴 Bcfg 32/64/128/256 | M，4×L40S，输出4096；同一 vLLM、只改 aggregate GPU KV 容量20/40/60GB，KV offload 启用 | 至少一个“批次已完全可调度”和一个“KV 受限”工作点；记录每 iteration Bact、抢占及实际 KV。当前真实引擎固定 KV/无 KV offload，尚不支持原实验 |
| 图 3，§4 | 系统结构示意 | forward proxy、PagedTensor、GPU压缩/CPU层级、planner | 可画实现架构并注明独立设计，不填性能数据 |
| 图 4 / Exp#1，§6.2 | 三行：throughput、AVG TTFT、AVG TPOT；每模型两列分别扫Bcfg/输出长度 | G 8×H20、Q 4×L40S；FluxMoE、vLLM、vLLM-O、KT；共同离线扫描 | 基线必须可全驻留，同时长输出/大批次使其 KV 受限；同平台比较压缩是否提高 Bact，允许小批次解压开销导致负收益。Q/G 架构及多rank集成未完成 |
| 图 5 / Exp#2，§6.2 | throughput/TTFT/TPOT 柱状图；同样两组离线扫描 | M 2×L40S；FluxMoE、FluxMoE-H、vLLM-O、KT；相同12.5% CPU专家与host-KV预算；vLLM为容量失败 | 2×A40 可作为移植目标，但先做超显存流式loader和TP/EP；保留“全驻留不够、压缩分页能放下”约束及同预算。不能用量化或CPU MoE计算替换方法侧 |
| 表 3 / Exp#3，§6.3 | 每层每卡BF16输出体积、存储节省%、平均/P99解码延迟、吞吐；可另绘分组柱状图 | G/H20、Q和M/L40S；每模型 N×G 个层/rank shard；gate/up与down两个stream并发；M只解码GPU后端87.5%体积 | 单卡即可测单个真实预训练专家分片；保留原shape/dtype/码分布、双stream与稳态输出空间；吞吐=未压缩输出bytes/完成解码窗口。现有smoke含初始化/核验或诊断开销，不能作约400GB/s复现 |
| 图 6＋表 4 / Exp#4，§6.3；表 5补B64/128 | 图6：平均每decode iteration forward时间、平均Bact，横轴Bcfg；表4/5：H2D、解码、E2E物化、无stall窗口、暴露stall，ms | G、M；图6共享图例FluxMoE/FluxMoE-H/vLLM/vLLM-O，M的vLLM不可部署；输出4096；表4 B32/256，表5 B64/128；物化表仅G的FluxMoE和M的FluxMoE/H，H分别报CPU层C与GPU层D | 可在小型真实MoE上比较预取开关和两后端放置，保持相同compressed/CPU/双缓冲/KV总预算；须证明RAW/WAR、双后端并发，并量化暴露stall及forward争用。只画重叠时间线不足以证明收益 |
| 图 7＋表 6 / Exp#5，§6.4 | iteration→throughput曲线；上行扫CI、下行扫AG；表6记录释放down tensor数/bytes及分母 | G/Q Bcfg256，M Bcfg32，输出4096；all-compressed固定对照；CI={8,16,32,64}且AG=.25；AG={.25,.5,.75,1}且CI16；仅down阶段，无gate/up释放 | 必须在真实GPU后端接planner与安全回收，保留有效内存压力和负载时间跨度；真实KV固定时只能声称专家后备变化，不能声称同launch扩大KV池。现有CPU模拟planner仅支持控制逻辑测试 |
| 图 8 / Exp#6，§6.4 | iteration→throughput曲线；w/与w/o planner；另报告aggregate throughput | G/Q/M主部署；Bcfg256、输出4096；41次engine launch，后续prompt接已有输出重建KV；前40次每次至少256×100新token，末次完成余量；固定对照用§6.1配置 | 这是**多launch模拟**，不是在线动态KV allocator。缩小时保留增长KV与驻留转换、记录launch边界；原统计只算首轮prefill+所有decode，排除重建prefill和relaunch，必须同时保存这些排除量，可另报实际作业墙钟。当前无真实GPU planner，未实现 |
| 图 9 / Exp#7，§6.5 | E2E latency CDF，横轴秒（对数刻度）；上下行并发上限256/512 | G/Q/M主部署；10 req/s，Poisson到达；10预热+1000测量请求，ShareGPT原始response长度；FluxMoE/H、vLLM/O、KT，M无vLLM | 可先在小模型选择一个未饱和与一个饱和负载，保存相同到达轨迹、每请求各时间和完成/失败；不能用短同步`generate`诊断代替online服务。当前HTTP/online实验驱动未覆盖 |
| 图10，附录A | (a)跨层/iteration active-expert均值±std柱图；(b)层×Bcfg active-expert热力图 | 全驻留vLLM，G和Q，Bcfg32/64/128/256，§6.1模型/prompt；仅保留scheduled序列数≥该run最大值90%的decode iterations；shared expert不计 | 统计实际被调度token的unique routed专家；不得把Bcfg当Bact或用独立均匀随机路由替代预训练路由。缩小模型只能检验路由计数机制 |
| 图11，附录A | 两组直方/频率曲线：exponent符号0…255、mantissa符号0…127 | M第0层、Q第20层的预训练BF16 routed-expert权重；代表层展示，各层检查分布 | CPU读取真实层权重即可，不需完整推理或多卡；保留真实BF16 checkpoint与专家聚合方式。随机tiny的熵不能代表预训练权重 |
| 图12，附录B | 离线throughput/TTFT/TPOT柱状图，同图4/5两组扫描 | G/Q/M；FluxMoE（TP+EP）vs FluxMoE-TP；GPU、驻留、请求同原对照；TP8/4/2 | 必须实现并验证至少2 ranks、TP-only与EP的合法分片/collective；单rank无法检验“收益不来自EP”的主张 |
| 图13，附录B | E2E latency CDF，上下并发256/512 | G/Q/M；FluxMoE vs FluxMoE-TP；复用Exp#7 10req/s条件 | 同图12的多rank前置项＋同图9的在线请求驱动；不是单rank开关 |
| 图14，附录B | E2E latency CDF；分别5/20/40req/s，上下并发256/512 | 同图9的模型、基线、请求数和原始response长度 | 必须扫描至少轻/重负载；不能把固定req/s迁到较慢A40后仍称相同负载位置，须报告利用/队列/Bact及是否饱和 |
| 表7 / Exp#8，附录C | WikiText-2 perplexity，vLLM vs FluxMoE与绝对差；可补同组柱状图，但原始数值表保留 | G TP/EP8/8、Q4/4、M4/4；完整WikiText-2-Raw-v1 test；按原生tokenizer双换行拼接，不加BOS/EOS；window2048、stride512，重叠仅作上下文，每窗只计新token；batch1，无跨窗KV/prefix复用 | 小预训练MoE可做同协议移植质量验证，但不能替代三个原模型；随机tiny只证明一致性。两侧checkpoint/tokenizer/并行度必须一致，M质量实验不是2卡受限配置 |

### 需要特别确认的图例

图6的共享图例列出 FluxMoE、FluxMoE-H、vLLM、vLLM-O；G和M子图按可行方法取其子集。表4/5只报告 G 的 FluxMoE 和 M 的 FluxMoE/FluxMoE-H 物化数据。正式绘图应从各方法真实结果生成图例，不为不可部署的M-vLLM虚构柱子。

图7的M工作量为Bcfg32，而图8的M为Bcfg256，不能为了复用一次运行而混淆。图14在论文跨两页展示同一个figure号。

## 容量与能做的范围

本段复用 [CAPACITY.md](CAPACITY.md) 的十进制GB、按rank理想均分公式。它没有建模压缩索引、额外副本、初始化峰值、实际TP复制、激活、通信、CUDA Graph和碎片。每卡预算为驱动可见容量×预算比例，不是多卡统一池。

| 目标 | 参数/专家放置与双层缓冲的静态下界 | 可执行范围的条件 |
| --- | --- | --- |
| 2×A40上的M | 约31.43GB/卡（12.5% CPU专家），43.2GB规划预算下尚11.77GB/卡供KV/workspace | 需要先补超显存loader、合法TP/EP及rank间正确性。完整BF16全驻留约47GB/卡，大于90%预算；现有单rank loader不能依赖该31.43GB启动 |
| 4×A40上的M | 全驻留约23.5GB/卡；分页全部GPU压缩约17.62GB/卡，12.5%CPU时15.72GB/卡 | 可选全驻留质量/性能对照；仍需真实分片/峰值验证 |
| 4×A40上的Q | 全部 routed 专家GPU压缩约29.25GB/卡，尚未加KV/state/workspace | 需Q架构与多rank集成；2个KV heads在TP4会复制，不能普通KV总量除4；linear-state与hybrid padding另计 |
| 2×A40上的Q | 12.5%CPU后约51.96GB/卡，未计KV即超预算 | 需大幅加CPU offload，属于新条件；仅碰静态容量线至少约29.3%CPU，若每卡另留8GB运行态则约44.5% |
| 2/4×A40上的G | 全压缩也约252.75/126.37GB/卡；只留非专家+双层缓冲约24.60/12.30GB/卡 | 即使静态全CPU大部分专家能放，需约91.8%/72.9%CPU专家才碰容量线，不是原8×H20实验，且主存/PCIe流量巨大 |

M的BF16 KV为全模型`2×32层×8 KV heads×128 head_dim×2 bytes=131072 bytes/token`。TP2在KV heads均分时每卡65536 bytes/token。B16、约52 prompt+4096 output tokens需约4.35GB/卡KV；B32约8.70GB/卡；B256约69.6GB/卡。31.43GB权重/缓冲之后，B32只余约3.07GB/卡临时空间，不能保证不OOM。Bcfg大而Bact受限/有host-KV是论文机制之一，不等于所有请求全部KV同时常驻。

原先按用户描述采用“A40仅PCIe”的规划假设；下面的服务器只读审计已推翻该假设。原L40S无NVLink，A40的具体拓扑必须如实写入移植实验。理论FLOPS、带宽比例和GPU总显存均不能换算为论文吞吐倍数。

### 目标服务器的实际资源（根任务只读SSH审计转交）

本节信息由根任务的2026-10-31只读SSH检查提供；本文件作者未另连服务器、未启动远端任务。原始记录为 [hardware.json](../results/server_audit_20261031/hardware.json)、[software.json](../results/server_audit_20261031/software.json)、[extra.json](../results/server_audit_20261031/extra.json)、[driver.json](../results/server_audit_20261031/driver.json)，同目录保留对应`read_only_*_audit.py`。采样值不是本地机器或后续运行的资源保证。

| 项目 | 采样结果 | 对实验的影响 |
| --- | --- | --- |
| GPU | 4×A40，SM8.6、ECC On；各卡total46068MiB/free45477MiB；无计算任务 | 每卡总量48.3052GB，90%预算43.4746GB；必须逐卡留KV和workspace |
| 互联 | 0↔1与2↔3均为NV4，其余GPU对为SYS | 2卡优先选择同NVLink配对；4卡涉及跨SYS路径。不能写成无NVLink，也不能写成4卡全互联NVLink |
| Driver能力查询 | 四卡VMM支持=1、async engine count=3、所有GPU pair peer access=1 | 满足继续核验VMM和异步路径的硬件前置项；不是kernel正确性、H2D/peer有效带宽或重叠收益实测 |
| RAM | total540797251584B、available528231120896B（约503.7/492.0GiB）；8个NUMA节点，仅0/3/4/7有内存；无numactl | CPU专家后备、host-KV、加载恢复副本及系统均占主存；需明确NUMA绑定策略，不能把总量当单路有效带宽 |
| 软件 | 驱动595.80；`/usr/local/cuda`有完整nvcc13.2.78但PATH未设；当前用户无conda/torch/vLLM运行环境；SGLang仅源码，要求Python≥3.11，system为3.10 | 尚无可复跑的远端框架环境；存在CUDA目录不等于vLLM/pager可运行。本轮未安装或构建 |
| 磁盘 | `/data`机械HDD约21.74TiB空闲，但审计用户`bks-agent`无写权限；root SSD约652GiB空闲 | `/data`容量够但当前不能写；SSD约700GB，容不下三模型全部原始权重，连710GB GLM原始checkpoint亦超当前空闲。M/Q按单模型轮换有容量可能，仍需确认目标目录可写及实际缓存/构建开销 |

用实测GPU容量替换48GB规划值后，2卡M的31.43GB静态项余约**12.05GB/卡**；4卡M全驻留余约19.98GB/卡；4卡Q全专家压缩余约14.23GB/卡。它们都没有包含KV/state/workspace，也不改变当前超显存loader和多rank功能缺失。4卡Q是容量上有机会的后续目标，并非当前实现已经支持。4卡G仍不能按原条件部署。

按本文件十进制单位，512GB host-KV＋M的11.25GB CPU专家合计523.25GB，占当前available后仅余**4.98GB**；加载/CPU恢复副本、索引、进程和pinned管理尚未计入，不能认定足够。Q的531.25GB和G的596.5GB静态项已超过当前available，G还超过整机总RAM。若框架参数单位为GiB，填写`512`会请求549.76GB host-KV，单此项就超过540.80GB整机RAM。较小host-KV预算可另做移植实验，但必须标明偏离论文512GB统一预算。

### 全论文硬件与主存/磁盘

完整原条件至少需要可分时使用的8×H20节点和4×L40S节点。按论文驱动可见容量与90%预算，全驻留G仅权重要求`ceil(710/(102.6×.9))=8`卡，Q为`ceil(160/(48.3×.9))=4`卡，尚未给运行态额外保证。换更大显存/不同架构是迁移实验；8×80GB GPU放不下710GB的同精度全驻留G基线。

| 模型 | 12.5% CPU routed-expert BF16 | 加512GB host-KV后的静态主存项 |
| --- | ---: | ---: |
| G | 84.5GB | 596.5GB |
| Q | 19.25GB | 531.25GB |
| M | 11.25GB | 523.25GB |

这些不是整机RAM要求。加载、暂存/压缩副本、pinned管理、进程和OS另计。若710GB原始G权重和512GB host-KV同时存在，仅两项约1.222TB；若分阶段分配，取阶段峰值而不是机械相加。论文没有给加载峰值，不能据约600GB静态项保证768GB/1TB足够；2TB/节点是已报告实测配置。

原始BF16三模型约710+160+94=964GB；完整压缩checkpoint约490.3+110.6+64.8=665.7GB，两套合计约1.63TB，未含依赖、镜像、缓存和日志。仅单个G的原始+完整压缩已约1.20TB。可顺序轮换降低总空间。v3未披露磁盘规格。

## 真实数据与交付图表规则

后续必须先完成目标平台最小正确性，冻结可比较配置，再采正式性能。当前诊断hooks含CPU读取/同步，已有engine耗时不能作为性能柱图。

每一测量点至少保存：`experiment_id/run_id/figure_ref`，源码及依赖版本、checkpoint/tokenizer/data revision与请求列表、模型dtype、GPU型号/每rank预算及TP/EP、方法放置、host专家和KV预算、原始计时与token/请求计数、状态及失败原因。在线保存到达轨迹与逐请求时戳；物化保存各rank/stream的事件依赖和完成窗口；planner保存逐iteration Bact、驻留/释放bytes、CI/AG及launch边界。设备显存同时区分allocated/reserved/driver占用，并计加载峰值。

| 产物 | 可接受的数据来源 | 不可接受的替代 |
| --- | --- | --- |
| 容量柱图 | checkpoint/config/request-token算术＋明确标注的实际设备运行态 | 把`fits_static_budget`当实测可部署；把未知运行态设0不说明 |
| 性能柱图/曲线/CDF | 无诊断hooks的正式真实运行及独立重复，保存原值、失败和计时边界 | CPU模拟、理论缩放、论文数字冒充本项目测量、已有tiny诊断耗时 |
| 物化机制柱图/时间线 | 真实CUDA event覆盖全部相关stream、最慢rank与暴露stall；另测无profiler E2E | 一条stream的event差当全部操作完成；只展示重叠就宣称吞吐收益 |
| planner曲线 | 真实GPU权重后备变更和控制状态；多launch emulation明确标注排除项 | 将CPU模拟当GPU性能；将41次launch称单engine动态KV resize |
| 质量表/柱图 | 固定预训练checkpoint＋同tokenizer/同评分窗的真实损失累积 | 随机tiny文本相同或解压逐bit正确就声称原模型perplexity已复现 |

所有最终图附对应论文figure/table号、真实硬件/模型/版本、运行次数、误差条含义、原始数据路径和重生成入口。未运行、OOM、超时、功能未支持与无收益点均保留；OOM不填0吞吐参与加速比。论文原值若作为参照，单独标为paper-reported，不与本项目测量混成同一来源。

本轮结论：图表映射与资源条件已明确；已有成果是tiny correctness。原检查点的超显存加载、多rank、原基线、真实GPU planner与正式性能矩阵仍未完成。本文件不把这些规划项标为实验成功。
