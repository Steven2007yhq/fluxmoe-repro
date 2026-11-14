# A40 packed 与双 stream：下一轮最小改动及对照

状态：**代码审查与实验方案；开关尚未实现，尚未编译或执行本轮实验。**
历史源码包、运行源码、二进制及所有原始结果保持冻结。服务器操作全部由用户手动执行。

## 依据与待验证主张

已核验输入：`a40-mechanism.sn3po49d`（无 profiler）与 `a40-nsys.e5739e3k`（时间线）。
本轮问题是：在同一 A40、真实 Mixtral 第0层 expert0/1 gate_up、M=256、chunk=256、两个 BF16 槽及相同工作量下，降低 packed 的每 CTA 资源成本、提高 compute stream 优先级，是否缩短完整解码/GEMM窗口？两种因素的效果是否能够分开识别？

事实：packed 单投影约130.03GB/s；M256无profiler串行19.0277ms、双stream18.8784ms，仅降低0.785%。NSYS每窗真实交叠约1.241ms，解码总时间增加0.610ms、GEMM增加0.374ms，首末kernel跨度净缩短0.251ms。GEMM已知前驱后最大间隙约1.673ms。

这些事实支持检查资源与调度，尚不能证明 DRAM、bank conflict、寄存器或某种调度规则是唯一瓶颈。kernel级GEMM增时约8.3%，不能套用旧事件包络的2.60倍。

## 代码审查发现

| 位置 | 已确认事实 | 含义 |
|---|---|---|
| `native/huffman_decode.cu:18`、`:132`、`:553` | block固定128，shared tile和launch bounds由此决定 | 改64线程时必须同时改模板、tile维度、grid和边界检查，不能仅改launch的block参数 |
| `native/huffman_decode.cu:141` | 普通prefix表4KiB + bulk表8KiB + packed tile8704bytes | packed每CTA静态shared约20992bytes，与NSYS记录一致 |
| `native/huffman_decode.cu:212`、`:215` | Bulk热路径读bulk表；普通表在widths=0时被next_exponent回退读取 | 可将这一回退改读已经存在的global_table，去除4KiB shared表及其拷贝；回退频度尚未量化 |
| `native/huffman_decode.cu:487` | source.lookup已上传且生命周期覆盖异步解码 | 不需要增加global表、改变压缩格式或额外H2D |
| `results/decoder_review/a40_mechanism_handoff_20261031/mechanism_bench.py:72` | load/compute/control均为默认优先级 | compute高优先级是一个小且可单独测的因素 |
| 同文件`:103`、`:110`、`:119` | 有slot WAR、weight RAW和两stream结束join | 保留这些依赖，不通过删等待制造加速 |
| `fluxmoe/cuda_pager.py:127`、`:235`、`:252` | pager创建默认loader stream；consumer用vLLM当前stream | 微基准的高compute优先级不能直接等同于引擎已启用；引擎接入需单独审查 |
| `native/huffman_decode.cu:540`附近 | 当前native明确拒绝CUDA Graph capture | 不把Graph作为本轮的低成本开关 |
| `fluxmoe/cuda_codec.py:139`、`:191` | builder可指定独立output_dir；自定义.so的metadata并非一律强制存在 | 新实验必须独立目录并显式要求/核验metadata，不绕过source/artifact哈希检查 |

## 第一优先级：两个独立开关

### P：compute优先级

只在新实验适配器增加compute_priority参数；load/control保持普通优先级。原始调度函数的循环顺序与RAW/WAR不变。

设备初始化后查询有效优先级范围，使用最高有效compute优先级；保存请求值、实际compute/load/control优先级和stream ID。优先级值越小越高。若实际compute与load优先级相同，记录该case不可用，不声称开启了处理因素。

该开关只影响调度偏好，不抢占已运行的低优先级工作，未必消除下一次decode已开始造成的等待。P无收益也不能单独否定调度竞争。

使用同一套两个BF16槽和一套输入/输出，预建普通/高优先级compute stream，在case同步边界切换句柄。禁止同时保留普通/高优先级两个完整pipeline导致额外驻留缓冲。所有case开测前，两条compute stream上的cuBLAS workspace、CUDA事件和source reader路径均先初始化、预热；额外workspace/框架allocated及reserved也计入内存记录，保持各case的已分配集合一致。

### F：packed回退表改为global读取

只对packed加入默认关闭的编译期开关，保持tree/lookup/tiled/bulk/buffered/vectorized的原行为。实现方式示意（不是现有CLI）：

```cpp
// 给既有kernel增加默认参数，旧调用仍为128线程、shared回退表。
template <bool Tiled, bool Bulk = false, bool Buffered = false,
          bool VectorIO = false, unsigned Threads = BLOCK_THREADS,
          bool GlobalFallback = false>
__global__ __launch_bounds__(Threads) void decode_prefix_chunks(...);

// 具体实现需同时设置合法组合static_assert与warp数量。
__shared__ uint32_t table[GlobalFallback ? 1 : LOOK_SIZE];
if constexpr (!GlobalFallback) {
    // 保留原table拷贝
}
// bulk_table热路径不变；只有fallback实参选择不同。
next_exponent(..., GlobalFallback ? global_table : table, ...);
```

保持32-bit打包读写、34列tile、有效输出上限、尾部/2-byte对齐回退、255-bit最长码回退和sticky error/source poisoning语义。仅普通表从shared移至global；该global表原本就存在。编译器应去除未使用的单元素table，具体shared大小以编译/NSYS元数据为准。

预期shared声明成本从20992降至约16896bytes/CTA；这不是已测性能结果，也不保证更高实际occupancy。长码路径可能因global访问变慢，必须覆盖长码与损坏码流检查。

拟改动范围：CUDA模板/packed dispatch，`cuda_codec.py`的明确构建选项与metadata，以及新的实验适配器。不改FMBF编码器、CPU/VMM模块或vLLM执行流。两个CUDA变体都在新源码树中编译，记录完整diff及源码哈希；K0采用新源码默认选项作同轮对照，历史130.03GB/s只作背景。

## 第二小步：64线程CTA，先不扩大首轮

若第一轮之后仍需要区分block资源与调度，增加Threads=64的编译变体。检查`BLOCK_WARPS`、`__launch_bounds__`、shared tile维度、grid计算、INT_MAX/chunk数边界及末尾部分warp；所有lane仍须参加warp barrier。

| 候选 | Threads | 普通回退表 | 估算shared bytes/CTA | 第一轮执行 |
|---|---:|---|---:|---|
| K0 | 128 | shared | 20992 | 是，原语义对照 |
| K1 | 128 | 已有global | 16896 | 是 |
| K2 | 64 | shared | 16640 | 后续条件对照 |
| K3 | 64 | 已有global | 12544 | 后续组合对照 |

这些是声明字节数推算，不包含可能的编译器布局/驱动保留成本。A40的sm86最多100KiB shared/SM、64K个32-bit寄存器。已有GEMM记录222regs/thread、256threads、73728bytes shared；packed记录37regs/thread、128threads、20992bytes shared。这提示共同驻留资源紧，但**不证明实际共驻block数**，寄存器分配粒度、每block保留及调度均需考虑。64线程也可能因表拷贝次数增加、活跃warp变少而使单解码更慢。

不在首轮强制降低maxrregcount或新增persistent grid，不猜测bank conflict已经成立；也不重做先前更慢的vectorized bit-window组合。若首轮有收益，先验证与确认，再决定是否引入K2/K3。

## 下一轮A40：先执行K0/K1，M256

所有命令及上传/编译/任务启动由用户手动执行。本文是方案，尚无新增CLI或可上传执行包；不把示意参数当现有接口。

### 1. 新目录与构建

- 新建实验源码树、每变体独立build目录和每次独立results目录；不改旧`fluxmoe-decoder-review.nPEGkFcD`源码或二进制。
- 沿用固定revision与已存在权重/编码缓存，chunk=256；不下载、不重新编码覆盖历史缓存。
- CPU/VMM可复用原产物与原编译记录，核验源/产物哈希；K0/K1 CUDA用nvcc13.2、sm86重新编译。
- 保持-O3、原runtime链接及相同其余选项，增加`-Xptxas=-v`保存寄存器/shared/spill报告；不开fast-math，不强制寄存器上限。
- 拟新增构建参数仅允许Threads=64/128及GlobalFallback布尔值；完整-D参数写入metadata和编译命令。不得通过不受控EXTRA_NVCC_FLAGS或缺失metadata加载变体。

### 2. 正确性先行

- K0/K1分别跑现有完整回归；A40历史166项只是参考，新增检查按实际发现数报告，要求零意外skip/fail。
- 单符号、所有BF16模式、2-byte输出对齐、非规则chunk、尾部guard、>10-bit和255-bit码字、错误偏移、损坏码流、失败后禁止继续使用source。
- 明确覆盖GlobalFallback开关。后续64线程时增加63/64/65个chunk等CTA边界，不能只依赖旧128线程测试。
- 真实E0/E1 gate_up逐位一致；每个完整窗口case前后全部GEMM和最终槽位逐位一致，不放宽容差。
- 变体kernel的实际路径、block尺寸、shared声明需与metadata一致。失败即停止该source与该变体性能测量。
- 若memcheck/racecheck可用，对改动相关边界另做短检查；与正式计时分开，不把旧sanitizer通过记成本轮新产物通过。

### 3. 无profiler正式筛选矩阵

单投影：真实E0 gate_up，K0/K1各3个独立进程，每进程预热10次、测30次；共180个正式样本。记录BF16输出bytes/time口径，保留单次延迟。不把`decoder_bench --runs 3`的同进程循环当三个独立进程。

完整窗口：M=256；沿用E0/E1与`0,1,1,0,0,1,1,0`序列、同一输入/输出哈希、两个BF16槽和相同两个native源。每窗8次GEMM，解码策略每窗8次decode，resident每窗0次。

| Kernel | serial：同stream普通优先级 | overlap：等优先级 | overlap：高compute优先级 |
|---|---|---|---|
| K0 | 测 | 测 | 测 |
| K1 | 测 | 测 | 测 |

加一个相同内存占用的GEMM-only控制，共7个case。每个kernel使用3个独立worker进程，各case预热5个完整窗口、测20窗；resident放在3个K0 worker中。共21个case-run、420窗、3360次GEMM、2880次计时内decode。独立worker共6个，不能把case数当进程数。

三次重复外层kernel顺序K0/K1、K1/K0、K0/K1；策略顺序作三次循环轮换，并记录resident插入位置。所有路径先初始化和预热；profile关停、没有每job全局同步。初始填充和最终排空计入共同GPU begin/end窗口，同时保存host调用窗口。

统计先每进程聚合，再报3个进程均值及样本标准差，保留全部样本/失败。测P时固定kernel；测F时固定策略。若收益约1%或接近波动，胜出组合与同轮K0再增加到5个独立进程确认，筛选和确认分别报告，不能只保留获胜样本。

### 4. NSYS只解释机制

正式样本结束后再采集K0/K1各serial、等优先级overlap、高优先级overlap，共6份短报告，每份3个8-job窗口。关闭GPU metrics、CPU采样/context-switch；明确设置`--cuda-event-trace=false`作为主时间线口径，保存实际stream优先级、kernel属性和环境白名单。

主要核验：每窗8个GEMM、0/8个decode，RAW/WAR，真实kernel交叠，GEMM/decoder寿命增长，CPU提交与GPU尾部，以及已知前驱后的剩余间隙。不把CPU StreamWaitEvent调用时长当GPU等待。

若需要进一步看event完成，可对K0与候选另开`--cuda-event-trace=true`配对诊断；先确认非零timestamp，若仍为0如实标记不可用。该模式可能增加overhead或假依赖，只能与false配对解释，不能合并正式计时或先验认为它“更准确所以无扰动”。

原来cuBLAS API采集缺失，本轮不依赖它作为验收依据；真实GEMM kernel名、launch correlation、工作量和数值检查必须完整。NCU仍按权限受限记录，不尝试提权或修改驱动。

## 选择与后续接入

首要指标是完整窗口更早完成，不是交叠面积更大、单投影GB/s更高或静态shared更低。若提高优先级使GEMM更快但decoder更慢、最终总窗口不改善，记录无收益；若更小CTA只改善并发却明显拖慢单解码，保留两套结果，不贸然改全局默认。

获胜候选先确认M32是否退化，再做真实整层bulk/packed对应pilot及默认路径的完整resident/paged vLLM正确性检查。tiny模型仅做正确性。分页引擎使用vLLM当前compute stream，不能照搬微基准“新建高优先级compute stream”后宣称已优化vLLM；应单独设计loader/consumer接入及同步验证。

不在本轮使用host sleep/poll制造空隙、不删RAW/WAR、不计时外预解码初始槽、不增第三个BF16槽、不依据固定已知序列宣称在线路由预测收益。

资源界限：初轮仅2个CUDA构建、180个单投影样本、420个完整窗口、6份短NSYS报告；不重跑旧bulk/buffered/vectorized矩阵或本机真实权重长实验。编译/加载/哈希耗时仍存在，整个新流程尚未A40试跑；实现交付时应给每步超时、PID、退出码及新结果目录。

## 依据

- 项目已核验报告：`results/decoder_review_a40/a40-nsys.e5739e3k/deliverable/report.md`。
- [CUDA stream优先级：调度提示，不抢占已运行工作](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__STREAM.html)。
- [Ampere sm86资源限制、shared保留与分配](https://docs.nvidia.com/cuda/archive/13.2.1/ampere-tuning-guide/index.html)。
- NSYS服务器2025.6.3实际help已保存：`results/decoder_review_server/nsys.0kLlAROg/a40-nsys.e5739e3k/nsys-profile-help.log:98`，包含event-trace额外overhead/假依赖说明。
- [NSYS CUDA Event Trace限制](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-event-trace)。
- PyTorch2.13已安装源码`torch/cuda/streams.py`确认priority参数与数值方向；在线stable文档已为2.14，不将其版本当本轮依赖要求。
