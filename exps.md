Steven2007yhq/fluxmoe-repro

## 历史实验（补录）

以下是 2026-10-30 至 2026-11-14 已经跑过的实验，按时间排列，从 `docs/`、`results/` 中的报告、运行回执和交付包整理。这段时间项目没有任何 git 提交，各条 commit 留空；原始记录里没有写明的种子同样留空。文中路径都相对项目根目录。

### 共同信息

**机器与环境**

- 本机：RTX 4060 Laptop 8 GB（SM 8.9，驱动 595.99.02）。用的是两个已有的 venv，不是为本项目单独建的环境：`/home/steve/torch_env`（Python 3.14.4，PyTorch 2.14.0+cu126）和 `/home/steve/vllm_env`（Python 3.14.4，vLLM 0.29.0，PyTorch 2.13.0+cu130，Transformers 5.17.0）。本机原生库用 `vllm_env` 里 NVIDIA wheel 自带的 nvcc 13.4.92 编译。快照见 `results/environment.json`。
- 服务器：主机名 `ps`，4×A40（SM 8.6，ECC 开，每卡驱动可见 46,068 MiB），GPU 0↔1、2↔3 为 NV4，两对之间为 SYS；CPU 为 AMD Threadripper PRO 5995WX（无 AVX-512），内存约 504 GiB，8 个 NUMA 节点；驱动 595.80，CUDA Toolkit 13.2（`/usr/local/cuda-13.2`，nvcc 13.2.78，cudart 13.2.75），g++ 11.4.0。四张卡的 UUID 依次为 `GPU-e81a0988-c3ff-59b8-8589-bee2898ca21e`、`GPU-52772f42-b234-dfc4-8f1d-dd535086e8ee`、`GPU-49a16f24-98bf-794a-4ad0-e62d6dff5dec`、`GPU-ea48857b-eef3-58d7-2b1d-02dc57f56d46`；11/14 的一次重启后它们的索引变为 1–4，下文的 GPU 编号指当时的索引。服务器与其他用户共享，多次准入失败是因为别人的进程占了卡。
- 服务器 Python 环境：conda 环境 `fluxmoe-steve`（`/home/bks-agent/miniconda3-steve/envs/fluxmoe-steve`），Python 3.12.16，torch 2.13.0+cu130，vLLM 0.29.0，Transformers 5.17.0，tokenizers 0.23.2，safetensors 0.8.0，numpy 2.3.5，triton 3.7.1，huggingface-hub 1.32.0，出图用 matplotlib 3.10.8。直接依赖见 `configs/requirements-a40-py312.txt`，完整包列表见 `results/server_environment/post_install_checks.json`。本项目的三个原生库（CPU 编码器、CUDA 解码器、VMM）都在服务器上用 nvcc 13.2 以 `sm_86`、`-O3` 重新编译。论文用的是 vLLM 0.23.0。

**模型与数据**

- tiny Mixtral：本地路径 `models/tiny_mixtral`。随机初始化（种子 27415）的 Mixtral 架构 fixture：4 层、每层 4 专家、hidden 128、intermediate 128、4 个 attention head / 2 个 KV head、head_dim 32、词表 256、BF16，配本地 WordLevel tokenizer。只用于检查执行链路和数值一致性。
- Mixtral：HF `mistralai/Mixtral-8x7B-Instruct-v0.1`，revision `eba92302a2861cdc0098cc54bc9f17cb2c47eb61`，19 个 BF16 分片共 93,405,713,504 字节。服务器上完整 checkpoint 在 `/home/bks-agent/fluxmoe-data/mixtral-full-offline-eba92302.2jipyb54`（10/31–11/01 补齐并离线核验 19 个分片的 SHA256），只含第 0 层所需分片的目录为 `/home/bks-agent/fluxmoe-data/mixtral-l0-eba92302`；本机的第 0 层分片在 `weights-transfer/mixtral-l0-eba92302`。
- cache512：全部 512 个 routed 专家投影（32 层 × 8 专家 × gate_up/down）预先编码好的 FMBF 容器，服务器路径 `/home/bks-agent/fluxmoe-runs/manual-paper-receiptfix.eql6hj2k/cache512`，长负载配置里记录的 `cache512_sha256` 为 `52ba964e6f94a18215bcb59f957bd47cf868f5e3ddf1c5bf855cea9ce94a0325`。
- 三请求短负载：3 条固定 prompt，经 Mixtral tokenizer 后长度为 16、22、290 token，由交付包中 `ep2_validation.py` 的 `workload()` 定义（标识 `fixed-pinned-three-prompt-v1`）；每请求 greedy 生成 8 个 token。下文多卡引擎实验除特别说明外都用这个负载。
- ShareGPT 长负载：HF 数据集 `anon8231489123/ShareGPT_Vicuna_unfiltered`，revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`，文件 `ShareGPT_V3_unfiltered_cleaned_split.json`。按选择种子字符串 `M-A40-P32-L4096-HB-ShareGPT-v1` 的 SHA256 顺序取首轮人类发言，按 token 长度分层抽出 32 条（输入 17–138 token，均值 46.69）。冻结清单为 `configs/host_kv_long/REQUEST_MANIFEST.json`（SHA256 `a4013c4615a9c8f35383c2bff4d2438d06b6319f6258ce27c9e243a2d4c923a1`）。

**方法与术语**

- FMBF：本项目自定的 BF16 无损压缩格式。指数用 canonical Huffman 编码，符号和尾数原样保存；每 256 个元素一个 chunk，索引记录各 chunk 码流的起始 bit。
- 解码器路径：tree 每线程顺序解一个 chunk，每个元素逐 bit 遍历 Huffman 树；lookup 改为 10-bit 前缀查表；tiled 在查表基础上用 warp 内 32×32 shared memory 转置，使输出连续；bulk 一次查表最多解出 4 个短码；buffered 为 bulk 加寄存器内缓存两个 32-bit 码字；packed 为 bulk 加 uint32 打包读写和 34 列 shared tile；vectorized 为 buffered 与 packed 的组合。长于 10 bit 的码统一回退到树。K0 指 packed 原实现，K1 把 packed 回退时读的普通前缀表从 shared memory 改为读已有的 global 表，每 CTA 静态 shared 从 20,992 降到 16,896 字节。11/01 起默认用 K1 packed、每 CTA 128 线程、chunk 256、所有 stream 优先级 0。
- 分页执行：每个专家投影在 GPU 上有固定的 CUDA VMM 虚拟地址，物理上每种投影只有两个缓冲块，按层轮换（双缓冲）。权重来自 GPU 上常驻的压缩源（CUDA 解码直接写入 VMM 地址）或 pinned host 内存（非阻塞 H2D）。计算 stream 读某层前等待该层加载完成事件（RAW），加载 stream 覆盖同一物理块前等待上一次读取完成事件（WAR），热路径上没有设备同步。
- 四卡拓扑 TP2/PP2/EP2：global rank 0/1 为 PP stage 0（全局层 0–15），rank 2/3 为 stage 1（层 16–31）；TP/EP 组为 [0,1]、[2,3]，TP-rank 0 持有专家 0–3，TP-rank 1 持有专家 4–7。两卡拓扑 TP2/PP1/EP2 中两个 rank 各持全部 32 层的 4 个专家。
- 驻留策略：U16、U8、U0 指每 rank 16 个本地层中以未压缩形式常驻 GPU 的层数，其余层走分页；R0 是 vLLM 原生加载的全驻留引擎。静态放置 A（对应论文 FluxMoE-H）把全局层 28–31 的全部本地专家放 CPU；静态放置 B（对应论文 FluxMoE）在每层把 TP0 本地专家 0 的 gate_up、TP1 本地专家 0 的 down 放 CPU。两种放置的 CPU 专家都是 routed 专家原始字节的 12.5%，共 10.5 GiB。
- 多卡引擎的数值门槛：attention 输出对独立参考 atol=1e-3、rtol=1e-2；完整 32,000 词表 logits 和层激活 atol=rtol=1e-5；KV 写入与历史读取按 BF16 位型逐位比较。比较基准一般是全驻留引擎在同一请求、同一位置上的结果。
- 多卡引擎上的计时都只有一到两个独立引擎、少量窗口，部分开着 profiler 或诊断采集，没有按论文协议做独立重复。

### H01 CPU 离散事件模拟

- 时间与机器：2026-10-30，本机 CPU（`/usr/bin/python3`，Python 3.14.4，只用标准库）
- 模型与数据：合成 toy MoE，权重和输入由种子生成。`configs/toy.json` 为 8 层、4 专家、hidden 16、intermediate 32、top-2；`configs/planner.json` 为 6 层、4 专家、hidden 8、intermediate 16、top-2。
- 内容与方法：在 CPU 上做真实的 BF16 编解码和小型 MoE 前向（top-k 路由、fused gate/up、SwiGLU、down、残差，Python float 累加），用离散事件模拟 H2D、解码、计算三类共享资源的排队和 RAW/WAR 依赖，每轮与常驻参考输出比较。`sweep` 比较五种放置策略（resident、compressed、hybrid、whole_layer_cpu、whole_layer_hybrid）× 预取开/关 × batch 2/4，共 20 个点，关闭驻留 planner；`planner` 单独运行开启 planner 的配置，检查控制决策和元数据版本发布。时间都是由配置里的带宽与计算时间参数推出的模拟时间。
- 超参：toy.json 为 cpu_fraction 0.125、iterations 4、remaining_batch 16、CPU 带宽 0.008 GB/s、解压带宽 0.080 GB/s、压缩读带宽 0.200 GB/s、launch 0.005 ms、专家计算 0.030 ms/token、非专家 0.030 ms、GPU 预算 2,000,000 字节、KV 8,192 字节、workspace 4,096 字节；planner.json 为 batch 2、iterations 8、remaining_batch 32、cpu_fraction 0.5、control_interval 1、adjustment_granularity 0.25。
- 种子：模型 17；第 \(i\) 轮（从 0 起）输入的种子为 \(17 + 1009\,(i+1)\)。
- 环境：本机 CPU
- commit：
- 交付包 SHA256：本机运行，无交付包。源码快照 `results/source_snapshot.tar.gz` 的 SHA256 为 `d7e5ae5ab2b65840aaa2ec77c3bc72d5f9a969d83e59265d962d20185bd5331d`。
- 结果：20 个扫描点全部通过，输出与常驻参考的误差为 0。开预取后 compressed/hybrid 的模拟等待明显减少，例如 batch 4 的 hybrid 从 5.63 ms 降到 2.15 ms。
- 记录：`results/validation/`、`results/demo/`、`results/sweep/`、`results/planner/`

### H02 本机 CUDA 原语与独立数值参考

- 时间与机器：2026-10-30，本机 RTX 4060
- 模型与数据：toy MoE 默认配置（6 层，batch 8）；解码 smoke 为 65,536 个 BF16 元素；VMM smoke 为 6 层、每层 1,031 字节。
- 内容与方法：三项独立检查。一是 PyTorch oracle，在 GPU 上用 PyTorch 矩阵运算（float64 累加）重算 toy MoE，与 Python float 参考对照。二是 CUDA Huffman 解码 smoke：CPU 完整校验 FMBF、建立 chunk 索引和解码树，上传后由 tree kernel 解码，再拷回与原数据逐位比较。三是 CUDA VMM 别名检查：用 Driver API 给 6 层各保留独立虚拟地址，按层号奇偶映射到两个物理块，检查写一层后同奇偶的层可见、地址保持稳定。另在 `FLUXMOE_TEST_CUDA=1` 下跑完整单元测试。
- 超参：解码 chunk 256；oracle 容差 1e-10
- 种子：PyTorch oracle 用 toy 模型默认种子 17；解码 smoke 的输入种子未记录
- 环境：`torch_env`；原生库构建记录在 `results/build_records/`
- commit：
- 交付包 SHA256：本机运行，无交付包。源码快照同 H01。
- 结果：oracle 最大绝对误差 1.11e-16；解码输出与输入逐位一致（首次调用 kernel 1.70 ms，含 CPU 校验与建索引的端到端 375 ms）；VMM 别名检查通过；82 项测试全部通过、无 skip。
- 记录：`results/torch_cuda_reference.json`、`results/cuda_codec_smoke.json`、`results/cuda_vmm_smoke.json`、`results/validation_cuda/`、`results/verification_summary.json`

### H03 单卡真实异步分页与 vLLM MoERunner 接入

- 时间与机器：2026-10-30，本机 RTX 4060
- 模型与数据：随机 BF16 专家权重。分页验证为 5 层、4 专家、hidden 64、intermediate 96、batch 8；vLLM smoke 为 4 层、4 专家、hidden 64、intermediate 128、top-2、batch 16。
- 内容与方法：`CudaMoEPager` 在初始化时完成 CPU 编码与索引、压缩源上传、pinned 分配和 VMM 映射；按 cpu_fraction 选出的投影走 pinned H2D，其余走 GPU 解码；每种投影各有一条解码 stream 和一条 H2D stream。每轮 `begin_iteration` 预取前两层，计算侧 `acquire(i)` 建立 RAW，`release(i)` 记录最后读取并允许预取第 \(i+2\) 层。`async_validation` 跑 5 轮真实 GPU MoE 前向，每轮与常驻权重结果比较，并检查物化权重的 BF16 位型。vLLM smoke 在 `process_weights_after_loading` 之后用 `Parameter.data = VMM 视图` 把 VMM 地址绑定到 vLLM 0.29 的 `MoERunner` 参数，forward pre/post hook 调用 acquire/release，在非默认 stream 上跑真实 `UnquantizedFusedMoEMethod → TritonExperts` 3 轮，与同一算子的全驻留结果比较。另在 `vllm_env` 的 PyTorch 2.13 下检查 VMM 零拷贝 Tensor 视图的指针、事件顺序和生命周期。
- 超参：分页验证 cpu_fraction 0.25、预取开、5 轮；vLLM smoke cpu_fraction 0.125、3 轮、TP1/EP1、eager、不编译、无 CUDA Graph
- 种子：两项均为 260402715
- 环境：分页验证用 `torch_env`，vLLM smoke 与 VMM 视图检查用 `vllm_env`
- commit：
- 交付包 SHA256：本机运行，无交付包。源码快照 `results/async_paging/source_snapshot.tar.gz` 的 SHA256 为 `d89d1cadb18ee943ad9f9ce0f4c3fe030fdbf6c5f476814022cb51c18c3d8cab`。
- 结果：分页验证 5 轮最大误差 0，50 次权重位型检查通过，VMM 指针稳定，共 125 次解码、75 次 H2D、75 次 RAW 等待、69 次 WAR 等待。vLLM smoke 3 轮与全驻留逐位一致，8 个参数地址稳定。vLLM smoke 第一次因未初始化 worker 的 `WorkspaceManager` 失败。基础回归 109 项中 108 项通过，跳过的 1 项是另在 `vllm_env` 单独运行并通过的 vLLM 集成测试（4 项）。
- 记录：`results/async_paging/`、`results/async_validation_cuda/`、`results/vllm_paging_smoke.*`、`results/vllm_paging_attempt1.*`、`results/vmm_torch_view_vllm_env.json`

### H04 tiny Mixtral 整引擎全驻留与分页对照

- 时间与机器：2026-10-30 至 10-31，本机 RTX 4060
- 模型与数据：tiny Mixtral；`fluxmoe/engine_validation.py` 内置的 11 个请求，覆盖短文本混合、长 prompt 的 chunked prefill、重复 prefix 的冷/热请求和 KV block 复用；去掉诊断 hook 后再跑 3 个请求。
- 内容与方法：注册两种 vLLM `load_format`。`fluxmoe_resident` 走 vLLM `DefaultModelLoader` 全驻留；`fluxmoe` 用同一 loader 完成 postprocess 后抽出专家权重，建 `CudaMoEPager` 并绑定 VMM。两侧在独立进程中顺序运行 `LLM.generate`，检查点、tokenizer、请求、采样相同。比较：加载后专家权重与直接从磁盘 safetensors 拼出的布局逐 tensor SHA256 对照；生成 token、停止原因、生成与 prompt logprob；第一层 attention 的 KV slot 写入逐位核对，历史读取用独立 FP32 CPU SDPA 重算；去 hook 后重放比对。
- 超参：BF16、eager、TRITON_ATTN、MoE backend triton、block size 16、KV 池固定 1 MiB、max_model_len 128、max_num_seqs 4、max_num_batched_tokens 32、chunked prefill 与 prefix caching 开、gpu_memory_utilization 0.4；temperature 0、每请求 16 token、`ignore_eos=True`、logprobs 5；分页侧 cpu_fraction 0.25（选中 12 个 down 投影走 pinned H2D）、预取开；`VLLM_USE_FLASHINFER_SAMPLER=0`
- 种子：vLLM seed 27415；tiny 模型初始化种子 27415
- 环境：`vllm_env`；`results/engine_validation/provenance.json` 记录了编译器、原生库 SHA256 和 vLLM 相关源文件 SHA256
- commit：
- 交付包 SHA256：本机运行，无交付包。源码快照 `results/engine_validation/source_snapshot.tar.gz` 的 SHA256 为 `bd495942d951e7ca91e77b3d05797a903a84a7c5c2e84733942ff53e5ed2a6d0`。
- 结果：通过。11 个请求、176 个生成 token 一致；176 个生成 logprob 与 200 个 prompt logprob 最大误差 0；每侧 454 次 KV 写入核对、165 次历史 attention 对照无失败（最大绝对误差 0.00184，在混合容差内）；去 hook 后 3 个请求 48 个 token 一致。分页侧共 107 次 iteration、2,140 次解码、1,284 次 H2D。最早的启动探测因 FlashInfer sampler JIT 找不到 nvcc 失败，此后统一关闭 FlashInfer sampler。
- 记录：`results/engine_validation/`、`results/engine_validation_attempt1/`、`results/engine_probe/`

### H05 A40 单卡 tiny Mixtral 迁移验收

- 时间与机器：2026-10-31，服务器单张 A40（`CUDA_VISIBLE_DEVICES=0`），用户手动执行
- 模型与数据：tiny Mixtral（与本机同一份 checkpoint），H04 的请求
- 内容与方法：在服务器重编译原生库后，按 `docs/A40_MIGRATION.md` 依次运行 CPU 测试、GPU 测试（解码、VMM、VMM Torch 视图、pager）、H03 的分页验证和 H04 的整引擎对照。结果拷回本机后独立重算对照。
- 超参：同 H03、H04
- 种子：同 H03、H04
- 环境：`fluxmoe-steve`；`HF_HUB_OFFLINE=1`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、`VLLM_USE_FLASHINFER_SAMPLER=0`、`CUDA_DEVICE_ORDER=PCI_BUS_ID`
- commit：
- 交付包 SHA256：
- 结果：146 项测试通过、无 skip；整引擎对照与 H04 相同（11 请求 / 176 token，logprob 最大误差 0，每侧 454 次 KV 写入与 165 次历史 attention 核对无失败）；分页验证 5 轮、50 次权重位型检查通过。服务器的原生构建记录没有拷回。
- 记录：`results/a40_server/a40-verify.IhdPv5/`

### H06 Mixtral 第 0 层真实权重：图 11 分布与表 3 pilot

- 时间与机器：2026-10-31，服务器 A40 GPU 0，用户手动执行
- 模型与数据：Mixtral（固定 revision），按 `configs/real_weight_plans/mixtral_layer0.json` 只下载第 0 层所需分片（约 4.89 GB）
- 内容与方法：`analyze` 读取第 0 层 8 个专家全部 24 个 gate/up/down 矩阵的原始 BF16 位，统计 exponent（256 个符号）和 mantissa（128 个符号）的分布，画图 11 的 Mixtral 面板。`benchmark` 对应表 3：工作单元是“一层 × 一个逻辑 EP rank”，Mixtral 按逻辑 EP2 划分。按 12.5% CPU 专家预算，rank 0 首个本地专家的 fused gate_up、rank 1 首个本地专家的 down 不参与 GPU 解码（论文没有给出专家 ID，这是本项目的假设），两 rank 合计解码原始专家体积的 87.5%。C++ 准备器在计时前完成编码、建索引和上传；gate_up、down 各用一条 CUDA stream，两者等待同一个 start event，control stream 等两路完成后记录 end，计时覆盖完整的双 stream 窗口。测量前后各对原始 BF16 逐位检查一次。两个逻辑 rank 在同一张卡上先后测量，解码器为当时唯一的 tree 路径。吞吐 = 未压缩输出字节 / 窗口时间（十进制 GB/s）。
- 超参：chunk 256；warmup 5、samples 20；覆盖 2/64 个 case（只有第 0 层）
- 种子：
- 环境：`fluxmoe-steve`
- commit：
- 交付包 SHA256：`fluxmoe-realweights-src.tar.gz`，`7b1f4f27495585de5800a64ad069b7e6fe60b36135e9709a07b1eab162957f02`（运行记录中的源码哈希与包内文件一致）
- 结果：图 11 Mixtral 面板完成，24 个矩阵共 1,409,286,144 个 BF16 值，exponent 熵约 2.554 bit、mantissa 熵约 6.973 bit。表 3 pilot：rank 0 解码 1,174,405,120 字节约 288 ms，rank 1 解码 1,291,845,632 字节约 318 ms，平均吞吐 4.07 GB/s（论文在 L40S 上为 401.9 GB/s）；压缩容器节省 33.71%，计入 GPU 上的索引和解码树后节省 32.14%。
- 记录：`results/real_weight_server/mixtral-offline.FqxVN6/`（图在 `figures/`）

### H07 本机解码器优化：tree → lookup / tiled / bulk

- 时间与机器：2026-10-31，本机 RTX 4060
- 模型与数据：本机的 Mixtral 第 0 层分片。单投影实验用第 0 层 expert 0 的 fused gate_up（[28672, 4096]，234,881,024 字节）；整层实验用 H06 的两个逻辑 rank case。
- 内容与方法：先离线复核 H06 的原始数据，在本机用同 revision 权重重算图 11 的直方图与熵。Nsight Systems 显示 tree kernel 占计时窗口约 99.8%，窗口内没有拷贝，瓶颈是解码 kernel 本身（每线程串行解 256 个元素、逐 bit 访问全局内存中的树、同一 warp 相邻线程的输出地址相隔 512 字节）。新增 lookup、tiled、bulk 三条路径，与 tree 在同一二进制中对照；另做 chunk 32 消融，并限制循环展开把 bulk 的寄存器从 197 降到 37 个/线程。
- 超参：chunk 256（消融 32），CTA 128；单投影 3 个独立进程，每路径 warmup 10、samples 30；整层双 stream 的 tree 与 bulk 各一个进程
- 种子：
- 环境：`vllm_env`（Python 3.14.4，PyTorch 2.13.0+cu130）
- commit：
- 交付包 SHA256：本机运行。本轮为 A40 准备的 `fluxmoe-decoder-opt-src.tar.gz`（`931e5bbe6b9d0dd6acd15867865387c702941ba34ad4161aa665ef467696bccb`）没有在服务器上运行，被 H08 的版本取代。
- 结果：单投影 tree 3.196 GB/s，bulk 10.473 GB/s，延迟比 3.277×；整层观测吞吐 2.837 → 10.462 GB/s，但 tree 有 2–3 秒的长尾样本。chunk 32 没有收益。163 项回归和 tiny Mixtral 引擎对照通过。
- 记录：`results/decoder_optimization/`

### H08 本机评审改进：bulk / buffered / packed / vectorized

- 时间与机器：2026-10-31，本机 RTX 4060
- 模型与数据：同 H07
- 内容与方法：按一份外部代码评审增加 buffered、packed、vectorized 三条路径，packed 设为默认，四条路径在同一二进制中对照；第二个进程反转路径顺序。另做 Nsight Systems 时间线、166 项回归和 tiny Mixtral 引擎对照（引擎默认走 packed）。
- 超参：chunk 256，10-bit 前缀表，CTA 128；单投影 3 个独立进程，每路径 warmup 10、samples 30；整层双 stream 的 bulk、packed 各一个进程，每 case 20 个样本
- 种子：
- 环境：`vllm_env`
- commit：
- 交付包 SHA256：本机运行。同轮源码打成的 `fluxmoe-decoder-review-src.tar.gz` 即 H09 在 A40 上运行的包。
- 结果：单投影三进程均值 bulk 9.575、buffered 9.060、packed 14.242、vectorized 14.007 GB/s，packed 相对 bulk 延迟比 1.488×；整层 bulk 9.011、packed 13.524 GB/s，两条路径都有超过中位数 2 倍的长尾样本。
- 记录：`results/decoder_review/`

### H09 A40 解码器评审版验证（stage 1/2）

- 时间与机器：2026-10-31，服务器 A40 GPU 0，用户手动执行
- 模型与数据：服务器上的 Mixtral 第 0 层分片；tiny Mixtral
- 内容与方法：stage 1 在服务器编译后跑完整回归（`FLUXMOE_TEST_CUDA=1 FLUXMOE_TEST_VLLM=1`），检查四条路径对真实 expert 0 gate_up 的解码逐位一致，并用 compute-sanitizer 的 memcheck 和 racecheck 跑改动相关的边界测试。stage 2 做单投影四路径对照、整层双 stream 的 bulk/packed 对照和 tiny Mixtral 引擎对照，所有窗口保留，统计在本机离线重算。
- 超参：chunk 256，CTA 128；单投影 3 个独立进程，每路径 warmup 10、samples 30（第二进程反序）；整层 bulk、packed 各一个进程，warmup 10、samples 50
- 种子：tiny 引擎 vLLM seed 27415
- 环境：`fluxmoe-steve`，nvcc 13.2.78 / sm_86，compute-sanitizer 2026.1.1.0
- commit：
- 交付包 SHA256：`fluxmoe-decoder-review-src.tar.gz`，`85a58b593253e5aa43fed7eede6e7163bb3569174fbd04686c28789931aaa4a8`；stage 2 另以散文件交付，清单为 `results/decoder_review/a40_stage2_handoff_20261031/stage2.sha256`
- 结果：166 项回归无 skip；四条路径逐位一致；sanitizer 0 错误、0 hazard。单投影 bulk 83.654 GB/s、packed 130.029 GB/s，延迟比 1.554×；整层观测延迟比 1.554×（仍为 2/64 case）；tiny 引擎 11 请求 / 176 token 一致，logprob 误差 0。
- 记录：`results/decoder_review_server/stage1-nPEGkFcD.e1vE5Hr2/`、`results/decoder_review_server/stage2.U5aGUKgp/`；离线审计 `results/decoder_review_a40/a40-stage2.2_1tkjjt/`

### H10 A40 解码与 GEMM 的双 stream 重叠（stage 3 与 Nsight）

- 时间与机器：2026-10-31，服务器 A40 GPU 0，用户手动执行
- 模型与数据：Mixtral 第 0 层 expert 0、expert 1 的 gate_up（各 [28672, 4096] BF16）；输入激活为固定种子生成的合成数据
- 内容与方法：模拟一个分页 MoE 层的“物化 + 计算”流水。只有两个 BF16 权重槽，按固定专家序列 0,1,1,0,0,1,1,0 执行 8 个 job，每个 job 先把压缩源解码进槽，再做一次 \([M, 4096]\) 激活与该权重的 GEMM，保留槽的 WAR、权重的 RAW 依赖以及两条 stream 的结束 join。比较三种调度：gemm_only（两个专家预先驻留，计时内不解码）、serial（解码与 GEMM 在同一 stream）、overlap（解码与 GEMM 分属两条 stream）。组合为 \(M \in \{32, 256\}\) × decoder ∈ {bulk, packed}。计时从 control stream 的公共 begin 到两路完成后的 end，包含初始填充与排空。NCU 因计数器权限（`ERR_NVGPUCTRPERM`）没有采到数据。随后在 M=256 下用 Nsight Systems 对 bulk/packed 的三种调度各采一个进程，分析真实 kernel 交叠。
- 超参：每 case 3 个独立进程 × (warmup 5 + 20 窗)；chunk 256；Nsight 每 case 1 个进程 × 3 窗
- 种子：激活生成种子 \(260431 + M\)
- 环境：`fluxmoe-steve`；nsys 2025.6.3
- commit：
- 交付包 SHA256：在 H09 的 decoder-review 包解包目录上运行，另以散文件交付，清单为 `results/decoder_review/a40_mechanism_handoff_20261031/stage3.sha256` 和 `results/decoder_review/a40_nsys_handoff_20261031/nsys_handoff.sha256`
- 结果：8-job 完整窗口（ms，3 进程均值）如下。

  | M | decoder | gemm_only | serial | overlap | serial→overlap |
  |---:|---|---:|---:|---:|---:|
  | 32 | bulk | 3.760 | 26.080 | 25.507 | −2.20% |
  | 32 | packed | 3.757 | 18.091 | 17.782 | −1.71% |
  | 256 | bulk | 4.702 | 26.817 | 26.516 | −1.12% |
  | 256 | packed | 4.686 | 19.028 | 18.878 | −0.79% |

  双 stream 只带来很小的收益；即使 overlap，packed 窗口仍是 gemm_only 的 4.0–4.7 倍。Nsight（M=256 packed）显示每窗真实 kernel 交叠约 1.24 ms，但同时解码 kernel 总时长增加 0.61 ms、GEMM 增加 0.37 ms，首末 kernel 跨度净缩短只有 0.25 ms。
- 记录：`results/decoder_review_server/stage3.9Z5bnNNM/`、`results/decoder_review_server/nsys.0kLlAROg/`；离线审计 `results/decoder_review_a40/a40-mechanism.sn3po49d/`、`results/decoder_review_a40/a40-nsys.e5739e3k/`；方案文档 `docs/A40_PACKED_NEXT_ROUND.md`

### H11 A40 K0/K1 与 compute stream 优先级筛选

- 时间与机器：2026-10-31，服务器 A40 GPU 0，用户手动执行
- 模型与数据：同 H10，M=256
- 内容与方法：两个可分开识别的因素。F 为 K0 与 K1（见共同信息）；P 为把 compute stream 设到可用的最高优先级（实际为 −3，load/control 保持 0），或保持等优先级。单投影比较 K0/K1；完整 8-job 窗口共 7 个 case：K0、K1 各自的 serial、overlap 等优先级、overlap 高优先级，再加 K0 的 gemm_only。两条 compute stream 共用同一组两个权重槽，在 case 边界切换句柄。统计先在进程内对窗口求均值，再对独立进程求均值和样本标准差。另对 6 个配置采短 Nsight 时间线解释机制。
- 超参：单投影每 kernel 3 个独立进程 × (warmup 10 + 30)；完整窗口每 case 3 个独立进程 × (warmup 5 + 20 窗)；共 12 个正式 worker；chunk 256，CTA 128
- 种子：激活种子 \(260431 + 256\)
- 环境：`fluxmoe-steve`；K0、K1 分别用 nvcc 13.2.78 / sm_86 编译
- commit：
- 交付包 SHA256：`fluxmoe-k0k1-priority-src.tar.gz`，`0a57246f2e38874a3206c91310a8fe0e5a736889e778bd4a3319ec76ba61c47f`
- 结果：单投影 K0 1.812 ms（129.6 GB/s）→ K1 1.680 ms（139.8 GB/s），降 7.31%；同一调度下完整窗口 K1 比 K0 降 5.3%–5.5%；高优先级在 K0、K1 上分别再降 0.70%、0.87%；K1 高优先级相对 K0 等优先级从 18.956 ms 降到 17.780 ms（−6.20%）。K0/K1 各 172 项回归和 sanitizer 检查通过。
- 记录：`results/decoder_review_server/k0k1.5SXVJVCF/`；离线审计 `results/decoder_review_a40/k0k1-priority.443m_gi2/`

### H12 A40 五进程确认与 M=32 对照

- 时间与机器：2026-10-31，服务器 A40 GPU 0，用户手动执行
- 模型与数据：同 H10，M ∈ {256, 32}
- 内容与方法：复用 H11 的 K0/K1 二进制，不重新编译。每个 M 比较 K0 等优先级、K1 等优先级、K1 高优先级三个配置，配置顺序按事先固定的平衡设计。另有 6 个独立正确性 worker，不进入统计。
- 超参：每配置 5 个新独立进程，每进程 warmup 5 + 20 窗
- 种子：激活种子 \(260431 + M\)
- 环境：同 H11
- commit：
- 交付包 SHA256：`fluxmoe-confirm-m256-m32-src.tar.gz`，`f477b6de281e3e220693dea5e34b7d73118898a4184be7e915833738250d9ff2`
- 结果：M=256 时 K1 等优先级比 K0 等优先级降 5.43%，高优先级再降 0.92%；M=32 时 K1 等优先级降 5.92%，高优先级反而比 K1 等优先级慢 1.69%。此后默认配置定为 K1、等优先级。
- 记录：`results/decoder_review_server/confirm-n5.awQn1KtD/`；离线审计 `results/decoder_review_a40/confirmation.xqj1fn46/`

### H13 表 3 全 64 case 解码矩阵：bulk 与 packed

- 时间与机器：2026-11-01，服务器 A40 GPU 0，助手经 SSH 执行（用户授权）
- 模型与数据：完整 Mixtral checkpoint。32 层 × 逻辑 EP2 共 64 个 case，每个逻辑 rank 持 4 个连续专家，按 H06 的放置去掉 1 个 CPU 投影，每 case 解码 7 个投影；全部 448 个投影共 78,920,024,064 字节原始数据。
- 内容与方法：先在 CPU 上把 448 个投影编码成 FMBF 并回解核对。每个 case 在计时外准备 native 表并上传，然后在 gate_up、down 两条 stream 上解码 7 个投影，窗口定义同 H06。对照同一 K1 构建中的 bulk 与 packed 路径。正确性阶段由两个独立进程各跑全部 64 case，检查初始和预热后的 GPU 输出 SHA256 与原始 BF16 一致；性能阶段与正确性阶段分开。
- 超参：chunk 256，CTA 128，所有 stream 优先级 0；每配置 5 个新独立进程，每进程 64 case × (warmup 5 + 20 窗)，配置顺序 BP、PB、BP、PB、BP
- 种子：
- 环境：`fluxmoe-steve`
- commit：
- 交付包 SHA256：`fluxmoe-next-stage.tar.gz`，`8894e94df1dccfa4a69c25f619ca4cdfb5ac701c90761094b435c499a08e211a`（服务器解包为 `projects/fluxmoe-next-stage.4KP0Cf8G`）
- 结果：两条路径 64/64 逐位正确，12,800 个正式窗口全部保留。每 case 均值 bulk 13.78–15.87 ms，packed 8.19–9.49 ms；平均吞吐 bulk 83.30 GB/s，packed 140.84 GB/s（论文 401.9 GB/s）。packed 每个 case 耗时降 38.34%–41.60%，64 case 的几何平均加速 1.691×。回收的结果包 SHA256 为 `944e85b0e5b9b79580e0d4f845fba5725e6f957df469aebc45e8cb5ea42ef3f3`。
- 记录：`results/authorized_server_20261031.pUSyREHp/execution.0P1fjJVB/heartbeat.0dv0te6l/BULK64_REPORT.md` 与同目录 `bulk-analysis-rendered/`

### H14 TP1/PP4/EP1 真实 Mixtral 引擎正确性（R1–R9）

- 时间与机器：2026-10-31 至 11-01，服务器 4×A40，助手经 SSH 执行（用户授权）
- 模型与数据：完整 Mixtral checkpoint；三请求短负载跑两轮，再去掉全部数值 hook 重放一轮
- 内容与方法：第一个能加载完整 Mixtral 的多卡候选。PP4 每卡 8 层；CPU 分阶段加载、逐模块在 GPU 上后处理，CPU 专家直接绑定 VMM，避开原单卡入口先把整模型放上 GPU 的做法。同样四张卡上先后运行 resident（全驻留）与 paged 两个引擎，比较完整 logits、生成与 prompt logprob、全部 32 层的 KV 写入与历史值，attention 对独立 CPU 参考。每次修正都打新包、用新目录运行。
- 超参：BF16、eager、TRITON_ATTN、MoE backend triton、每 rank KV 17 MiB、K1 packed CTA 128；作业内设 `NCCL_CUMEM_ENABLE=0`（默认设置下四 rank all_reduce 探针失败）；R7 起加 `CUDA_MODULE_LOADING=EAGER`
- 种子：vLLM seed 27415
- 环境：`fluxmoe-steve`
- commit：
- 交付包 SHA256：
  - R1 `kv-extension-source.tar.gz`：`d63161b37a307099a70a867a748a3b90a4e4e543ee70ef2fc325b870c0486dc9`
  - R2：`ac19a6615f586afad72113e63050b16edce427bda14aef9f96099c440b937d3b`
  - R3：`a64a22b7cd5beb1477be7272e2a91a3bd4006265b6500843dd3288d839fe1974`
  - R4：`4c0f832baa4f82c423c4c2a8980dee6579767f6e3652194514de9cefb5067407`
  - R5：`05d7dcfb4d9a4a9380c3819d64e88ce9aeaf83a5e154bbb61138d90364cff9b3`
  - R6：`588272626b14d505736c8199ac64a2b0a4a83935eca65dc7791d6ffe9e239c90`
  - R7：`a587baae91af03bf28280a5076aa3403e8babb813929512e1c69cd8150002c4a`
  - R8：`10e4c8d967e954da5b79e81c2edcc762cc5d3099351c2162fa8e6b50884dd0b8`
  - R9：`effcb90481d3f7fa1f6ddd1605351893194409bea848b9785d9c351802db91be`
- 结果：
  - R1、R2 的三次失败依次是 NCCL 初始化失败、KV 预算没算 vLLM 保留的 null block、跨进程 MsgPack 拒绝 Python 函数对象。
  - R3 resident 完成三轮请求，32 层 KV 写入（22,336 次）和历史 slot（152,320 次）逐位通过，但第 13–31 层 attention 与 FP32 CPU 参考超差。之后从 vLLM 0.29 源码确认 Triton attention 在 P×V 之前把未归一化概率转成 BF16，参考改为相同语义。R3 paged 在初始化预热卡住超过 300 秒，被停止。R4、R5 在上 GPU 前取消。
  - R6 resident 初始化时 rank 2 停在 Triton `load_binary`，其余 rank 的 NCCL RECV 600 秒超时；R7 加 EAGER 后仍失败。
  - R8 在初始化时预热 KV 写入内核并做一次 Gloo barrier，resident 与 paged 都完成。32 层 KV 检查通过；首轮两侧 349 个 logprob 误差 0，各侧首轮与去 hook 重放误差 0，greedy token 一致。但按同请求、同位置对齐后，第二轮 paged 与 resident 有 99/349 行完整 logits 超出 1e-5（最大 2.125）；resident 两轮之间逐位一致，paged 两轮之间同样是这 99 行不同。从第二轮开始两侧的 prefill 打包不同，例如 paged 第二轮局部步骤 1 的 query 长度为 [22, 106]，resident 为 [22]。这 99 行的 top-1 都没有变，生成 token 不变。R8 证据包 SHA256 为 `14ecb46d77895fc87229220d2101345957e5865b9a32902b0aef034e8c7864c7`。
  - R9（统一入队后再恢复调度）在上 GPU 前被用户叫停。
- 记录：`results/authorized_server_20261031.pUSyREHp/execution.0P1fjJVB/`（`CURRENT.md`、`heartbeat.0dv0te6l/REPORT.md`、`correctness-followup.plv47mhh/REPORT.md`）；R8 logits 影响的离线分析 `results/logits-output-impact.s4kefpb5/`

### H15 TP2/PP2/EP2 U16 全驻留控制的正确性

- 时间与机器：2026-11-01 至 11-02，服务器 4×A40，用户手动执行
- 模型与数据：完整 Mixtral checkpoint、cache512；三请求短负载，两轮
- 内容与方法：在四卡 TP2/PP2/EP2 上比较 vLLM 原生加载的 resident 引擎与新的混合驻留引擎（paper 分支）。U16 让每 rank 的 16 个本地层全部未压缩常驻，作为混合引擎在不分页时与 resident 的数值控制。准备阶段先用 manual-paper 包在服务器编译混合 VMM 与动态 K1 库；原生 smoke 因构建回执缺少 cudart 字段失败，用 receiptfix 脚本补齐后完成 smoke，并复用 H13 的 448 个投影、新编码 64 个，得到 cache512。U16 首次运行在引擎初始化停住 2,487 秒；300 秒通信诊断显示四个 rank 都停在 PP 初始化预热后的同步，PP 通信走了跨 SYS 的 P2P/IPC；只对子作业设 `NCCL_P2P_LEVEL=NVL` 后，通信检查 29.5 秒完成，跨 SYS 改走 SHM。下一次运行在首次入队后失败，原因是前端把 enqueue 返回的内部请求 ID 当成外部 ID，修正后重跑。这一轮中旧的 CPU attention 参考在第 17 层失败（global rank 2、请求 1、位置 24、head 5、dim 66），本机分析认为是某个中间 attention 权重落在 BF16 舍入分界附近。于是在单张 A40 上用 84 份已捕获的原始 Q/K/V 回放原 kernel，又写了只读原始 Q/K/V、按 BF16 语义计算的独立 GPU attention 参考，先在 84 个真实样本、16 个已知闭式答案的控制和 2 个反向控制上验证，再接入整引擎重跑 U16。
- 超参：每 rank 总预算 42,929,750,016 字节，诊断 KV 17,825,792 字节；max_num_seqs 1，max_num_batched_tokens 128，context 512；BF16、eager、TRITON_ATTN；K1 packed CTA 128，stream 优先级 0
- 种子：vLLM seed 27415
- 环境：`fluxmoe-steve`；四卡作业内 `NCCL_P2P_LEVEL=NVL`
- commit：
- 交付包 SHA256：
  - 准备：`fluxmoe-paper-mixed-prepared.tar.gz`，`65a10d4edd9b1bb3cd329c7d26aff3ddab255a369d6b6b360aa6d2cde86ba946`；补丁脚本 `receiptfix_jnknenar.py`，`26a7220fbe741d497c0145b3c7a07ee987efbe94bd9277a9a15ffedf5804f2fd`
  - 通信诊断、请求 ID 修正、第 17 层回放和 GPU 参考验证均以散文件交付，清单分别在 `results/comm-init-diag.nc2lozc8/`、`results/comm-nvl-scope.jl4fkkwt/`、`results/u16-idfix.7935xbyt/`、`results/u16-attn-replay.wk825rwk/delivery/`、`results/u16-oracle-gpu.dlrefuty/delivery/` 的 `delivery.sha256`
  - 通过的 U16 运行：`results/u16-oracle-engine.igs3ewdu/delivery.tar.gz`，`16982b0d66e97cf15f3576ebac70c968b90016e71f5340b3247a6a3d528a3ef2`
- 结果：U16（WORK `u16-oracle-engine.5rpu64rj`）通过。两侧共 5,376 次 attention 与新 GPU 参考逐位一致；两轮 698 个语义行完整 logits 最大差 0；99 个指定位置 × 32 层 × 2 个 TP 分片的层激活通过；每侧 44,672 次 KV 写入、304,640 次历史 slot 和 2,688 次历史 attention 核对无失败。旧 CPU 参考在第 17 层的失败保留在记录里。H16、H17 都以这次运行的 resident 结果为比较基准。
- 记录：`results/u16-engine-review.49urub04/`（含审计入口）、`results/u16-attn-arithmetic.wyF5bHPS/`、`results/u16-oracle-review.2hcxjq3j/`、`results/paper-mixed.s27gt4uc/`

### H16 TP2/PP2/EP2 U8 与 U0 固定策略的正确性

- 时间与机器：2026-11-02，服务器 4×A40，用户手动执行
- 模型与数据：同 H15
- 内容与方法：只运行混合引擎一侧，与 H15 的 resident 结果比较，门槛同 H15。U8 每 rank 8 层常驻、8 层压缩分页；U0 每 rank 16 层全部分页（每 rank 128 个压缩源），并保留原有的自然驻留控制器。
- 超参：同 H15
- 种子：vLLM seed 27415
- 环境：同 H15
- commit：
- 交付包 SHA256：U8 `u8-oracle-engine.qdjg_974.tar.gz`，`d2fd91126e52512debcd8b0063f8a7215d22a7c146cfd6aca4cc3472f842f694`；U0 `u0-oracle-engine.a2fi871l.tar.gz`，`9033995a8915fcea3ee4dc4ac54a90c2cd07e6b97c023540e65c4e80a800bedc`
- 结果：两者都通过（WORK 分别为 `u8-oracle-engine.lda7lu4z`、`u0-oracle-engine.ilg945rc`）。U0 的两个分页槽都实际用到；自然控制器在每个 rank 上发生了一次 down 投影的 CPU 换出与恢复，但恢复发生在完整数值采集结束之后。
- 记录：`results/u8-engine-review.2rsv3wy1/`、`results/u0-engine-review.zsjpc_rs/`

### H17 U0 受控换出/恢复（cycle、cycle2、两层栈）

- 时间与机器：2026-11-02 至 11-04，服务器 4×A40，用户手动执行
- 模型与数据：同 H15
- 内容与方法：在 U0 上通过受控 RPC 在指定轮次之前把投影换到 CPU（offload）或恢复成压缩源（restore），让 CPU 后备状态与恢复后的状态都落在完整数值采集的轮次内，与 H15 的 resident 结果比较。依次为：cycle，每 rank 一个 down 投影，第 0 轮前 offload、第 1 轮前 restore；cycle2，两次 down 的 offload/restore，共四轮；stack2，O/O/R/R 两层 down 栈，深度 2 的后进先出恢复，四轮的 CPU 投影数为 1/2/1/0，覆盖两个奇偶分页槽；gate_up stack2，同样的两层栈改用 gate_up 投影。
- 超参：同 H15
- 种子：vLLM seed 27415
- 环境：同 H15
- commit：
- 交付包 SHA256 与各次运行：
  - cycle `2c0f1cffbf0edd954808e192298422289d5fc330d32cdd1cec7fd6c0a4cae9da`：通过（WORK `u0-cycle-engine.co4opmg3`）
  - cycle2 `6b1235cad3919fe77c5a9685317a8d08a9496cf459edd147a5018063a5686b9b`：通过（WORK `u0-cycle2-engine.au5xgype`）；每 rank 159 次 iteration、20,300 次解码、52 次 H2D，四轮 logits 最大误差 0
  - stack2 `81514051b9ff8cba6812185b6eb8b8a1b0997be3089b7e13d6cfec1d860efe02`：失败，rank 1 的 worker 在引擎预热阶段退出，受控转换没有执行
  - stack2-diag `e14f7d551e21b1ed97724ffa664e8d4ba248cb010a20f227f1765d8ad07376a9`：失败，EngineCore 收到 SIGSEGV（当时开着周期性 Python 栈转储）
  - stack2-nodump `e928c60fa99e201256210810a7621e334fd237f7cad8a61748da4d29f7e1f53d`：只关掉周期栈转储，四轮数值与两层 LIFO 机制通过，但关闭时 EngineCore 被 SIGKILL，四个 worker 的退出记录缺失
  - stack2-shutdown `ec9aa0ca4cb02c8ef3fdc2ee63086e6fe72261a688a44ee019b0125c0ffd8d74`：第一次运行被资源门禁拒绝（有组外进程占用 GPU），未进入模型初始化；第二次以 `u0-stack2-shutdown-next.LnzbD9ry` 目录重新交付（散文件，清单 `SHA256SUMS`）后整轮通过，EngineCore 退出码 0，四个 worker 在父进程 SIGTERM 后以 −15 退出，关闭在 20 秒上限内
  - gate_up stack2 `b4fec6f9471b23583026d65ecc87c31dd293092305b7d63d7770f939ac154c0a`：CPU 预检失败，原因是包里固定的是笔记本 Python 3.14 的 `ast.dump` 哈希，与服务器 Python 3.12 的表示不同
  - gate_up sourcefix `1069335ba970d22840d3b1505099330eaaf5671931024cf02b504d4b84ac3322`：运行中资源门禁失败
  - gate_up guardcheck `845bf4cd5d1e0bd641d004e3d468c9ef71cd789246bb017314c15e3e4c92a79d`：通过（WORK `u0-gate-up-guardcheck.vn6w8ruq`）
- 结果：受控单次、两次 down 循环，两层 down 栈和两层 gate_up 栈都通过数值门槛。以 gate_up 两层栈为例：四轮 1,396 行完整 logits 最大误差 0，277,200 行层激活通过，5,376 次 attention 通过，89,344 次 KV 写入与 609,280 项历史值逐位一致；四个 rank 的 CPU gate_up 栈深度依次为 1→2→1→0，卸载的（全局层:专家）在 rank 0/1 上先后为 15:3、14:3，在 rank 2/3 上为 31:3、30:3，恢复顺序与卸载相反；每 rank 20,248 次解码、104 次 H2D。
- 记录：`results/u0-cycle-review-fix.y1nrqu9r/`、`results/u0-cycle2-engine-review.h0b8f09n/`、`results/u0-stack2-failure-review.s3d02x3j/`、`results/u0-stack2-diag-failure-review.vmk0u8n9/`、`results/u0-stack2-nodump-review.u9x5nhdf/`、`results/u0-stack2-shutdown-failure-review.pubgp3b4/`、`results/u0-stack2-shutdown-review.Mibg8zUv/`、`results/u0-gate-up-stack2-preflight-failure-review.lvACjpwp/`、`results/u0-gate-up-sourcefix-resource-failure-review.mjCd9UWe/`、`results/u0-gate-up-guardcheck-review.GZeuQbkF/`

### H18 R0 与 U0 的固定 batch 计时（并发 1 与 3）

- 时间与机器：2026-11-04，服务器 4×A40，用户手动执行
- 模型与数据：同 H15；每批为三请求短负载
- 内容与方法：先跑一次无计时资格（3 个预热批 + 2 个校验批，输出与已审计 resident 一致，不产生性能样本），验证测量入口。然后在同协议下分别计时 R0（全驻留）和自然 U0（全部分页，自然 planner 开、Eq7 关）。每个计时窗口从四 rank 设备完成同步之后开始，到生成返回并再次完成四 rank 同步结束；预热批不计入。先做并发 1（max_num_seqs 1），再做并发 3。
- 超参：每次 5 个预热批 + 20 个测量批，每批 3 请求 × 8 token（测量部分共 60 请求、480 token）；其余同 H15
- 种子：vLLM seed 27415
- 环境：同 H15
- commit：
- 交付包 SHA256 与各次运行：
  - 无计时资格 `b3c29e7317620ecd1b2d1cc52a3135fca26446182cfabedba8a2bd9bb5e07ad6`：通过
  - R0（并发 1）`b92e3b23294705c6079843267d52a37cce60491cea327e159f9ce48afbff7714`：启动失败，resident 启动器的 native 库选择与 scheduler 开关耦合；修正版 `982cd28581b581e1eb9d24e7637fdad25dfdc3f4ea1e7b446970bf9558cb9eaf` 通过
  - U0（并发 1）`75e8144cf6eff3a71c9aa0f17a1017ee7577abdeb740d4251cf10370aac01607`：初始化成功后、预热前被新增的加载身份校验拦下；修正版 `6afdfd400d23202e3468d51ce20a41454fbb52a002bd8948a35e5048e46e95b2` 完成，服务器上的 CPU 审计入口失败，本机原始审计通过
  - 并发 3 资格 `2092cae9054084ba2f7600c7e0c2ec9d76308a629b2722de6031402f7b719ba7`：reference、自然 U0 正确性都完成，最终审计因输出文件已存在而失败，之后本机补审通过
  - 并发 3 计时 `1ce3abcf8404c2b99170cef0de5592995bba06c4984405f99f7dd87c64e1d887`：CPU 入口导入失败（0.199 秒）；`a1e77dc452a18a9bd1733c462a4ae3f8734b5206ffa845b656f66f48f6d58c14`：R0 计时完成，但参考审计时一个进程经 libzmq 断言收到 signal 6；`c997b5609e5636dcac34ec3feb6899a7c7106d24f7d796ef2d1f318256eef7d4`（只改 ZeroMQ 关闭路径中 `SpinCondition` 的取消操作）通过
- 结果：并发 1 时 R0 窗口 23.898 s（20.09 token/s），U0 174.099 s（2.76 token/s），U0 耗时是 R0 的 7.29 倍；全程实际 decode batch 为 1，每 rank 每次 forward 要重新物化约 22.55 GB 专家权重。并发 3 时 R0 12.608 s（38.07 token/s），U0 67.680 s（7.09 token/s），比值降到 5.37。U0 全程后端版本为 0，没有 H2D、没有 CPU↔GPU 迁移，开销来自 GPU 压缩源的反复解码物化。离线分析显示从并发 1 到并发 3，U0 的物化次数减少 61.5%，窗口耗时减少 61.1%，每次 forward 的平均耗时只上升 1.1%。
- 记录：`results/tp2pp2ep2-untimed-pilot-review.qTicttSH/`、`results/tp2pp2ep2-resident-r0-nativefix-review.bc7m_z2d/`、`results/tp2pp2ep2-natural-u0-audit-import-review.3z00kzd3/`、`results/tp2pp2ep2-batch3-short-zmqfix-review.z9fmo9z7/`、离线分析 `results/u0-r0-cause-analysis.bu4wlh57/`、`results/bcfg1-bcfg3-offline-analysis.dlfuej5m/`

### H19 单引擎 ABBA 机制诊断（五项单因素）

- 时间与机器：2026-11-04 至 11-07，服务器 4×A40，用户手动执行
- 模型与数据：同 H15，U0，并发 3，每批为三请求短负载
- 内容与方法：每项在一个新引擎内按 A、B、B、A 四段运行，每段 1 个预热批和 1 个诊断批，前面先跑 1 个 B 路径的数值资格批（门槛同 H15）。窗口从四 rank 设备完成同步后开始，到生成返回并再次同步后结束；每臂比较两个诊断窗口的均值。运行时开着 Nsight Systems。五项依次叠加：前一项的较优或中性设置在后一项中固定。
  1. 预取：A 为两层预取开，B 为按需加载。
  2. retirement：两臂预取都开；A 为原 `VersionedBackends._collect`，B 在退休队列为空时直接返回。动机来自对上一项 trace 的离线分析：前向边界每 rank 每段有 26–31 ms 主机开销，其中空退休队列的读者扫描可以短路。
  3. native_call 观测：两臂 retirement 都用 B；A 记录 `native_call` 内部 JSON/NVTX/计时，B 关闭。
  4. 物化 stream 拓扑：A 为 gate_up、down 各用一条物化 stream，B 把 down 映射到 gate_up 的 stream。
  5. 投影提交顺序：A 为 gate_up → down，B 为 down → gate_up。
- 超参：每项 9 批（1 资格 + 4 预热 + 4 诊断），共 27 请求、216 token（native_call 一项无资格批，8 批）；其余同 H15
- 种子：vLLM seed 27415
- 环境：同 H15；nsys 2025.6.3
- commit：
- 交付包 SHA256：
  - 预取：`a37abbe8c1b33686bb3b746aef4f70dee76792dd6bfb7e920817a4627b01b58c` 启动失败；`6254fc97c8eeace5f6614ada218fcdc5bd28270095f00528ad566700de1af9bb`（缩短 IPC 路径）完成（WORK `tp2pp2ep2-prefetch-ipcfix.8mq97zp8`）；`6894f8b5068dd1a590cee0896595611e15d45b9e577f2181dfbd55575bb752eb` 只用于重新导出证据
  - retirement：`tp2pp2ep2-retirement-fastpath` `73279dc5125f7a03d47f0ae7e581fc9354ef354ee536d34abfaae6d8d772e272` 被资源准入拒绝；`admission` `2a8b927ca64f8ac3efff2c5acdc557481b86a9c2b14f81fff5c280e9aea6a2d1` 启动了引擎，但在资格批之前的辅助进程退出检查中，因 4 个 `which nvcc` 子进程（vLLM deep_gemm 查找 CUDA_HOME 时启动）退出码为 1 而停止；`cudahome` 两包 `3a3614253dd8ae1625ac5f6fe50fa486dad203350e740a255fb1172e91a278c9`、`1e92b98624a15776d5907890be845b9309bc5f868e7f9c38e1e960727da9ab11`，`trackerfix` `0f4c967e93c3b9dea8c59c664c8b724ec497f64130b01210991ac98776a06c2a`，`bootfix` `bdf2b228ab5158c5f90ed3e0b352c7f26213976c9e1da7e57e02eaffdbd0d8ed` 都停在启动前的前序退出核验、boot 身份或 CPU 预检；`astfix` `813217197971cca7f274e33b42d7988cc501ad33e3210921ffef4d3dbf07b4a0` 完成（WORK `tp2pp2ep2-retirement-astfix.luwwnxib`）
  - native_call：`98c7aa7aee50e833c6d725ecd5cb752ab645dd584b10b856dbe59fd916ce3c43`（WORK `tp2pp2ep2-native-observer.y_68ruhz`）
  - stream 拓扑：`6781685520d1486451ae139b2c31a2e5f2fb51c6abf4e9f64a3d900af129b106`（WORK `tp2pp2ep2-stream-topology.5wmjxm4q`）
  - 提交顺序：`d717b27dd4bfb0509e829212201c07f484a9670639320ef68a83fc769c50b430`（WORK `tp2pp2ep2-projection-order.so1g5pl3`）
- 结果（每臂两窗均值）：
  1. 预取：A 3.3962 s，B 3.4036 s，预取开快 0.22%。
  2. retirement：B 比 A 少 15.77 ms（−0.46%）。
  3. native_call：A 3.37550 s，B 3.37599 s，关闭记录没有缩短完整窗口。
  4. stream 拓扑：B 比 A 多 309.92 ms（+9.18%），保留两条物化 stream。
  5. 提交顺序：B 比 A 少 5.79 ms（−0.17%），未达到预设的 1% 筛选线。

  带资格批的四次运行中，B 路径数值资格都通过。U0 全程后端版本为 0、H2D 为 0，没有发生迁移。Nsight 采集带有一批保留的告警（含 worker CUPTI 缓冲区可能缺失事件），全局 trace 完整性没有闭合。
- 记录：`results/tp2pp2ep2-prefetch-review.5l0_h4_3/offline-review.8o6svx5z/`、`results/tp2pp2ep2-host-launch-analysis.tw5sd7f0/`、`results/retirement-complete-review.z1ixw20f/`、`results/native-observer-complete-review.f6iz4z4g/`、`results/stream-topology-complete-review.s0uet35q/`、`results/projection-order-complete-review._5asbqzm/`

### H20 静态 CPU 专家放置 A（FluxMoE-H）与 B（FluxMoE）

- 时间与机器：2026-11-07，服务器 4×A40，用户手动执行
- 模型与数据：同 H15，每批为三请求短负载
- 内容与方法：唯一变化是静态 CPU 专家放置（见共同信息）。两臂 CPU 专家都是 10.5 GiB，各 rank 分布 A 为 [0, 0, 5.25, 5.25] GiB，B 为 [3.5, 1.75, 3.5, 1.75] GiB；每 rank CPU 上限相同，为 5.25 GiB。其余固定：并发 3、cache512、两层预取、retirement B、native_call 记录关闭、gate_up → down 顺序、四条优先级 0 的 stream；运行中暂停驻留调整，不发生迁移。每臂先过新的数值资格（门槛同 H15，GPU 参考在资格请求和四 rank 同步之后运行）。共四步：带 profiler 的 A、B 各一个引擎；无 profiler 的 A 单引擎（未完成）；A 的正常生命周期验证；最后无 profiler 的四引擎 A1 → B1 → B2 → A2，每个引擎 5 批（1 批 token 重放 + 2 批预热 + 2 批计测），每臂两个独立引擎。
- 超参：同 H15，并发 3
- 种子：vLLM seed 27415
- 环境：同 H15
- commit：
- 交付包 SHA256：
  - 带 profiler 的 A/B：`76ad18bf8d14b423ee859a65e1f7f8570f15111f044a6d467f37f23c21c52c5a`（WORK `tp2pp2ep2-static-placement.bds082a_`）
  - 无 profiler 单引擎 A1：`fd04b54c3f927995f28eca8aa258089676cdaa83a7f219ccfb3a5dfa1c66a6e4`，在测量前失败，EngineCore 关闭超时后被 SIGKILL
  - A1 生命周期：`165e882de1b0100cd43d2fe159936e2788cc0044377c3ae3377396ba93210cdb`，通过
  - 四引擎 ABBA：`e731fd7de27a810b3f736bef4213a7cdf27743ec0d38e059d34642eff81b9ab4`
- 结果：两臂资格都通过（349 行 logits 误差 0，69,300 行层激活，1,344 对 GPU attention 参考误差 0，KV 22,336 / 1,344 / 152,320 次逐位）。带 profiler 时每窗 A 7.037 s、B 3.703 s（−47.4%），trace 显示 B 的 decoder 与 H2D 重叠增加、后一 PP stage 的物化区间缩短。无 profiler ABBA 中各引擎均值 A1 5.695 s、B1 3.502 s、B2 3.383 s、A2 6.183 s，四引擎汇总 A 5.939 s、B 3.443 s，B 耗时低 42.03%，两个顺序相反的区块方向一致。这是固定小 batch 下的开销对比，没有涉及容量。
- 记录：`results/static-placement-complete-review.d6f291qv/`、`results/static-unprofiled-failure-review.rb279bym/`、`results/static-lifecycle-complete-review.uk00xu6m/`、`results/static-abba-complete-review.irqmc4_x/`（原始证据在对应 `*-prep.*/received.*/raw`）

### H21 TP2/PP1/EP2 两卡短负载的严格数值资格

- 时间与机器：2026-11-08 至 11-11，服务器 2×A40（前期 GPU 0/1，`ycs08yu8` 起 GPU 2/3），用户手动执行，后期由助手经 SSH 执行
- 模型与数据：完整 Mixtral checkpoint、cache512；三请求短负载，一个真实 batch；参考为 H18 并发 3 资格运行中 PP2 resident 的已审计数组（12,665 个索引文件；PP1 的专家与 dense 权重哈希等于对应 PP2 层的并集）
- 内容与方法：把静态放置迁到两卡 TP2/PP1/EP2（论文主实验的部署形态），单个静态 B 引擎（后补 A），门槛同 H15。比较键为（轮次、请求序号、输入位置），PP2 取 rank 2/3 的 logits，PP1 取 rank 0/1。第一次完整数值运行有 190/698 个位置的 logits 超差，首个超差边界在全局第 0 层 attention 输出。为定位原因，在第 0 层采集 RoPE 前后的 Q/K/V 与 InputBatch，做现场交接、PP1 自回放和按 PP2 记录布局回放三道逐位门槛。结论是两者在 step 1 的批布局不同（PP1 处理两个短请求的首次 decode 和长请求的 [90, 216)，resident 只处理长请求的 [90, 218)），差异来自不同的批内行上下文。随后的完整数值运行通过 vLLM 已有的 eligibility 条件把短请求顺延一步，使布局与 resident 完全一致。
- 超参：BF16、eager、TRITON_ATTN、K1 packed CTA 128；其余沿用 H15 的预算与容差
- 种子：vLLM seed 27415
- 环境：同 H15
- commit：
- 交付包 SHA256 与各次运行：
  - `tp2pp1-cache-qualification-prep.j8t0lfza` `5efc2a0c7a960271caba3ed4285c80630f1c45d1632245d474a0d389b0cd2dd7`：停在历史前序门禁的 source_pins 比较，未建 WORK
  - `tp2pp1-lineagefix-prep.ct0uhzlc` `759478be7decf16dc4830d227f8e28489bcbfe1615e691e7b78e9a32128970d0`：policy 顶层导入失败，未建 WORK
  - `tp2pp1-importfix-prep.hbnyl1xp` `db8628349885f4baf40e68f30ef3fa441460bcb94ef6083cca4295592882318c`：两 rank × 32 层初始化完成后前端诊断日志容量耗尽，未进入数值资格
  - `tp2pp1-capturefix-prep.qj7hdnfx` `d66e5ae9732b02e7ce8850af1647cc309225522dd6a9d86ad7f7c2328ef09a27`：完成，数值失败，698 个位置中 190 个超差、最大绝对差 5.25，698 个 argmax 全部相同
  - `tp2pp1-qkv-context-prep.qLshN4ji` `64d6926f942eb0a00d3f85bcf652a9c265fb74bb2150bebf8b98a0f1f29e5dc1`：采集 hook 的 Q/K 视图检查失败
  - `tp2pp1-qkv-handoff-prep.JhXQaLeJ` `e4c6b85a0db3e0a9c5c3908df417053208d605d77b5b2573878c2eb0957dd3c4`：各层共用同一个 RoPE 对象，非第 0 层的调用触发了 `duplicate layer-0 boundary`
  - `tp2pp1-qkv-owner-prep.546soock` `3c8f580ef9fe4c9d17674a195341752d0ef8934453bb0f68eb2ee67cd4d9d683`：准入时 GPU 0/1 被占，未解包
  - `tp2pp1-qkv-owner-prep.ycs08yu8` `6f1d8a4fdcc78b82606c5edf9339b0d08f6f511fe4134b88678843c5cc888c42`：通过（WORK `tp2pp1ep2-qkv-owner.3behj7nr`），现场交接、PP1 自回放、PP2 布局回放逐位一致，两 rank 各 12 个历史见证复现
  - `tp2pp1-full-numeric-prep.oq98hnk8` `0818fcb55d62d57e1aec3248a69b80cca59482e2dcc64b36c70b3b6f9a76d989`：B 臂通过（WORK `tp2pp1ep2-full-numeric.hqqkvsu8`）；此前一次准入因 GPU 2 被占而未上传
  - `tp2pp1-short-a1-prep.IkcIlWaU` `abc2c3c9d366f78da4b210916a793d5316ba7e8bba3da89584821c5219f8ade7`：A 臂通过（WORK `tp2pp1ep2-full-numeric.l5yjiehl`）
- 结果：布局对齐后，B 臂 698 行完整 logits 最大误差 0，两轮各 69,300 行层激活最大误差 0，44,672 次 KV 写入、304,640 项历史值、2,688 次独立 attention 全部通过；A 臂同一协议通过。两臂短资格是 H22 长负载的前提。
- 记录：`results/pp1-admission-complete-review.a12mvdkm/`、`results/pp1-import-admission-review.76aapjng/`、`results/pp1-capture-complete-review.2bm1vk5w/`、`results/pp1-numeric-failure-review.04bzzeyn/`、`results/pp1-qkv-context-failure-review.Fu6oL0mf/`、`results/pp1-qkv-handoff-failure-review.5mP5fE3w/`、`results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/`（`HANDOFF.md`、`checks/b1-gpu23-operation.9PmV7BuA/`、`checks/full-model-b1-operation.zHxKNCeP/`、`checks/a1-operation.vTI8DDjZ/`）

### H22 ShareGPT 32×4096 长负载的 H/B 两臂资格（含 host-KV）

- 时间与机器：2026-11-11，服务器 2×A40（GPU 0/1，NV4），助手经 SSH 执行
- 模型与数据：完整 Mixtral checkpoint、cache512；ShareGPT 长负载 32 条请求，每条输出 4,096 token，共 131,072 个输出 token
- 内容与方法：论文图 5(a)、图 6(b)、表 4 在 Mixtral 上共用的首个坐标点 \(B_{cfg}=32\)、输出 4096。H 臂用静态放置 A，B 臂用放置 B，两臂每 rank 都是 32 个 CPU 投影、224 个压缩投影，CPU 专家 H 为 5.25/5.25 GiB，B 为 7/3.5 GiB。KV 在 GPU 和 host 两级：vLLM 的 `OffloadingConnector`（native 后端）把 KV 页换出到 pinned host 内存，之后回填并被 attention 读取。验收包括：全部请求完成、两 rank 设备完成同步、原父进程退出；17 个由解码生成的 KV 页在两 rank 全部 32 层完成 GPU→host、释放旧 GPU 块、host→GPU、实际 attention 读取，并逐位比较其位型；每最终 TP rank 保存 209 个完整 logits 行及对应的全层样本，供 H24 的独立参考比较。运行时开着诊断采集。
- 超参：max_model_len 5232，max_num_batched_tokens 128，max_num_seqs 32，chunked prefill 开，prefix cache 关，eager，BF16，TRITON_ATTN，MoE triton，async scheduling 关；temperature 0、`ignore_eos=True`、min_tokens = max_tokens = 4096；GPU KV 4 GiB/rank（LBHNC 布局，16-token 页），host-KV 每 rank 12 GiB pinned、共 24 GiB；单臂时限 7,200 s，累计 RSS 上限 160 GiB
- 种子：vLLM seed 27415；请求抽样种子字符串见共同信息
- 环境：同 H15
- commit：
- 交付包 SHA256：H 首次 `32c4b09ac2b2dec7878adab97e8dabee02bc8f0cafba84e9811072e3973d2536`；B `5083258a4c679faf26d1d1a6374d2b094b05557e81262d1e01c44bd1bd150133`；H 重跑 `8dbfb389d1168d63a0b9c4b08b4f5d683b08599f4d09b3e0775a21219da89fa3`（均为各目录下的 `SOURCE.tar.gz`）
- 结果：H 首次运行 rank 1 收到 SIGSEGV（退出码 −11，rank 0 为 −9），完成请求数 0。B 通过：32 个请求、131,072 个 token 完成，17 个恢复读取见证逐位一致，tracker 峰值 RSS 约 154.4 GB。H 重跑只在模型加载结束时取消周期性栈转储，通过，同样 17 个见证逐位一致。两臂 131,072 个 greedy 输出位置全部相同。
- 记录：`results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/tp2pp1-long-hb-gpu01-prep.iYdNMCzc/`、`.../tp2pp1-long-hb-gpu01-h-prep.6AUlL7gG/`；首点配置 `.../pp1-long-p32-l4096.kTmCeXpc/POINT.json`

### H23 四卡全驻留 host-KV 长资格（未完成）

- 时间与机器：2026-11-11，服务器 4×A40，助手经 SSH 执行
- 模型与数据：完整 Mixtral checkpoint；ShareGPT 长负载 32×4096
- 内容与方法：在四卡 TP2/PP2/EP2/DP1 的 vLLM 全驻留 BF16 引擎上单独验证 host-KV 换出恢复：先跑一个带 native `OffloadingConnector` 的引擎完成全部请求并取证，再用独立新引擎按同一布局回放作为参考。方案见 `docs/HOST_KV_LONG_QUALIFICATION.md`。
- 超参：GPU KV 每 rank 4,096 个 16-token 块（约 2 GiB），CPU KV 池聚合 18 GiB；BF16、eager、TRITON_ATTN；单次两引擎合计时限 7,200 s
- 种子：
- 环境：同 H15
- commit：
- 交付包 SHA256 与各次运行：`DTzwpKRk` `26ae2d1ad7a5855b5a2063260d8fe54b2a0bc30e7d33c3050bbf83e4a6f91bab`：模型预热时 FlashInfer 采样需要的 `ninja` 不在 PATH（启动脚本用了绝对路径 Python，没把 conda 环境的 `bin` 加入 PATH），提交请求数 0；`1akNj9Jy` `61e5d4146f2f8e597341709c55b47d5ac1675724c33b1330e5e913d123d00326`：初始化 rank 2 时 GPU 2 被其他用户占用，可用显存不足；`4p5EJHzP` `1efa5cc3a84956ee7f90fbbccc8871a30e8bff5e2d91d038515a3889491a1b13`：启动前复查时 GPU 2 被占，未启动
- 结果：没有完成任何一侧的请求。
- 记录：`results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/checks/host-kv-four-gpu-qualification.*/`

### H24 四卡常驻 BF16 独立长参考与 H/B 对照

- 时间与机器：2026-11-11 至 11-13，服务器 4×A40，助手经 SSH 执行
- 模型与数据：完整 Mixtral checkpoint（vLLM 原生 safetensors 加载）；H22 的 H/B 输出历史
- 内容与方法：在四卡 TP2/PP2/EP2/DP1 上运行 vLLM 默认 BF16 全驻留引擎，KV 全部常驻 GPU。参考按 H22 记录的实际 InputBatch 逐步回放 6,207 个 step，每步以方法实际生成的 token 作为输入，独立给出 greedy 预测、209 个采样位置的完整 logits、32 层采样张量，以及 17 个恢复锚点上的常驻 KV 页，与 H、B 两臂逐项比较。第一次参考的 greedy 最早分歧在 step 60，离线分析发现参考侧没有按 `min_tokens=4096` 屏蔽 EOS（该位置掩码前的 argmax 正是 EOS）。按目标 vLLM 原生的 EOS 归一化（掩码集合 {2}、`ignore_eos=True`、`stop_token_ids=[]`、min_tokens = max_tokens = 4096，恢复后重新注册）重做了一次全长参考，因服务器重启，参考与 H/B 分属两个 boot，按跨 boot 配对审计。
- 超参：每 rank 5 GiB 常驻 GPU KV；BF16、eager、TRITON_ATTN；容差同共同信息
- 种子：vLLM seed 27415
- 环境：同 H15
- commit：
- 交付包 SHA256：第一次参考 `tp2pp2-resident-bf16-reference-prep.zCPozoMj` `c92f41da6586afaa89093b772ff1db9736b3a4eb2393f390c8a9bf804b6b35d8`；EOS 对齐后的全长参考 `tp2pp2-resident-bf16-long-cross-boot-prep.dTVoVmG7` `393e3874d7e274bb189f063f16d6fc05cc74987ccc1550cbb43f58623e177bf1`（均为 `SOURCE.tar.gz`）
- 结果：H 与 B 的 43,200 个采样数组彼此逐位一致。对第一次参考：每臂 172/209 个去重 logits 行超差（最大 6.648），层激活 61,006/80,256 行、attention 9,720/13,376 行超差，greedy 预测 748/131,072 个不同，最早在 step 60；17 个恢复锚点的历史 KV 在两 TP rank 共 34 页、17,825,792 个 BF16 值与参考完全一致。EOS 对齐后：step 60 的差异消失，greedy 不同降到 277/131,072（265 个在第一次恢复之前），最早剩余差异在 step 395（`sharegpt-p32-15` 第 391 个输出，H/B 为 28723，参考为 28733）；logits 仍是 172/209 行超差；host-KV 的 17 个恢复位置仍逐位一致。
- 记录：`results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/tp2pp2-resident-bf16-reference-prep.zCPozoMj/`、`.../bf16-reference-first-divergence.ik2gVytg/`、`.../tp2pp2-resident-bf16-long-cross-boot-prep.dTVoVmG7/live-index-change.p7X2JSSC/`

### H25 分歧定位短回放：step 0–61、step 395、step 372–374

- 时间与机器：2026-11-12 至 11-14，服务器 A40（B 两卡 TP2/PP1/EP2，参考四卡 TP2/PP2/EP2），助手经 SSH 执行
- 模型与数据：同 H24；从 H22/H24 原件中截取的前缀布局
- 内容与方法：B 臂和四卡参考各运行一次短回放，双方消费同一份冻结的 InputBatch 与 token 历史，在被怀疑的 step 附近逐层采集：RoPE 前后的 Q/K/V、Triton attention 的实际调用参数与 launch、按实际 block table 读到的历史 K/V 和当步写入、O 投影的 TP 局部结果与归约结果、层激活和 PP 交接张量、掩码前后的完整 logits。所有采集都在本臂全部 rank 完成同步后落盘，然后按位置配对比较。三次回放：
  1. step 0–61（32 请求、3,264 个输入 token），采 step 59–61，检验 step 60 的 EOS 解释。
  2. step 0–396（13,984 个输入 token），采 step 394–396，定位 step 395 的分歧。
  3. step 0–374，采 step 372–374，追踪位置 459 的 K/V 写入链。
- 超参：B 沿用 H22 的配置（4 GiB/rank GPU KV、24 GiB host-KV、cache512），参考沿用 H24（5 GiB/rank 常驻 KV）；容差同共同信息
- 种子：vLLM seed 27415
- 环境：同 H15
- commit：
- 交付包 SHA256 与各次运行：
  - step 0–61：`step60-short-replay-prep.32kMOSdy` `593213606a679597c01724f0deccd74abd9fb2722ddde536e86d9a822a133cdd`，B 采集通过，参考在权重哈希后的准入中因 GPU 2 被占而退出；`step60-reference-eos-prep.HRTbzmv9` `a25ab68f2ea370013d13a6c802164f600d81853abfc4753f619196e64a690cbd` 因服务器 boot 已变而停在上传前；`step60-reference-cross-boot-prep.0L324gbt` `4be3fa385259019dce2f5958e62fb97b2141a6b4405bb5b666db10ad99471a79` 第一次因 GPU 0/1 被占停在上传前，第二次完成
  - step 395：`step395-short-replay-prep.br5iSXuY` `f1a6a0cd3b2e37111171d5895558970be702442c55b9526bc24d843e4ba3e51a`
  - step 372–374：`step373-short-replay-prep.WajmSVZ7` `363f4b084fc2bb0879f7d37c94f70a4d1889535e22aaf463a89387cec389fe46`，在 `step373-short-replay-prep-r2.f1Jf7uIn` 目录中执行
- 结果：
  1. step 0–61 两侧 1,802 个 greedy token 全部一致，step 59–61 的 6,144,000 个 logits 值与请求 11 的 3,456 组层内张量全部逐位一致；step 60 请求 11 的掩码前 argmax 两侧都是 EOS，施加 `min_tokens` 掩码后都采到 13，确认此前的差异来自参考侧 EOS 掩码未对齐。
  2. step 395 回放中两臂各 12,522 个预测分别与各自长跑原件一致，唯一的选词分歧仍是 step 395 请求 15（28723 对 28733）。请求 15 在第 11 层读到的历史 K/V 只在输入位置 459 不同，该位置由 step 373 写入。
  3. step 372–374 回放显示存储与读回不是差异来源：位置 459 的 K/V 在 step 373 写入前就已不同；更上游是第 6、8、10 层的 Q 在 RoPE 之后各有 1 个元素不同（RoPE 之前相同），都在频率 18。用 vLLM 0.29 的 `_compute_cos_sin_cache` 公式在 CPU 上重建缓存，可逐位复现 B 臂的焦点 Q/K；把 `inv_freq[18]` 下调 1 个 fp32 ulp 则可复现参考。原因是 B 臂的加载器 `pp1_long_loader.py` 在 `torch.device('cpu')` 下构建模型，RoPE cos/sin 缓存在 CPU 上算，而参考按 vLLM 原生路径在 GPU 上算；位置 459 处 \(\sin\theta\) 在 fp32 下距 BF16 舍入中点仅 \(2.4\times10^{-8}\)，1 ulp 的差异翻转了 BF16 舍入。这一差异与卸载、KV 存储和 host-KV 恢复无关。修正（B 在 GPU 上重算 RoPE 缓存）及修正后的全长复验尚未执行。
- 记录：`results/codex-four-thread-handoff.TbHV6Yvp/work/01-rope-owner/step60-short-replay-prep.32kMOSdy/live-operation.LnRxYbHi/`、`.../step60-reference-cross-boot-prep.0L324gbt/live-reference.9BD3odYy/LIVE_REPORT.md`、`.../step395-live.qDnLDxTz/LIVE_REPORT.md`、`.../step373-short-replay-prep-r2.f1Jf7uIn/STEP373_LIVE_REPORT.md`

### H26 真实 FMBF 投影的长码统计

- 时间与机器：2026-11-06，本机 CPU
- 模型与数据：Mixtral 第 0 层 expert 0 的 fused gate_up 完整投影（本机已有的 FMBF 缓存 `results/decoder_workload_cache/`）
- 内容与方法：按 bulk 路径的 10-bit 前缀表逐 chunk 重放解码过程，统计长于 10 bit 的长码比例、bulk 表一次查表输出的符号数，以及长码在 warp 组之间的分布，用来判断长码回退是否值得优先优化。
- 超参：chunk 256，10-bit 前缀表
- 种子：
- 环境：本机 CPU
- commit：
- 交付包 SHA256：本机运行，无交付包
- 结果：117,440,512 个指数、306,021,471 个有效 bit、458,752 个 chunk 全部计数。长于 10 bit 的码只占 0.058697%，bulk 表输出了 99.941303% 的权重，每次成功查表平均输出 3.415 个符号；长码分散在几乎所有 warp 组中。
- 记录：`results/decoder-longcode-stats.klxgqxg2/`
