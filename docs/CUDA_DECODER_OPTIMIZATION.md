# A40 原始数据核验与 CUDA 解码优化

> 本文记录上一轮bulk版本；新版默认packed、评审处置与手动命令见[CUDA_DECODER_REVIEW.md](CUDA_DECODER_REVIEW.md)。本文既有数字和源码归档保持原实验口径。
本轮服务器操作均由用户手动执行。原始 A40 结果保持不变，见
`results/real_weight_server/mixtral-offline.FqxVN6/`；本机诊断、重复测试、时间线及核验记录在 `results/decoder_optimization/`。

## 原始数据核验

已重新读取原始 `cases.jsonl`，校验它及 metadata 的 SHA256，并重算 mean、empirical P99、吞吐与跨 case 标准差。两组各 20 样本，warmup=5，全部通过测量前后的原始 BF16 逐位检查。

| 指标 | A40 原始记录 |
| --- | ---: |
| rank0 解码体积 | 1,174,405,120 bytes |
| rank0 平均延迟 | 约 288 ms |
| rank1 解码体积 | 1,291,845,632 bytes |
| rank1 平均延迟 | 约 318 ms |
| 两个 case 的吞吐均值 | 4.070244985 GB/s |
| 压缩容器节省 | 33.7051% |
| 含索引、树等的 GPU 后备节省 | 32.1423% |

两 rank 因 Mixtral CPU 后备投影不同而解码体积不同，故其延迟差不能直接解释为测量噪声。原始 `std=20.77 ms` 是两种 case 的均值之间的标准差，不是同一 case 的运行抖动或置信区间。20 个样本不能支撑稳定尾延迟结论。

服务器记录的四个 Python 实现文件哈希与优化前源码一致；native 构建记录为 nvcc13.2、`sm_86`、`-O3`。GPU 快照显示 P0、1740MHz，没有低频证据，但快照不能排除整个窗口内的其他进程。服务器 ELF 未回传，其记录的 artifact hash 尚不能独立复核。

在本机用同 revision 的原始 BF16 权重重新统计：24 个 gate/up/down 矩阵、1,409,286,144 个 BF16 值，tensor SHA256、直方图、熵全部与回传图11记录一致。指数熵约2.5541 bits，尾数熵约6.9729 bits。详细证据为 `a40_audit.json` 及 `recomputed_analysis/figure11.json`。

## 瓶颈与实现

原 kernel 的每个线程顺序解一个256元素chunk，每个元素又逐bit遍历全局内存里的Huffman树。同一warp相邻线程的BF16输出地址相隔512 bytes，读写与串行依赖均不适合GPU。单纯替换树查询收益有限，连续写回的消融收益更大。

本机 RTX4060 的 Nsight Systems 时间线显示，旧 decoder kernel 占该 CUDA event 窗口约99.8%，各解码窗口没有H2D/D2H；因此在该复现案例中，GPU kernel本身是主要耗时。不能把这份本机时间线冒充A40 profiler证据。

保留四条显式路径，默认使用`bulk`：

| `--decoder` | 路径 |
| --- | --- |
| `tree` | 保留旧GPU逐位树遍历kernel，作为同机对照 |
| `lookup` | 10-bit前缀查表，保留原访存布局 |
| `tiled` | 查表加warp内32×32共享内存转置，连续输出 |
| `bulk` | 一次查表最多解4个短码；批量消耗查询结果；连续读sign/mantissa并转置输出 |

超过10-bit的码字仍走精确树回退。没有改动BF16位、FMBF容器、Huffman码字或256元素chunk的默认索引粒度。GPU后备统计额外计入每个源的12KiB前缀表和8-byte安全lookahead padding；输出buffer仍单列。这是源数组请求分配字节，非整个进程显存峰值；CUDA上下文、驱动页粒度与框架reserved开销需结合GPU快照另看。旧`tree`消融也保留这些表分配，避免在同一二进制里混入不同生命周期实现。

边界测试覆盖全部65,536种BF16位模式、尾块、chunk1..65536的代表边界、最长255-bit码字、输出尾部哨兵、非默认stream和多stream生命周期。CUDA Graph capture仍显式拒绝。

优化未承诺达到论文401.9GB/s。当前目标是保持逐位正确性，在同机同工作量下改善吞吐；A40提升必须由以下手动A/B实测确认。

本机最终单投影测试为3个独立进程，每条路径各30个样本、warmup10，tree约3.196GB/s，bulk约10.473GB/s，延迟比约3.277x。整层双stream也逐位通过，但旧路径出现2–3秒长尾；所有样本保留，不将其均值比作为稳定加速结论。额外整层Nsight采集再次观察到旧kernel的elapsed执行时间长尾，主机launch API没有同量级停顿；具体频率/调度原因尚缺高频遥测证据。此处的本机长尾不用于解释A40原始稳定的4GB/s。

限制循环展开使本机bulk kernel的寄存器数从197降至37，无spill；当前单投影吞吐没有进一步显著变化，保留较低资源占用，不把它单独计作额外加速。163项回归和完整vLLM生成对照通过。本机NVCC13.4已交叉编译SM86目标，这不替代服务器NVCC13.2编译和实际A40验证。

## 1. 笔记本上传新源码包

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现/results/decoder_optimization/delivery'
scp -P 3168 fluxmoe-decoder-opt-src.tar.gz fluxmoe-decoder-opt-src.tar.gz.sha256 \
  bks-agent@115.154.139.236:/home/bks-agent/
```

## 2. 服务器新目录编译

以下命令在服务器运行。保留旧项目和实验结果。当前只需在现有`fluxmoe-steve`环境重新编译CPU编码准备器与CUDA解码库。

```bash
ssh -p 3168 bks-agent@115.154.139.236
```

登录完成后再执行下一块：

```bash
cd "$HOME"
sha256sum -c fluxmoe-decoder-opt-src.tar.gz.sha256
mkdir -p "$HOME/projects"
FLUXMOE_OPT_PROJECT="$(mktemp -d "$HOME/projects/fluxmoe-decoder-opt.XXXXXX")"
tar -xzf "$HOME/fluxmoe-decoder-opt-src.tar.gz" -C "$FLUXMOE_OPT_PROJECT"
cd "$FLUXMOE_OPT_PROJECT"
source "$HOME/miniconda3-steve/etc/profile.d/conda.sh"
conda activate fluxmoe-steve
nvidia-smi
```

选择当前空闲卡，以下示例为0号：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export FLUXMOE_NVCC="$CUDA_HOME/bin/nvcc"
export FLUXMOE_CUDART="$CUDA_HOME/lib64/libcudart.so"
export FLUXMOE_CUDA_ARCH=sm_86
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export FLUXMOE_DATA_ROOT="$HOME/fluxmoe-data"
unset CUDA_LAUNCH_BLOCKING
mkdir -p results
FLUXMOE_OPT_RUN="$(mktemp -d "$PWD/results/decoder-opt.XXXXXX")"
export FLUXMOE_OPT_RUN
echo "$FLUXMOE_OPT_RUN"
set -o pipefail

python -m fluxmoe.fast_codec build --compiler /usr/bin/g++ \
  2>&1 | tee "$FLUXMOE_OPT_RUN/build-cpu.log"
python -m fluxmoe.cuda_codec build --nvcc "$FLUXMOE_NVCC" --architecture sm_86 \
  2>&1 | tee "$FLUXMOE_OPT_RUN/build-cuda.log"
FLUXMOE_TEST_CUDA=1 python -m unittest tests.test_cuda_async_codec tests.test_weight_experiments -v \
  2>&1 | tee "$FLUXMOE_OPT_RUN/tests.log"
python -m pip freeze > "$FLUXMOE_OPT_RUN/pip-freeze.txt"
```

每块成功后再继续。新库扩展了C ABI，须重新编译；不要复用旧`.so`。

## 3. 一张真实专家投影的诊断消融

此项取layer0/expert0的fused gate/up（234,881,024 bytes），逐条测试四种decoder，并加入纯设备拷贝对照。它是单stream、单投影诊断，不是表3整层结果。准备、源上传、分配和正确性检查在计时外；每个run测量前后均对照原始BF16。

```bash
python -u -m fluxmoe.decoder_bench \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
  --cache "$FLUXMOE_DATA_ROOT/decoder-cache" \
  --warmup 10 --samples 50 --runs 3 \
  --output "$FLUXMOE_OPT_RUN/projection-ablation" \
  2>&1 | tee "$FLUXMOE_OPT_RUN/projection-ablation.log"
```

该入口的`runs=3`是同进程重复并交替正反顺序，不是3个独立进程；可使用新输出目录分别启动3次获得独立进程数据。缓存只是离线压缩文件，按raw/container双SHA256校验，不替代GPU解码。

## 4. 表3对应的同机、整层、双stream A/B

重新测旧kernel可核对之前约4GB/s是否仍成立。两条命令保持同chunk=256、同权重、同GPU与同采样配置，先后执行。

```bash
python -u -m fluxmoe.weight_experiments benchmark \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
  --decoder tree --chunk-size 256 --warmup 10 --samples 50 \
  --output "$FLUXMOE_OPT_RUN/layer-tree" \
  2>&1 | tee "$FLUXMOE_OPT_RUN/layer-tree.log"
```

```bash
python -u -m fluxmoe.weight_experiments benchmark \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
  --decoder bulk --chunk-size 256 --warmup 10 --samples 50 \
  --output "$FLUXMOE_OPT_RUN/layer-bulk" \
  2>&1 | tee "$FLUXMOE_OPT_RUN/layer-bulk.log"
```

两者最终都必须为`status: passed`。保留每个case的原始样本，检查共享负载、频率与温度；50样本P99仍只用于pilot，不能据此宣布稳定尾延迟。

## 5. 生成同机对照图并拷回

```bash
python -u -m fluxmoe.weight_experiments analyze \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
  --layers 0 --output "$FLUXMOE_OPT_RUN/analysis"
python -m fluxmoe.weight_experiments report \
  --analysis "$FLUXMOE_OPT_RUN/analysis/figure11.json" \
  --benchmark "$FLUXMOE_OPT_RUN/layer-tree/summary.json" "$FLUXMOE_OPT_RUN/layer-bulk/summary.json" \
  --output "$FLUXMOE_OPT_RUN/figures"
```

`table3.csv`和图中会标明decoder，避免把两条同名模型的柱混淆。覆盖仍是2/64个case。确认A/B后，再推进完整网格；使用原手册的全层入口时默认选择新`bulk`。其中`benchmark`也可显式传`--decoder tree`保留旧路径；逐层`stream`入口当前使用默认bulk。

可选时间线采集：若服务器已安装`nsys`，在单独输出目录运行下面命令。该结果用于诊断，不替代上述无profiler性能样本。

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output "$FLUXMOE_OPT_RUN/profile" \
  python -u -m fluxmoe.decoder_bench \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
  --cache "$FLUXMOE_DATA_ROOT/decoder-cache" \
  --warmup 2 --samples 5 --runs 1 \
  --output "$FLUXMOE_OPT_RUN/profiled-ablation"
```

最后在笔记本执行（不复制大权重和编码缓存）：

```bash
mkdir -p '/run/media/steve/Lenovo/FluxMoE论文复现/results/decoder_opt_server'
scp -r -P 3168 \
  'bks-agent@115.154.139.236:/home/bks-agent/projects/fluxmoe-decoder-opt.*/results/decoder-opt.*' \
  '/run/media/steve/Lenovo/FluxMoE论文复现/results/decoder_opt_server/'
```

## 参考与边界

- [NVIDIA coalesced access说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)：用于解释访存布局；不以理论峰值换算实验成绩。
- [原始表3/图11实验手册](REAL_WEIGHT_EXPERIMENTS.md)：权重revision、EP逻辑分片和CPU后备口径。
- A40优化后吞吐、跨设备加速比及论文性能达成情况均须等待真实服务器A/B；本机结果只能支持RTX4060上的已测条件。
