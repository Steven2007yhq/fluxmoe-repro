# FMBF 评审建议的实现与 A40 手动验证

本轮依据用户提供的《FMBF Huffman 解码 CUDA 代码评审》核对当前代码。评审中的性能判断原本来自静态阅读；本轮用真实 Mixtral 权重做本机对照。结果、原始样本和重算脚本保存在 `results/decoder_review/`。服务器操作全部由用户手动执行。

本机RTX4060的三次独立进程对照：旧bulk 9.575GB/s，packed 14.242GB/s，延迟比1.488x；buffered 9.060GB/s，组合路径14.007GB/s。166项回归及完整vLLM生成对照通过。整层双stream两侧均有长尾，所有样本保留，不将其均值比作为稳定加速主张。详细数据见 `results/decoder_review/report.md`。

## 采纳与暂缓

| 建议 | 处理与边界 |
| --- | --- |
| 位流寄存器缓存 | 新增 `buffered`：缓存两个 MSB-first 32-bit word，跨 word 时补充；长码跳转后重新定位。不修改 FMBF 字节顺序。预实验未观察到收益，保留作消融。 |
| 向量化 sign/mantissa 读取与输出 | 新增 `packed`：对齐的完整 warp tile 使用 uint32 读写，输入一次取4个字节，输出一次写2个BF16；尾部、非4倍数chunk和仅2字节对齐的输出使用标量回退。默认选择此路径。 |
| 合并两项优化 | `vectorized` = 位流缓存 + packed IO，作为组合消融；它不意味着默认优于 `packed`。 |
| 16-bit shared tile padding | packed 路径用34列uint16，即每行17个32-bit bank宽度，允许对齐的uint32读写；旧路径保留33列。实际bank conflict未用Nsight Compute测量，不以地址推导代替计数器证据。 |
| 完整块快路径与尾部 | 向量IO仅在完整tile启用；packed路径每tile计算一次有效输出上限，减少逐符号边界判断。保留位流结束检查与最长255-bit回退。 |
| 128硬编码 | 统一 `BLOCK_THREADS/BLOCK_WARPS`，加入static_assert与launch_bounds，索引用blockDim.x。 |
| 单符号码树 | 现有编码端保证至少1 bit。增加单符号GPU测试；原生C ABI明确拒绝零位根叶子，避免宽度0表项歧义。 |
| 错误标志 | 保留跨stream粘性标志，不在热路径清零；显式check检测到失败后将source标记为不可再提交，需重建。加入损坏码流、越界/截断offset故障注入与输出哨兵测试。它是结构检查，不是每次GPU输出CRC。 |
| CUDA last-error | 使用cudaPeekAtLastError，报告但不清除同一runtime实例的已有错误；不能据此断言错误一定来自本次kernel。 |
| 主机padding副本 | 直接分配额外8字节GPU空间并清零尾部，不再复制整份主机码流构建padding。上传仍在计时外。 |
| 死分支 | 清理不可达Bulk三元表达式，合并连续Bulk分支。 |
| LSB-first、交错码流、rANS、固定码长异常表 | 涉及格式或编码机制改变，本轮不混入原FMBF复现。 |
| cp.async、8–12位窗口扫描、循环展开 | 暂未采用；本轮固定10位表和chunk=256，隔离已测改动，避免重现上一轮寄存器膨胀。 |
| CUDA Graph | 继续明确拒绝。仅预建event不足以解决图重放、source生命周期和VMM映射失效问题，需要独立设计。 |

`tree/lookup/tiled/bulk`保留算法消融。旧bulk算法在本轮同一二进制中作为对照，共用参数检查和launch配置修复。本轮比率以本轮原始样本重算，不与上一轮不同时间采集的数据拼接。CPU准备、分配、上传及逐位检查均在解码计时外。

本机未找到 `compute-sanitizer` 和 `ncu`。普通回归不替代memcheck/racecheck；Nsight Systems只能确认时间线、kernel资源和实际路径，不提供本轮未测的stall、实际occupancy、sectors/request或bank conflict结论。

## 1. 笔记本上传新版源码

只上传源码，继续复用服务器已校验的4.89GB权重分片。

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现/results/decoder_review/delivery'
scp -P 3168 fluxmoe-decoder-review-src.tar.gz fluxmoe-decoder-review-src.tar.gz.sha256 \
  bks-agent@115.154.139.236:/home/bks-agent/
ssh -p 3168 bks-agent@115.154.139.236
```

## 2. 以下命令在登录后的服务器终端运行

```bash
cd "$HOME"
sha256sum -c fluxmoe-decoder-review-src.tar.gz.sha256
mkdir -p "$HOME/projects"
FLUXMOE_REVIEW_PROJECT="$(mktemp -d "$HOME/projects/fluxmoe-decoder-review.XXXXXX")"
tar -xzf "$HOME/fluxmoe-decoder-review-src.tar.gz" -C "$FLUXMOE_REVIEW_PROJECT"
cd "$FLUXMOE_REVIEW_PROJECT"
source "$HOME/miniconda3-steve/etc/profile.d/conda.sh"
conda activate fluxmoe-steve
nvidia-smi
```

下例选择GPU 0；先确认它空闲，如正在使用则改成空闲GPU编号。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export FLUXMOE_NVCC="$CUDA_HOME/bin/nvcc"
export FLUXMOE_CUDART="$CUDA_HOME/lib64/libcudart.so"
export FLUXMOE_CUDA_INCLUDE="$CUDA_HOME/include"
export FLUXMOE_CUDA_ARCH=sm_86
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export FLUXMOE_DATA_ROOT="$HOME/fluxmoe-data"
unset CUDA_LAUNCH_BLOCKING
set -o pipefail
mkdir -p results
FLUXMOE_REVIEW_RUN="$(mktemp -d "$PWD/results/decoder-review.XXXXXX")"
export FLUXMOE_REVIEW_RUN
echo "$FLUXMOE_REVIEW_RUN"
python -m pip freeze > "$FLUXMOE_REVIEW_RUN/pip-freeze.txt"
nvidia-smi > "$FLUXMOE_REVIEW_RUN/nvidia-smi.txt"
python -m fluxmoe.fast_codec build --compiler /usr/bin/g++ 2>&1 | tee "$FLUXMOE_REVIEW_RUN/build-cpu.log"
python -m fluxmoe.cuda_codec build --nvcc "$FLUXMOE_NVCC" --architecture sm_86 2>&1 | tee "$FLUXMOE_REVIEW_RUN/build-cuda.log"
python -c 'from fluxmoe.cuda_vmm import build_cuda_vmm; print(build_cuda_vmm())' 2>&1 | tee "$FLUXMOE_REVIEW_RUN/build-vmm.log"
FLUXMOE_TEST_CUDA=1 FLUXMOE_TEST_VLLM=1 python -m unittest discover -s tests -v 2>&1 | tee "$FLUXMOE_REVIEW_RUN/tests.log"
```

编译或测试失败时先保留日志，停止后面的性能测试。此处新增模式必须重新编译CUDA模块；旧so不兼容新版source哈希。

## 3. 三个独立进程的真实权重对照

四条路径构成旧bulk、仅缓存、仅打包、缓存加打包的对照。第二次反转次序。每个进程各路径warmup10、采样30；全部保留，不去掉慢样本。

```bash
for run in 0 1 2; do
  modes=(bulk buffered packed vectorized)
  if [ "$run" = 1 ]; then modes=(vectorized packed buffered bulk); fi
  python -u -m fluxmoe.decoder_bench \
    --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
    --cache "$FLUXMOE_DATA_ROOT/decoder-cache" \
    --modes "${modes[@]}" --warmup 10 --samples 30 --runs 1 \
    --output "$FLUXMOE_REVIEW_RUN/projection-$run" \
    2>&1 | tee "$FLUXMOE_REVIEW_RUN/projection-$run.log" || break
done
```

## 4. 表3对应的整层双stream与vLLM正确性

```bash
for mode in bulk packed; do
  python -u -m fluxmoe.weight_experiments benchmark \
    --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
    --decoder "$mode" --chunk-size 256 --warmup 10 --samples 50 \
    --output "$FLUXMOE_REVIEW_RUN/layer-$mode" \
    2>&1 | tee "$FLUXMOE_REVIEW_RUN/layer-$mode.log" || break
done
python -u -m fluxmoe.engine_validation --output "$FLUXMOE_REVIEW_RUN/vllm-engine" \
  2>&1 | tee "$FLUXMOE_REVIEW_RUN/vllm-engine.log"
```

vLLM默认走packed解码；这里仍是随机tiny Mixtral流程正确性，不是预训练模型质量。整层仅Mixtral第0层两个逻辑rank，即2/64 cases，不是完整表3。

```bash
python -u -m fluxmoe.weight_experiments analyze \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
  --layers 0 --output "$FLUXMOE_REVIEW_RUN/analysis"
python -m fluxmoe.weight_experiments report \
  --analysis "$FLUXMOE_REVIEW_RUN/analysis/figure11.json" \
  --benchmark "$FLUXMOE_REVIEW_RUN/layer-bulk/summary.json" "$FLUXMOE_REVIEW_RUN/layer-packed/summary.json" \
  --output "$FLUXMOE_REVIEW_RUN/figures"
```

## 5. 可选sanitizer验证

服务器若已有 `compute-sanitizer`，在正确性阶段运行；未安装时记录未执行，不将普通测试的OK视作替代。这些命令不用于计时。

```bash
command -v compute-sanitizer
FLUXMOE_TEST_CUDA=1 compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m unittest tests.test_cuda_async_codec.PersistentSourceCudaTests.test_all_decoders_match_all_bf16_patterns_and_unaligned_chunk_tails \
  tests.test_cuda_async_codec.PersistentSourceCudaTests.test_gpu_malformed_stream_and_offsets_are_sticky_and_poison_source \
  tests.test_cuda_async_codec.PersistentSourceCudaTests.test_single_symbol_tree_and_two_byte_aligned_destination -v \
  2>&1 | tee "$FLUXMOE_REVIEW_RUN/memcheck.log"
FLUXMOE_TEST_CUDA=1 compute-sanitizer --tool racecheck --error-exitcode 99 \
  python -m unittest tests.test_cuda_async_codec.PersistentSourceCudaTests.test_all_decoders_match_all_bf16_patterns_and_unaligned_chunk_tails \
  tests.test_cuda_async_codec.PersistentSourceCudaTests.test_gpu_malformed_stream_and_offsets_are_sticky_and_poison_source -v \
  2>&1 | tee "$FLUXMOE_REVIEW_RUN/racecheck.log"
```

## 6. 回到笔记本终端拷回结果

```bash
mkdir -p '/run/media/steve/Lenovo/FluxMoE论文复现/results/decoder_review_server'
scp -r -P 3168 \
  'bks-agent@115.154.139.236:/home/bks-agent/projects/fluxmoe-decoder-review.*/results/decoder-review.*' \
  '/run/media/steve/Lenovo/FluxMoE论文复现/results/decoder_review_server/'
```

核对依据：[CUDA Runtime错误状态语义](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__ERROR.html)、[CUDA访存与测量指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[Compute Sanitizer工具说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。本机工具版本与编译哈希以每次运行的build_records为准。
