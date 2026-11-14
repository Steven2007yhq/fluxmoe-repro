# 论文表3与图11：真实权重实验手册

本手册中的服务器命令由用户手动执行。当前已经核验 A40 tiny 正确性及Mixtral第0层的真实权重pilot（2/64个case），旧tree解码约4.07GB/s。Qwen、GLM及完整网格尚未运行。当前packed解码器的A40 A/B见[评审改进手册](CUDA_DECODER_REVIEW.md)。本机软件自检使用明确标注的随机 fixture；其图表与吞吐不是论文结果。

## 已核验的 A40 前置结果

[独立核验报告](/run/media/steve/Lenovo/FluxMoE论文复现/results/a40_server/a40-verify.IhdPv5/local_audit.md)：146项测试全部通过、无skip；重新计算的同机全驻留/分页对照通过；11请求/176生成token，176项生成与200项prompt logprob最大差值均为0。两侧各454项KV写入及165项历史SDPA核验无失败。原始文件未改写，文件SHA256及重算证据在同目录 `local_audit.json`。

## 本轮测量与论文的对应关系

图11使用原始BF16位，不转FP16/FP32、不量化、不实例化完整模型。Mixtral第0层的8个专家、Qwen第20层的512个专家中，所有gate/up/down矩阵都计入分布；统计exponent的256个符号和mantissa的128个符号，sign不混入mantissa。

表3的工作单元是一个MoE层、一个逻辑EP rank持有的权重分片。程序在**一张A40上顺序测量这些逻辑分片**，不进行多卡通信，也不声称已支持多卡推理。默认逻辑EP度数M/Q/G分别为2/4/8。完整网格分别为32×2、48×4、89×8个case。

Mixtral按论文的12.5% CPU专家字节预算排除后备：rank0首个本地专家的fused gate/up、rank1首个本地专家的down不参加GPU解码，两rank合计正好解码原始专家BF16体积的87.5%。论文未公布具体专家ID，此处采用首个本地专家并在metadata记录。Qwen/GLM测量全部routed专家。GLM checkpoint的layer92属于辅助MTP模块，依据config的num_hidden_layers=92排除；主模型MoE层为3..91，共89层。

当前GPU实现是独立的FMBF解码器：保留旧tree路径，默认packed路径采用批量前缀查表、warp转置与uint32打包读写。作者源码未公开；**不能预先承诺论文396–435 GB/s，也不能将A40结果按带宽比例换算成L40S/H20结果**。本机优化结果不能替代A40 A/B。

## 已固定的官方权重与下载量

下列不可变revision及所需safetensors分片均已通过官方Hub元数据核对。当前Mixtral layer0分片已在本机下载、校验并由用户上传服务器；其他计划尚未下载大权重。`plan`本身只读取小型索引/config和元数据。

| 计划 | revision | 下载量，十进制GB | 测量覆盖 |
| --- | --- | ---: | --- |
| `mixtral_layer0.json` | `eba92302a2861cdc0098cc54bc9f17cb2c47eb61` | 4.8928 | 图11 Mixtral完整代表层；表3仅2/64个case的预实验 |
| `qwen_layer20.json` | `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7` | 7.9997 | 图11 Qwen完整代表层；表3仅4/192个case的预实验 |
| `glm_layer3.json` | `cbb2c7cfb52fa128a9660cb1a7a78e017899e115` | 7.8713 | 可选表3 GLM单层预实验，8/712个case；图11原文无GLM面板 |
| `mixtral_all.json` | 同上Mixtral revision | 93.4057 | Mixtral表3全部32层×2逻辑rank |
| `qwen_all.json` | 同上Qwen revision | 159.3580 | Qwen表3全部48层×4逻辑rank |
| `glm_all.json` | 同上GLM revision | 703.6507 | GLM表3全部89层×8逻辑rank，按层下载以适配home磁盘 |

计划在 `configs/real_weight_plans/`。每个文件含tensor→shard映射、大小、Hub LFS SHA256；`fetch`检查下载上限、磁盘余量并核对每个实际文件的SHA256。默认不下载完整模型；选`all`才需要对应全层分片。完整GLM分片超过当前home盘可用空间，必须使用第7节的逐层命令或更大磁盘，不能把单层结果标成完整GLM行。

官方来源：[Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)、[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)、[GLM-4.5](https://huggingface.co/zai-org/GLM-4.5)、[Hub按revision下载](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)。论文未给checkpoint revision；这里明确固定所用官方快照。

## 1. 本机重新打包并上传

旧服务器目录尚无新增实验代码。以下命令在笔记本执行，包含新的C++准备器，排除本机二进制和历史results：

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'
mkdir -p "$HOME/installers/fluxmoe-realweights"
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.so' \
  -czf "$HOME/installers/fluxmoe-realweights/fluxmoe-realweights-src.tar.gz" \
  pyproject.toml README.md configs fluxmoe tests docs \
  native/huffman_decode.cu native/paged_arena.cpp native/cpu_codec.cpp \
  models/tiny_mixtral
cd "$HOME/installers/fluxmoe-realweights"
sha256sum fluxmoe-realweights-src.tar.gz > fluxmoe-realweights-src.tar.gz.sha256
scp -P 3168 fluxmoe-realweights-src.tar.gz fluxmoe-realweights-src.tar.gz.sha256 \
  bks-agent@115.154.139.236:/home/bks-agent/
```

## 2. 服务器新建项目与结果目录

导师已同意大文件先放服务器home；本手册使用 `~/fluxmoe-data`。这里是服务器磁盘，不是笔记本。保持原验证目录和结果不变：

```bash
ssh -p 3168 bks-agent@115.154.139.236
cd "$HOME"
sha256sum -c fluxmoe-realweights-src.tar.gz.sha256
mkdir -p "$HOME/projects"
FLUXMOE_WEIGHTS_PROJECT="$(mktemp -d "$HOME/projects/fluxmoe-realweights.XXXXXX")"
tar -xzf "$HOME/fluxmoe-realweights-src.tar.gz" -C "$FLUXMOE_WEIGHTS_PROJECT"
cd "$FLUXMOE_WEIGHTS_PROJECT"
source "$HOME/miniconda3-steve/etc/profile.d/conda.sh"
conda activate fluxmoe-steve

export FLUXMOE_DATA_ROOT="$HOME/fluxmoe-data"
mkdir -p "$FLUXMOE_DATA_ROOT"
test -w "$FLUXMOE_DATA_ROOT"
df -h "$FLUXMOE_DATA_ROOT"
export HF_HOME="$FLUXMOE_DATA_ROOT/hf-cache"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p results
FLUXMOE_WEIGHTS_RUN="$(mktemp -d "$PWD/results/weights.XXXXXX")"
export FLUXMOE_WEIGHTS_RUN
echo "$FLUXMOE_WEIGHTS_PROJECT"
echo "$FLUXMOE_WEIGHTS_RUN"
set -o pipefail
```

以下命令继续在同一服务器终端执行。每块成功后再继续，失败时保留日志和新结果目录，重试使用新输出目录。

## 3. 目标机编译与入口自检

选一张当前空闲GPU；示例为0号。新目录没有旧`.so`，需要重新构建。两个GPU模块与C++准备器都保存编译记录；Python构建脚本实际调用nvcc或g++。

```bash
nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export FLUXMOE_NVCC="$CUDA_HOME/bin/nvcc"
export FLUXMOE_CUDART="$CUDA_HOME/lib64/libcudart.so"
export FLUXMOE_CUDA_INCLUDE="$CUDA_HOME/include"
export FLUXMOE_CUDA_ARCH=sm_86
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VLLM_USE_FLASHINFER_SAMPLER=0
unset CUDA_LAUNCH_BLOCKING

python -m pip install 'matplotlib==3.10.8' 'numpy==2.3.5'
python -m pip check
python -m fluxmoe.fast_codec build --compiler /usr/bin/g++ \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/build-cpu.log"
python -m fluxmoe.cuda_codec build --nvcc "$FLUXMOE_NVCC" --architecture sm_86 \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/build-cuda.log"
python - <<'PY'
import os
from fluxmoe.cuda_vmm import build_cuda_vmm
print(build_cuda_vmm(cuda_include=os.environ['FLUXMOE_CUDA_INCLUDE'], compiler='/usr/bin/g++'))
PY
FLUXMOE_TEST_CUDA=1 python -m unittest tests.test_weight_experiments -v \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/new-entry-tests.log"
python -m pip freeze > "$FLUXMOE_WEIGHTS_RUN/pip-freeze.txt"
```

自检中的小型随机夹具只检验编码、索引、双stream计时和数值检查流程，不参与真实模型统计。GPU测量前确保所选卡没有其他计算任务；正式性能与profiler分开运行。每个case前后的GPU温度、功耗、频率和占用会记录在raw JSON中。

## 4. 首轮：Mixtral第0层

先查看计划，再下载约4.89GB到独立目录；`--max-download-gb 6`是这次允许的分片总容量上限，不含缓存和临时文件余量。

```bash
python - <<'PY'
import json
p=json.load(open('configs/real_weight_plans/mixtral_layer0.json'))
print(p['repo_id'], p['revision'], p['selected_layers'])
print('计划下载GB:', p['required_download_bytes']/1e9)
print(p['shards'])
PY
python -u -m fluxmoe.weight_experiments fetch \
  --plan configs/real_weight_plans/mixtral_layer0.json \
  --directory "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302" --max-download-gb 6 \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/download-mixtral.log"

python -u -m fluxmoe.weight_experiments analyze \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
  --layers 0 --output "$FLUXMOE_WEIGHTS_RUN/mixtral-analysis" \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/mixtral-analysis.log"

python -u -m fluxmoe.weight_experiments benchmark \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-l0-eba92302/download_manifest.json" \
  --warmup 5 --samples 20 --output "$FLUXMOE_WEIGHTS_RUN/mixtral-pilot" \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/mixtral-pilot.log"

python -m fluxmoe.weight_experiments report \
  --analysis "$FLUXMOE_WEIGHTS_RUN/mixtral-analysis/figure11.json" \
  --benchmark "$FLUXMOE_WEIGHTS_RUN/mixtral-pilot/summary.json" \
  --output "$FLUXMOE_WEIGHTS_RUN/mixtral-pilot-figures"
```

这会生成图11的Mixtral代表层面板，以及明确标为`partial_layers_pilot`的表3局部柱图。20个样本的P99仅用于调试观察，不作为稳定尾延迟结论。编码和索引准备可能需要一段时间，每个专家投影准备完都会打印进度。

## 5. 图11的另一个面板：Qwen第20层

此项是权重读取和解码微实验，不调用Qwen完整推理模型，因此不依赖尚未完成的Qwen vLLM分页接入。

```bash
python -u -m fluxmoe.weight_experiments fetch \
  --plan configs/real_weight_plans/qwen_layer20.json \
  --directory "$FLUXMOE_DATA_ROOT/qwen-l20-9c7f2fbe" --max-download-gb 9 \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/download-qwen.log"
python -u -m fluxmoe.weight_experiments analyze \
  --manifest "$FLUXMOE_DATA_ROOT/qwen-l20-9c7f2fbe/download_manifest.json" \
  --layers 20 --output "$FLUXMOE_WEIGHTS_RUN/qwen-analysis"
python -u -m fluxmoe.weight_experiments benchmark \
  --manifest "$FLUXMOE_DATA_ROOT/qwen-l20-9c7f2fbe/download_manifest.json" \
  --warmup 5 --samples 20 --output "$FLUXMOE_WEIGHTS_RUN/qwen-pilot"
python -m fluxmoe.weight_experiments report \
  --analysis "$FLUXMOE_WEIGHTS_RUN/mixtral-analysis/figure11.json" "$FLUXMOE_WEIGHTS_RUN/qwen-analysis/figure11.json" \
  --benchmark "$FLUXMOE_WEIGHTS_RUN/mixtral-pilot/summary.json" "$FLUXMOE_WEIGHTS_RUN/qwen-pilot/summary.json" \
  --output "$FLUXMOE_WEIGHTS_RUN/two-model-pilot-figures"
```

可选GLM微实验如下；这仅是8/712个case，不计作完整GLM表3复现：

```bash
python -u -m fluxmoe.weight_experiments fetch \
  --plan configs/real_weight_plans/glm_layer3.json \
  --directory "$FLUXMOE_DATA_ROOT/glm-l3-cbb2c7cf" --max-download-gb 10
python -u -m fluxmoe.weight_experiments benchmark \
  --manifest "$FLUXMOE_DATA_ROOT/glm-l3-cbb2c7cf/download_manifest.json" \
  --warmup 5 --samples 20 --output "$FLUXMOE_WEIGHTS_RUN/glm-pilot"
python -m fluxmoe.weight_experiments report \
  --analysis "$FLUXMOE_WEIGHTS_RUN/mixtral-analysis/figure11.json" "$FLUXMOE_WEIGHTS_RUN/qwen-analysis/figure11.json" \
  --benchmark "$FLUXMOE_WEIGHTS_RUN/mixtral-pilot/summary.json" "$FLUXMOE_WEIGHTS_RUN/qwen-pilot/summary.json" "$FLUXMOE_WEIGHTS_RUN/glm-pilot/summary.json" \
  --output "$FLUXMOE_WEIGHTS_RUN/three-model-pilot-figures"
```

## 6. 表3的Mixtral完整网格

先根据pilot实际准备耗时、样本波动估计总成本，再执行完整扫描。需要约93.41GB的独立checkpoint目录；和首轮分片目录会有少量重复。下例500样本是明确的测量配置，不是通用充分性保证。

```bash
python -u -m fluxmoe.weight_experiments fetch \
  --plan configs/real_weight_plans/mixtral_all.json \
  --directory "$FLUXMOE_DATA_ROOT/mixtral-all-eba92302" --max-download-gb 100
python -u -m fluxmoe.weight_experiments benchmark \
  --manifest "$FLUXMOE_DATA_ROOT/mixtral-all-eba92302/download_manifest.json" \
  --warmup 20 --samples 500 --output "$FLUXMOE_WEIGHTS_RUN/mixtral-full"
python -m fluxmoe.weight_experiments report \
  --analysis "$FLUXMOE_WEIGHTS_RUN/mixtral-analysis/figure11.json" "$FLUXMOE_WEIGHTS_RUN/qwen-analysis/figure11.json" \
  --benchmark "$FLUXMOE_WEIGHTS_RUN/mixtral-full/summary.json" \
  --output "$FLUXMOE_WEIGHTS_RUN/mixtral-full-figures"
```

## 7. 表3的Qwen、GLM完整网格

以下均在pilot正确性通过后再运行。Qwen的全层分片约159.36GB；与上述Mixtral及三个pilot目录一起，权重总量约274GB。开始前重新查看磁盘可用量；`fetch`也会核验余量。

```bash
df -h "$FLUXMOE_DATA_ROOT"
python -u -m fluxmoe.weight_experiments fetch \
  --plan configs/real_weight_plans/qwen_all.json \
  --directory "$FLUXMOE_DATA_ROOT/qwen-all-9c7f2fbe" --max-download-gb 170 \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/qwen-full-download.log"
python -u -m fluxmoe.weight_experiments benchmark \
  --manifest "$FLUXMOE_DATA_ROOT/qwen-all-9c7f2fbe/download_manifest.json" \
  --warmup 20 --samples 500 --output "$FLUXMOE_WEIGHTS_RUN/qwen-full" \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/qwen-full.log"
```

GLM采用**下载一层→逐分片SHA256核验→测量8个逻辑rank→保留记录→清理本次临时权重**的顺序，总计712个case。官方分片的全层总下载约703.65GB，每层最多10.98GB；GLM各层所需分片互不重复，逐层方式不增加计划中的权重下载量。预留至少15GB临时磁盘空间。下载上限是显式预算，`--max-total-download-gb 710`不是会立即占满的磁盘容量。

```bash
df -h "$FLUXMOE_DATA_ROOT"
HF_HUB_DISABLE_XET=1 python -u -m fluxmoe.weight_experiments stream \
  --plan configs/real_weight_plans/glm_all.json \
  --data-root "$FLUXMOE_DATA_ROOT" \
  --max-layer-download-gb 12 --max-total-download-gb 710 \
  --warmup 20 --samples 500 \
  --output "$FLUXMOE_WEIGHTS_RUN/glm-full-stream" \
  2>&1 | tee "$FLUXMOE_WEIGHTS_RUN/glm-full-stream.log"
```

`stream`只删除其在data-root中新建的临时目录，不删除已有checkpoint；成功或失败都会保留当层download plan/manifest和已写入的benchmark记录。出错后停止扩大实验，日志与failed case保留，不自动隐藏重试。重新运行需新输出目录；已通过的层可用同一revision生成剩余层plan，完成后用 `merge --inputs .../summary.json ... --output 新目录` 合并。合并校验实现、设备、软件、采样协议及原始记录hash，并拒绝重复或缺失rank。

`HF_HUB_DISABLE_XET=1`在启动Python前设置，使用Hub的HTTP下载路径；权重通过`local_dir`写入临时checkpoint，不额外生成完整中央模型缓存。不同版本中旧的`HF_XET_CHUNK_CACHE_SIZE_BYTES`行为不同，因此不依赖它控制空间。参考[官方环境变量说明](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)。

完整三行的绘图命令如下，三个输入分别必须是64/192/712个case，图11仍使用此前两个代表层分析：

```bash
python -m fluxmoe.weight_experiments report \
  --analysis "$FLUXMOE_WEIGHTS_RUN/mixtral-analysis/figure11.json" "$FLUXMOE_WEIGHTS_RUN/qwen-analysis/figure11.json" \
  --benchmark "$FLUXMOE_WEIGHTS_RUN/mixtral-full/summary.json" "$FLUXMOE_WEIGHTS_RUN/qwen-full/summary.json" "$FLUXMOE_WEIGHTS_RUN/glm-full-stream/combined/summary.json" \
  --output "$FLUXMOE_WEIGHTS_RUN/three-model-full-figures"
```

## 计时、数值与统计口径

- C++ CPU准备器生成与原Python FMBF编码器逐字节兼容的流，并校验CRC、完整前缀树、所有符号和chunk offsets。该准备工作、GPU持久源上传、输出分配都发生在计时前。
- gate/up与down分别使用独立CUDA stream；公共start event由两stream等待；末尾control stream等待两stream完成event后记录end。latency覆盖整个两stream完成窗口，含CPU发射造成的空隙；不是把各kernel时间相加。
- 正确性在warmup前和测量后分别对照**原始BF16字节**，检查不在计时循环内；任何错误都终止并保留failed case，不绘成0吞吐。
- 每case保留全部`samples_ms`；throughput为未压缩输出bytes除以该sample的CUDA事件时间（十进制GB/s），case报告样本吞吐均值。P99使用NumPy线性插值percentile。表中mean/std在所选layer/rank的case汇总值上计算，std不是95%置信区间。
- `storage_saved_container_percent`包含FMBF文件头/码长表；`storage_saved_device_source_percent`额外考虑实际GPU后备的索引和解码树。输出物化buffer单列，不把单层存储比例当整模型显存节省。
- 完整网格覆盖仅表示层/rank齐全；仍需检查样本量、独立进程重复、温度/频率/其他任务和噪声，才能作稳健性能结论。论文表3为mean±std，不能把主实验的五次运行95%CI规则直接冒充此表统计。

## 原始产物与重画

每个benchmark目录包含`metadata.json`、逐case的`cases.jsonl`、`summary.json`及`build_records/`。每个case保留原始/压缩SHA256、输出体积、设备后备字节、真实计时样本及两次逐位校验结果。出图入口重新读取原始cases，核对hash并重算统计；跨NumPy版本重画仅允许极小舍入差异。

出图目录包含`figure11.png/.pdf/.csv`、`table3.png/.pdf/.csv`及`sources.json`。所有A40图标注实际设备、独立decoder和局部/完整覆盖；论文数值不混入实测列。约400GB/s若未达到，应据真实数据分析，不调整工作量或计时边界迎合论文。

测完可在笔记本手动拷回小型结果目录（不拷大权重）：

```bash
mkdir -p '/run/media/steve/Lenovo/FluxMoE论文复现/results/real_weight_server'
scp -r -P 3168 \
  'bks-agent@115.154.139.236:/home/bks-agent/projects/fluxmoe-realweights.*/results/weights.*' \
  '/run/media/steve/Lenovo/FluxMoE论文复现/results/real_weight_server/'
```
