# A40 单卡迁移与正确性验收

本文提供迁移步骤和验收条件，不记录尚未完成的 A40 实测结论。此前 4060 的正确性结果不能代替目标服务器验证。初次迁移限定一张 A40、BF16 tiny Mixtral、vLLM 0.29.0、eager、Triton，沿用现有代码明确支持的范围。

服务器上的命令由用户手动执行。本文中的打包、上传后的构建及测试步骤均为操作说明，不表示已经上传、构建或运行。

## 迁移内容

保留目录结构，打包以下文件即可执行全部相关验证：

- `pyproject.toml`、`README.md`、`configs/`。
- `fluxmoe/`、`tests/`，排除 `__pycache__/` 和 `*.pyc`。
- `native/huffman_decode.cu`、`native/paged_arena.cpp`。
- `models/tiny_mixtral/` 的全部文件，包括 `fixture_manifest.json`、safetensors、配置和 tokenizer。
- `docs/` 可一同保留，便于核对流程及限制。

使用已验证的 tiny checkpoint 文件，才能继续比较同一工件；不需要下载预训练模型或重新生成随机权重。该 checkpoint 约 2 MiB，不包含大型模型。

不要迁移 `native/build/` **或项目根目录的 `build/`**：前者含 4060 解码库，后者含 VMM 库及本机路径记录。也不迁移旧 `results/`、Python 环境、缓存和论文 PDF。源文件打包后保存包的 SHA256；目标端核对包 hash，再在新目录解包，避免覆盖已有实验。

以下是本地打包示例，输出路径可由实际任务指定：

```bash
tar --exclude='__pycache__' --exclude='*.pyc' \
  -czf /tmp/fluxmoe-a40-source.tar.gz \
  pyproject.toml README.md configs fluxmoe tests docs \
  native/huffman_decode.cu native/paged_arena.cpp models/tiny_mixtral
sha256sum /tmp/fluxmoe-a40-source.tar.gz
```

## 环境前提

使用服务器专用 Python 3.12 环境，其直接依赖见 `configs/requirements-a40-py312.txt`。Python 3.12 使用 `numpy==2.3.5`；本机 Python 3.14 环境中的 `numpy==2.4.6` 不满足 Mistral Common 的条件依赖。

在项目根目录、目标 Conda 环境中先检查：

```bash
python -m pip check
python -c 'import sys, torch, vllm, transformers; print(sys.executable); print(sys.version); print(torch.__version__, torch.version.cuda); print(vllm.__version__, transformers.__version__)'
```

预期 Python 3.12、PyTorch 2.13.0/cu130、vLLM 0.29.0、Transformers 5.17.0。此检查只证明元数据和导入；基础 GPU 算子、VMM 和真实生成仍需后续验证。保存实际 `pip freeze`、驱动、Toolkit、GPU UUID/型号、SM 和测试时间；不要把推荐版本当作已经安装的版本。

vLLM 0.29 的 CUDA 原生模块使用稳定 libtorch ABI 命名；`import vllm._C` 不是该版本 CUDA wheel 的正确验收方式。以下命令严格加载两个必要扩展并检查算子注册，但不执行 GPU kernel：

```bash
python - <<'PY'
import importlib
import torch

for name in ('vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch'):
    module = importlib.import_module(name)
    print(name, module.__file__)
print('rms_norm:', torch.ops._C.rms_norm)
print('topk_softmax:', torch.ops._moe_C.topk_softmax)
PY
```

Python 扩展模块名改变了，`torch.ops` 中的算子命名空间仍为 `_C` 和 `_moe_C`。不要仅调用 `current_platform.import_kernels()` 就宣称验收通过：该方法会记录或抑制部分 `ImportError`，上述直接导入会让失败明确退出。扩展加载成功之后才继续实际算子及完整引擎验证。

## 在目标机重建原生库

服务器既有审计确认 `/usr/local/cuda-13.2` 为系统 Toolkit。执行时仍应检查指定路径和版本：

```bash
export FLUXMOE_NVCC=/usr/local/cuda-13.2/bin/nvcc
export FLUXMOE_CUDART=/usr/local/cuda-13.2/lib64/libcudart.so
export FLUXMOE_CUDA_INCLUDE=/usr/local/cuda-13.2/include
export FLUXMOE_CUDA_ARCH=sm_86
test -x "$FLUXMOE_NVCC"
test -f "$FLUXMOE_CUDART"
test -f "$FLUXMOE_CUDA_INCLUDE/cuda.h"
"$FLUXMOE_NVCC" --version
g++ --version

python -m fluxmoe.cuda_codec build \
  --nvcc "$FLUXMOE_NVCC" --architecture sm_86
python - <<'PY'
import os
from fluxmoe.cuda_vmm import build_cuda_vmm
print(build_cuda_vmm(cuda_include=os.environ['FLUXMOE_CUDA_INCLUDE']))
PY
```

解码库输出 `native/build/libfluxmoe_huffman.so`；VMM 库输出 `build/libfluxmoe_vmm.so`。两者的相邻 JSON 和构建日志均须保存。源码或工具链改变后重新构建，并启动新的 Python 测试进程。

如需自定义输出位置，解码 CLI 支持 `--output-dir`，VMM 函数支持 `output_path=`；运行时分别用 `FLUXMOE_CUDA_LIBRARY` 和 `FLUXMOE_VMM_LIBRARY` 指定绝对路径，连同相邻构建 JSON 使用。

## 验证顺序

先核对选定 GPU 当前可用，再设置 `CUDA_VISIBLE_DEVICES` 为该卡；进程内设备编号从 0 开始。不要因为服务器有四张卡就启用 TP/EP；当前 adapter 明确只支持单 rank。

1. 先完成环境导入和少量 BF16 GPU 基础算子检查，记录实际设备及 CUDA 可用性。
2. 检查同一份 checkpoint 的文件 hash 和 CPU 测试。
3. 完成原生解码、VMM Tensor 别名和异步分页回归。
4. 运行全驻留/分页两个独立进程的完整引擎对照。

示例入口如下；每次执行使用新的结果目录，并保存命令、退出码和完整 stdout/stderr：

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m fluxmoe.checkpoint models/tiny_mixtral
python -m unittest tests.test_checkpoint tests.test_engine_checks tests.test_engine_integration -v

FLUXMOE_TEST_CUDA=1 python -m unittest \
  tests.test_cuda_codec tests.test_cuda_async_codec \
  tests.test_cuda_vmm tests.test_cuda_vmm_torch tests.test_cuda_pager -v

python -m fluxmoe.async_validation --output results/a40_async_run01
python -m fluxmoe.engine_validation \
  --model models/tiny_mixtral --output results/a40_engine_run01
```

明确启用的 GPU 测试不得通过跳过来验收。完整引擎对照需要检查 `comparison.json` 为 `passed`，并核对两个原始 JSON 和日志：checkpoint 专家 hash、相同生成 token/停止原因、prompt 与生成 logprob、KV slot 写入与历史 attention、chunked prefill、prefix hit、分页解码/H2D/RAW/WAR 计数、移除诊断 hooks 后的生成，以及清理状态。

KV 观察 hooks 包含同步，tiny 测试也包含额外 oracle 检查；这些运行时间不能作为吞吐或加速比。通过单卡 tiny 测试仅支持目标环境上的流程和数值正确性，不证明大模型容量、多卡通信、质量或论文性能。

## CUDA 与 Conda 的边界

- 系统 Toolkit 13.2 与 PyTorch wheel 内置 CUDA 13.0 runtime 是不同来源。解码库显式记录 Toolkit 的 `libcudart`，而同进程动态链接还可能复用已加载的同 SONAME runtime；最终以目标机真实测试及实际加载库记录为准。
- 本项目两份 native 库使用 C ABI/ctypes，不依赖 PyTorch C++ ABI；vLLM 自身的 libtorch 扩展仍必须与其要求的 PyTorch 构建匹配。
- 不要全局把 CUDA `lib64/stubs` 加入运行时搜索路径；VMM 需要实际驱动的 `libcuda.so.1`。也不要为排错直接把 Conda `lib/` 或系统 CUDA `lib64/` 整体前置到 `LD_LIBRARY_PATH`，以免覆盖 wheel 的库选择。
- 若遇到 `GLIBCXX`、`libstdc++`、OpenMP 或重复 runtime 报错，先记录 `ldd`/`readelf` 和实际库路径，再修复冲突来源。`OMP_NUM_THREADS`/`MKL_NUM_THREADS` 仅限制线程数，不修复二进制兼容；不要用 `KMP_DUPLICATE_LIB_OK` 掩盖重复 OpenMP 加载。
- 保留现有 `engine_validation` 中的 `VLLM_USE_FLASHINFER_SAMPLER=0` 和 eager/Triton 配置，避免把未验证的 JIT/sampler 路径同时引入首次迁移。
