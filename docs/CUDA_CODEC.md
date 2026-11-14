# CUDA BF16 Huffman 解码器：实现与验证边界

这是根据论文压缩机制独立实现的 **CPU 编码器 + CUDA 解码器**。CUDA kernel 读取设备端压缩指数码流，遍历 Huffman 树，重建 BF16 `uint16_t`。目前提供两个入口：旧的同步 smoke 接口分配输出、拷回 CPU 做逐位比较；新的 `CudaCompressedTensor` 持久保存压缩后备数据，在调用者指定的 CUDA stream 上直接写入外部设备地址或 VMM 缓冲区。非空输入没有静默退回 CPU 的路径。

这是独立的树遍历 kernel，尚未达到或验证论文约400GB/s的解压性能。当前的异步入口已经供 `CudaMoEPager` 使用；vLLM 的整体接入和验证范围应以相应 adapter 文档及原始记录为准。

## 持久压缩源与真实异步接口

```python
import torch
from fluxmoe.cuda_codec import CudaCompressedTensor

load_stream = torch.cuda.Stream(device=0)
compute_stream = torch.cuda.current_stream(0)

with CudaCompressedTensor(encoded_fmbf_bytes, device=0) as source:
    output = torch.empty(source.count, dtype=torch.uint16, device="cuda:0")
    source.decode_into(output.data_ptr(), load_stream.cuda_stream, source.output_bytes)
    ready = torch.cuda.Event()
    ready.record(load_stream)
    compute_stream.wait_event(ready)
    # 在 compute_stream 上使用 output；例如 view(torch.bfloat16) 后执行算子。
    copied = output.clone()
    compute_stream.synchronize()  # 示例的显式验证边界，非分页层内要求。
    source.check()
```

构造函数在 CPU 严格验证一次 FMBF、生成解码树和分块索引，再将 sign/mantissa、指数码流、索引、解码树及错误状态上传到 GPU。完整 CPU 已解码参考只用于准备过程，构造返回后不保留在 source 对象里。可读取 `count`、`output_bytes`、`device_bytes`、`container_sha256` 和 `preparation_ms`。

正常 `decode_into(destination, stream, output_bytes=None)` 路径执行设备和指针校验、kernel launch、读取完成事件记录。它直接写入调用者的目标地址，**不做 CPU 解码、H2D/D2H 权重往返、输出显存分配或 stream/device 同步**。每个 stream 首次使用时创建一个完成事件，后续反复记录该事件。`output_bytes` 默认是精确所需字节数；传入显式值时拒绝容量不足。接口检查 BF16 对齐、CUDA 设备一致性和 device pointer 类型，但不能从一个 VMM 大映射的内部地址推断张量边界，目标实际大小仍由调用方保证。

该 source 的多个 stream 可以并发读取同一份只读压缩数据。所有 kernel 共用的错误状态仅初始化时清零，失败通过 `atomicExch` 写入非零值，不在每次 launch 前清零，避免一个 stream 抹掉另一个 stream 的错误。`check()` 等待 source 登记的所有 stream 最后读取事件，再读回错误状态；这是显式检查边界。`close()` 同样等待所有最后读者再释放后备显存，不把对象析构当成立即释放。正常热路径没有这些等待；异常情况下若 kernel 已排队但完成事件无法登记，会同步该 stream 以保护生命周期，并抛出错误。

source 仅保护**自身的压缩后备存储**。目标 VMM 缓冲的生命周期、写前等待上次读者（WAR）、计算前等待本次加载（RAW）、不同计算 stream 间的 activation 依赖由调用者负责。`CudaMoEPager` 接入了权重 WAR/RAW；外部消费者必须按其 acquire/release 协议使用。CUDA Graph capture 当前明确拒绝，默认 stream 句柄0可用但正常分页应使用专用非默认流。

`device_bytes` 包含持久的压缩输入、chunk 索引、树和错误状态，不包含外部输出/VMM块、CUDA events、context或驱动元数据。无论当前压缩比如何，容量核算都不能只计算指数码流。

## 旧同步 smoke 接口的数据路径

`decode_cuda_with_metrics()` 依次执行：

1. CPU 调用 `decompress_bf16()`，完整检查 FMBF 格式、两项 CRC、码表、有效位数和填充，并产生完整的已解码 BF16 参考数据。
2. CPU 从参考数据和码长表建立每个 chunk 的码流起始 bit offset，构造最多 511 个节点的二叉解码树；单符号情况有 2 个节点。
3. 把 sign/mantissa、指数码流、chunk offsets 和解码树复制到 GPU。默认每个 CUDA 线程顺序解码 256 个元素，线程之间处理不同 chunk。
4. CUDA kernel 从压缩码流重建全部 BF16 字，并核对 chunk 结束位置。CUDA events 等待该 kernel 完成。
5. 把 GPU 结果复制回 CPU，与步骤 1 的完整参考结果逐字节比较。任何不一致均抛出异常。

**CPU 在启动 kernel 之前已经持有完整 decoded oracle，而且已经做过一次完整解码。** 这一设计用于严格验证 GPU kernel，不属于 GPU-only 解压管线。把 kernel 计时除以输出字节数，不能代表此接口的端到端解压吞吐，也不能拿来对比论文结果。

新的持久 source 已将完整 CPU 解码与索引构造移出每次物化路径，并直接把设备输出交给下游算子。目前索引仍在 source 初始化时生成，没有写入离线 FMBF 工件；大型模型启动时仍承担这项一次性工作和峰值主机内存。把索引提前持久化、优化解码 kernel 和复用码表仍是后续工作。

## 格式兼容与容量

磁盘输入继续使用 `codec.py` 定义的独立 **FMBF v1** 格式：288 字节固定头，sign/mantissa 每元素 1 字节，指数使用 canonical Huffman，码流按 MSB-first 排列。格式不是作者实现的兼容格式。

CUDA chunk 索引目前不写入 FMBF 文件。旧同步接口每次调用时生成，持久 source 每次构造时生成一次。默认每256个元素一个 chunk，两种 API 的 `chunk_size` 都可在1–65536范围内调整。码流本身不为 chunk 插入字节对齐或 padding；chunk可从任意bit位置开始。GPU解码树及索引占用额外显存，不能只按压缩负载大小计算设备预算。

记录中的 `gpu_allocated_bytes` 包含本次 native 调用显式申请的输入、索引、树、输出和错误状态；不包含 CUDA context、runtime 或驱动内部开销。

## API 和错误行为

```python
from fluxmoe.cuda_codec import (
    available,
    build_cuda_library,
    decode_cuda,
    decode_cuda_with_metrics,
    CudaCompressedTensor,
)

print(available())
# available() 只检查可访问的驱动/设备，不编译、不导入 torch。
# True 不保证 nvcc、CUDA runtime 或本地 .so 已可用。

library_path = build_cuda_library(nvcc="/absolute/path/to/nvcc")
raw_bf16_bytes = decode_cuda(encoded_fmbf_bytes)
raw_bf16_bytes, metrics = decode_cuda_with_metrics(encoded_fmbf_bytes, chunk_size=256)
```

损坏输入抛出 `ValueError`，错误输入类型抛出 `TypeError`；构建、动态库、CUDA driver/runtime 或设备执行错误抛出 `CudaCodecError`。空输入仅验证容器并返回空结果，明确记录 `kernel_launched=false`。CUDA 错误带有失败 API 名称和 runtime 错误信息。

## 构建环境选择

不安装包、不更改已有虚拟环境，不使用 torch C++ extension。主机 Python 仅通过 `ctypes` 调用 C ABI，因此可以在已有 torch 环境的 Python 中运行，而不导入其 torch 或 CUDA runtime。

编译器优先级如下：

1. `build_cuda_library(nvcc=...)` 或 CLI `--nvcc`。
2. 环境变量 `FLUXMOE_NVCC`。
3. `PATH` 中的 `nvcc`。
4. `CUDA_HOME/bin/nvcc`，随后 `CUDA_PATH/bin/nvcc`。
5. 当前 `sys.prefix`、`~/vllm_env`、`~/torch_env` 下现有 NVIDIA pip toolkit 目录。

CUDA runtime 优先使用 `FLUXMOE_CUDART` 指定的完整 `.so` 路径，否则在选定 nvcc 的 toolkit `lib`、`lib64`、`targets/x86_64-linux/lib` 及相邻 `cuda_runtime/lib` 内查找。不要手工把不同主版本的 toolkit 与 runtime 混配。

编译通过 `--cudart=none` 关闭 nvcc 的自动 runtime 链接，显式把选中的 runtime 路径交给 linker，并设置相应 RUNPATH。加载时也读取构建元数据中的 runtime 路径。此实现不依赖 PyTorch C++ ABI。本机已经验证在 `torch_env` 的 cu126 PyTorch 进程里，用该库的CUDA13 runtime向 PyTorch 创建的非默认 stream 提交 kernel，并直接写 PyTorch 分配的设备地址；这支持本次环境的互操作结论，不保证其他 runtime/驱动组合。移到目标服务器后需重新核对。

架构优先级为函数 `architecture=` / CLI `--architecture`、`FLUXMOE_CUDA_ARCH`、默认 `native`。`native` 面向构建机器的 GPU；移到 A40 时应在目标服务器重新构建，例如使用 `--architecture=sm_86`。它不意味着当前二进制兼容所有 NVIDIA GPU。

解码库优先使用 `FLUXMOE_CUDA_LIBRARY` 的绝对路径；否则使用 `native/build/libfluxmoe_huffman.so`，若文件不存在则自动构建。显式指定但不存在的库会报错，不会悄悄改用其他文件。`FLUXMOE_CUDA_DRIVER` 仅影响 `available()` 的 driver probe，默认是系统 `libcuda.so.1`。

本地 `.so` 存在时不会根据源码时间戳自动重编译。构建元数据保存 CUDA 源码 `source_sha256` 和编译工件 `artifact_sha256`；默认库每次加载或返回缓存前都要求并校验这两项。源码变化、工件被替换或旧元数据缺少指纹时明确报错要求重编译。显式指定的自定义库若带指纹也会校验；无元数据的自定义库只提供兼容加载，不具备源码溯源保证。源码或工具链变化后，应执行明确的 build 命令，再启动新 Python 进程；进程内已缓存的库若被新的工件替换，会明确要求启动新进程。

构建保存 `native/build/build.log` 与 `native/build/libfluxmoe_huffman.json`，记录准确命令、编译器路径、runtime 路径、版本、源码与工件 SHA256 及编译输出。构建日志和 `.so` 属于本机工件，换机器后需重新生成。

## 本机已经运行的验证

环境：RTX 4060 Laptop，SM 8.9，驱动 595.99.02；编译器为已有 vllm 环境中的 nvcc 13.4.92。解码 smoke 由 `/home/steve/torch_env/bin/python` 启动，runtime 自报 13000，driver API 版本自报 13020。nvcc、runtime、driver API 的版本字段含义不同，完整值保存在运行证据中。

```bash
/home/steve/torch_env/bin/python -m fluxmoe.cuda_codec build \
  --nvcc /home/steve/vllm_env/lib/python3.14/site-packages/nvidia/cu13/bin/nvcc

FLUXMOE_TEST_CUDA=1 /home/steve/torch_env/bin/python \
  -m unittest tests.test_cuda_codec tests.test_cuda_async_codec -v

/home/steve/torch_env/bin/python -m fluxmoe.cuda_codec verify --elements 65536
```

4 项真实 GPU 测试已经通过，包括所有 65,536 种 BF16 位模式、长 Huffman 码、非字节对齐 chunk、NaN、signed zero 以及 512 KiB 非均匀数据。另有 5 项 CPU 索引、输入检查和源码/工件指纹验证测试。未设置 `FLUXMOE_TEST_CUDA=1` 时，4 项真实 GPU 测试明确跳过；常规 CPU 测试不会隐式编译或启动 GPU。

新增持久 source 测试另覆盖全部65,536种位模式直接写非默认stream的目标、CPU准备函数被禁用时仍能物化、延迟stream中的提交不等待完成、消费者事件依赖、同source多stream并发、关闭时等待最后读者、参数边界和空输入。这组测试含4项真实GPU测试及2项纯CPU测试；是否在最终构建上通过，以本次最终验证日志为准。

`results/cuda_codec_smoke.json`（由 `native/build/cuda_smoke.json` 原样复制）保存一次 65,536 元素验证：输出 131,072 字节，真实 kernel 已启动，结果与输入逐位一致。该次 kernel 为约 1.70 ms，CPU 完整验证及索引约 120.77 ms，端到端约 375.21 ms。

**这些是首次调用的 correctness smoke 时间，不是性能 benchmark。** 没有进行足够预热、独立重复或稳定功耗控制；首调用还包含 context/runtime 初始化。`kernel_ms` 不含 CPU 预处理、显存申请、H2D/D2H 和比较；`native_allocate_transfer_kernel_copyback_ms` 包含 native 调用的设备申请、拷贝、kernel 与清理；`end_to_end_ms` 还包含 Python 预处理和首次调用可能发生的构建/加载。不能从这些数值推导论文吞吐、A40 性能或 MoE 推理加速比。

实现参考 NVIDIA 官方 [CUDA Runtime Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)、[Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)、[Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) 和 [Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html) API；原论文的方法依据仍以项目中的 PDF 为准。
