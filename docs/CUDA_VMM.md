# CUDA VMM 最小实现

`native/paged_arena.cpp` 使用 CUDA Driver API，`fluxmoe/cuda_vmm.py` 通过 ctypes 调用其 C ABI。无需 PyTorch C++ 扩展、CUDA runtime 链接或下载依赖。它验证真实 GPU 存储别名；CPU 事件模拟器仍负责完整调度与数值路径验证。

## 从源码编译

需要 Linux、C++17 编译器、包含 `cuda.h` 的目录，以及驱动提供的 `libcuda.so.1`。在项目根目录执行：

```bash
python3 - <<'PY'
from fluxmoe.cuda_vmm import build_cuda_vmm
print(build_cuda_vmm(cuda_include='/path/to/cuda/include'))
PY
```

当前机器已经验证的头文件目录是：

```text
/home/steve/vllm_env/lib/python3.14/site-packages/nvidia/cu13/include
```

也可设置 `FLUXMOE_CUDA_INCLUDE`，然后调用 `build_cuda_vmm()`。编译入口还检查 `CUDA_HOME`、`CUDA_PATH`、标准 toolkit 位置和当前 Python 的 NVIDIA wheel include 目录。编译本身不初始化或占用 GPU。

默认产物为 `build/libfluxmoe_vmm.so`。`build/vmm_build.log` 与 `build/libfluxmoe_vmm.json` 保存实际编译命令、编译器版本、头文件目录、源文件和头文件 SHA256、返回码及 stdout/stderr；成功构建另保存共享库 SHA256。编译失败也保留记录。

加载器每次获取库时核对本地 `native/paged_arena.cpp` SHA256 与构建记录，并核对共享库 SHA256。源文件修改后必须重新编译，旧库不会静默继续使用；当前进程已加载的库被替换时需要新开 Python 进程。`FLUXMOE_VMM_LIBRARY` 可以指定其他共享库，但旁边必须存在同名 `.json` 构建记录。

## 运行真实别名检查

```bash
FLUXMOE_TEST_CUDA=1 python3 -m unittest discover -s tests -p test_cuda_vmm.py -v
```

普通测试不会初始化 GPU。明确启用后，缺少驱动、设备、VMM 支持或构建库会报错，避免把未执行的硬件验证当成成功；本机真实 GPU 测试结果另保存在 `results/cuda_vmm_smoke.json`。

```python
from fluxmoe.cuda_vmm import CudaPagedArena

with CudaPagedArena(layers=6, layer_bytes=1031) as arena:
    addresses = arena.addresses
    arena.write(0, b'a' * 1031)
    assert arena.read(2) == b'a' * 1031
    arena.write(1, b'b' * 1031)
    assert arena.read(3) == b'b' * 1031
    assert arena.read(0) == b'a' * 1031
    assert arena.addresses == addresses
```

每个层号拥有不同的真实 CUDA 虚拟地址。初始化时调用 `cuMemCreate` 创建两个物理块，按 `layer % 2` 固定映射到各层独立保留的虚拟地址区间；普通读写不重新映射。`logical_bytes = layers × layer_bytes`，`physical_bytes` 包含两块按驱动最小粒度上取整后的空间。小张量的对齐开销可能比逻辑数据大，不能用本例宣称显存压缩收益。

## 验证边界

- `write` 要求一个完整层的数据，`read` 可以读取指定长度的前缀。初次映射的可访问物理块初始化为零。
- 写入层 2 会立即改变层 0、4 的内容，**别名本身不保证旧读者安全**。异步 GPU 内核接入前还需要真正的 WAR/RAW 事件协调。
- H2D/D2H 操作及初始化、关闭使用同步调用和 `cuCtxSynchronize`；没有异步加载流水线、压缩 GPU 解码器接入或 vLLM 集成。此处不报告吞吐、延迟收益。
- `addresses` 中的整数仅在 arena 存活期间有效。不能在外部 GPU 工作仍使用它们时关闭或并发写入。
- 保留设备 primary context，操作期间 push/pop，关闭时只释放自己的引用，不 reset 共享 context；多个 arena 可以独立关闭。构造失败会清理已取得的资源。
- 硬件测试验证地址稳定、真实别名、两物理块隔离、重复创建/关闭及非法输入；模拟器测试负责另一层的调度依赖。

API 依据：[CUDA Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)、[Primary Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html)。
