# 真实 VMM 地址作为 PyTorch Tensor

`CudaPagedArena.tensor(layer, shape, dtype=None, offset_bytes=0)` 把已有 CUDA 虚拟地址暴露为连续的 PyTorch Tensor。`dtype=None` 使用 `torch.bfloat16`。创建视图不复制权重、不分配另一份 GPU 权重，也不等待加载完成。

该接口要求可用的 CUDA PyTorch 环境及 NumPy 支持。本机使用 `/home/steve/torch_env/bin/python`；VMM 共享库的构建方法见 [CUDA_VMM.md](CUDA_VMM.md)。

```python
import torch
from fluxmoe.cuda_vmm import CudaPagedArena

arena = CudaPagedArena(layers=6, layer_bytes=8192, device=0)
weights = arena.tensor(0, (64, 64))
alias = arena.tensor(2, (64, 64))
assert weights.data_ptr() == arena.addresses[0]
assert alias.data_ptr() == arena.addresses[2]

copy_stream = torch.cuda.Stream()
compute_stream = torch.cuda.Stream()
ready, consumed = torch.cuda.Event(), torch.cuda.Event()
host = torch.ones((64, 64), dtype=torch.bfloat16).pin_memory()

with torch.cuda.stream(copy_stream):
    weights.copy_(host, non_blocking=True)
    ready.record()
with torch.cuda.stream(compute_stream):
    compute_stream.wait_event(ready)
    result = weights.float().sum()
    consumed.record()
with torch.cuda.stream(copy_stream):
    copy_stream.wait_event(consumed)  # 等旧读完成后，才允许覆盖同一个物理块
    alias.copy_(host, non_blocking=True)

copy_stream.synchronize()  # 仅在验证/销毁边界同步
del weights, alias
arena.close()
```

层0和层2有不同的稳定虚拟地址，但共享物理存储。Tensor 的地址稳定不代表内容常驻；预取和计算仍须由调用者连接 WAR/RAW 事件依赖。`tensor()` 自身不插入事件，不推断当前权重属于哪一层。

## 存储生命周期

实现先导出 `uint8` 的 CUDA array interface，再在 Torch 中通过 `view(dtype)` 与 `view(shape)` 重新解释存储。这样可以支持 BF16，而不依赖 NumPy 未统一的 BF16 `typestr`。导入后严格检查 `data_ptr()` 和 device，任何隐式复制或错误设备都会报错。

PyTorch v2.13/v2.14 的 CUDA array interface 导入器将 exporter 引用保存在 `from_blob` storage 的释放回调中。因此 exporter 内的 lease 强持有 arena，直到最后一个共享该 storage 的 Tensor、slice、detach 或 storage 对象释放。这个行为已按 [PyTorch v2.14 官方源码](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/utils/tensor_numpy.cpp) 和 [v2.13 官方源码](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/utils/tensor_numpy.cpp) 核对；[as_tensor 文档](https://docs.pytorch.org/docs/2.14/generated/torch.as_tensor.html) 说明其共享数据语义。

`arena.live_tensor_leases` 统计独立导出的 Torch storage，不是 Tensor/view 对象数。显式 `arena.close()` 在存在 lease 时抛错并保留 arena；先清除全部 Tensor 和派生视图，再关闭。不要把仍被计算使用的裸 `data_ptr()` 脱离 Tensor/arena 生命周期保存。关闭操作沿用原生 VMM 的同步销毁路径，允许在清理边界等待 GPU，但不在分页热路径执行全设备同步。

支持 BF16、FP16、FP32、FP64、普通整型和 bool。shape 必须连续且各维度大于0；空 tuple 表示标量。偏移须按 dtype 对齐，视图不得超出逻辑 `layer_bytes`。复数、量化和 packed dtype 当前明确拒绝。

## 可复跑验证

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'
FLUXMOE_TEST_CUDA=1 /home/steve/torch_env/bin/python -m unittest tests.test_cuda_vmm_torch -v
```

测试覆盖真实固定地址与双缓冲别名、slice/detach/storage 的持有关系、提前关闭拒绝、offset/dtype 边界，以及非默认 stream 上 pinned H2D→计算→同物理块覆盖的事件顺序。GPU 测试默认跳过，显式启用但设备不可用会失败。这里只验证存储及正确性，不能据此声称端到端推理性能收益。

本机 RTX 4060 Laptop 上，`torch_env` 的 PyTorch 2.14.0+cu126 已通过5项新GPU测试及2项纯CPU几何检查；旧VMM的6项测试也通过。另在 `vllm_env` 的 PyTorch 2.13.0 中验证了指针保持、事件排序、派生视图持有与关闭拒绝，记录在 `results/vmm_torch_view_vllm_env.json`。
