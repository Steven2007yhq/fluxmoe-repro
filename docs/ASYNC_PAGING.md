# 真实异步 GPU 分页

当前已完成单卡、固定后端放置的真实 GPU 执行链：持久压缩源 → CUDA 解码或 pinned-host H2D → 固定 VMM 地址 → PyTorch/vLLM 专家算子。CPU 离散事件模拟器仍独立保留。

## 执行与验证

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'
/home/steve/torch_env/bin/python -m fluxmoe.async_validation --output results/async_paging
FLUXMOE_TEST_CUDA=1 /home/steve/torch_env/bin/python -m fluxmoe validate --output results/async_validation_cuda
/home/steve/vllm_env/bin/python -m fluxmoe.integrations.vllm --output results/vllm_paging_smoke.json
FLUXMOE_TEST_VLLM=1 /home/steve/vllm_env/bin/python -m unittest tests.test_vllm_adapter -v
```

第一条入口在4060上运行5层、5轮真实 GPU MoE 前向，逐轮比较常驻权重参考，并检查50份物化权重的原始 BF16 位模式。输出、权重和地址检查均通过。`result.json` 保存实际设备、配置、数值比较、内存账本、计数器、源码指纹；`trace.json` 保存真实 CUDA events 的 Chrome Trace 时间线。

vLLM 入口使用安装版本0.29.0的真实 `MoERunner` 和 Triton backend，4层、3轮与相同 vLLM 算子的全驻留结果逐位一致；详情见 [vLLM 接入说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VLLM_INTEGRATION.md)。两个命令分别使用现有 PyTorch 环境，没有修改环境包。

## 权重与事件生命周期

`CudaMoEPager` 接收 CPU BF16 `w13[E,2I,H]` / `w2[E,H,I]`。初始化阶段完成 CPU 编码与索引、压缩源上传、pinned-host 分配和 VMM 映射。每个 projection 有两个物理缓冲区，所有层保留自己的稳定 CUDA VA。PyTorch Tensor 通过非拥有 storage 引用这些地址，storage lease 防止 arena 在 Tensor 存活时被释放。

每种 projection 分别有解码和 H2D stream，共4个加载 stream；计算沿调用方指定的 stream 执行。正常热路径为：

1. `begin_iteration()` 预取最初两层；保留上一轮的最后读取事件。
2. `acquire(i)` 让计算 stream 等待本层全部加载完成事件，建立 RAW 依赖。
3. 调用方在该计算 stream 上提交读取本层权重的算子。
4. `release(i)` 记录最后读取事件。后续复用同一物理块的加载 stream 等待该事件，建立 WAR 依赖，并可预取 `i+2`。
5. 所有层完成提交后 `end_iteration()` 结束逻辑迭代。它不等待 GPU 执行完毕，下一轮继续依赖已有事件。

热路径不进行 CPU 解码、权重 D2H、逐层权重或解码缓冲分配、stream/device 同步。CUDA events 自身仍有创建与记录开销。`synchronize()`、`trace()`、`close()` 是明确的验证或清理边界。CUDA 解码直接写目标 VMM 指针；H2D 使用 `copy_(..., non_blocking=True)` 从持久 pinned Tensor 复制。

调用方必须把所有权重读取安排在 acquire/release 指定的 stream 上；使用额外计算 stream 时，需先建立相应 join。分页器不代替模型 activation 的跨 stream 依赖。当前仅支持推理，不支持反向传播或 CUDA Graph 捕获。

## 错误与关闭

半提交的预取发生异常后，分页器进入失败状态并拒绝继续提交。关闭会先等待已登记的加载与消费者 stream，检查解码错误，再清理资源；即使检查报告错误，也继续尝试清理并最终汇总异常。

关闭前应先 detach 已绑定的 vLLM 模块，并释放用户持有的 VMM Tensor/派生 view。仍有外部 storage lease 时，关闭会报错并保留 arena，释放引用后可以重试。测试覆盖 slice/detach/storage 持有者和关闭时仍排队的读取。

时间线最多保留 `max_trace_events=10000` 条记录，超出部分计入 `trace_events_dropped`；设为0可不保留记录。事件检查和结果保存都位于测量边界，当前数据仅用于正确性诊断。

## 支持的结论与限制

实测支持：在已测4060、BF16同构单rank条件下，直接解码、pinned H2D、真实别名复用及事件依赖保持数值和位模式正确；vLLM 实际 MoE kernel 能直接读取绑定后的 VMM 权重。

尚未验证：A40、多卡 TP/EP、完整 LLM worker/检查点加载、KV 调度、CUDA Graph、在线 GPU 驻留调整、质量数据集和论文性能矩阵。vLLM bootstrap 仍从已加载的常驻权重提取 CPU 副本，尚未消除初次 GPU 加载峰值。

小张量受 VMM 分配粒度影响，物理缓冲区可能比原始权重更大。独立 GPU 验证含额外权重快照和结果比较；真实事件时间线中的重叠也不能直接证明吞吐提升。当前不作论文加速比或显存节省的结论。

事件语义参考：[PyTorch CUDA streams](https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams)、[CUDA stream/event API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html)。本项目实际行为以源码和保存的测试记录为准。
