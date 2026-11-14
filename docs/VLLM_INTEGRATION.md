# vLLM 0.29 单卡异步分页接入

本阶段已在本机 RTX 4060 Laptop 上把稳定 VMM 权重视图绑定到**实际安装的 vLLM `MoERunner`**，并运行其 `UnquantizedFusedMoEMethod → TritonExperts` 路径。验证使用随机 BF16 专家权重，不需要下载检查点；不是用一个同名 mock 模块代替 vLLM 算子。

实测环境为 `/home/steve/vllm_env/bin/python`，vLLM 0.29.0、PyTorch 2.13.0+cu130。论文使用 vLLM 0.23.0，因此这是当前版本的迁移适配，不能据此声称还原了论文原版本。

本文保留 MoE 模块阶段的独立 smoke 与结果。新增的检查点加载、KV cache、请求调度和 `LLM.generate` 集成见 [完整引擎验证说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VLLM_ENGINE.md)。

## 已验证结果

使用4层同构 MoE，每层4个专家，hidden=64、intermediate=128、top-k=2，batch=16。在非默认 CUDA stream 上连续执行3轮完整前向：

- 每一轮输出与相同真实 vLLM 算子的全驻留 BF16 对照逐位一致，最大绝对误差为0。
- 全部8个参数的真实 VMM 地址在重复执行中保持不变。
- 权重源通过真实 CUDA 解码或 pinned-host H2D 异步写入 VMM；算子通过绑定后的参数直接读取这些地址。
- 测试经过实际 world-size-1 分布式组和 worker workspace 初始化，强制使用 Triton backend。

原始证据：[vllm_paging_smoke.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/vllm_paging_smoke.json)、[完整执行日志](/run/media/steve/Lenovo/FluxMoE论文复现/results/vllm_paging_smoke.log)。JSON 包含配置、实际 runner 类名、vLLM 源文件 SHA256、VMM 地址和加载/读取时间线。首次尝试缺少 worker `WorkspaceManager` 初始化，已补齐并保留失败记录为 `results/vllm_paging_attempt1.*`。

这组小张量验证不代表显存节省或吞吐提升。VMM 的物理分配需要按设备粒度对齐，小张量下实际物理占用可能高于全驻留权重大小。

## 使用入口

在已构建 native CUDA 模块的本项目中运行：

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'
/home/steve/vllm_env/bin/python -m fluxmoe.integrations.vllm \
  --output results/vllm_paging_smoke.json
```

真实 vLLM 集成测试需单独显式启用，普通 CPU 测试不会加载或运行 vLLM：

```bash
FLUXMOE_TEST_VLLM=1 /home/steve/vllm_env/bin/python \
  -m unittest tests.test_vllm_adapter -v
```

此 smoke 在独立进程内初始化并销毁 world-size-1 分布式组，不能嵌入已有 vLLM 服务进程执行。初次执行可能编译 Triton kernel；当前记录仅用于正确性，未按性能实验协议测量。

## 接入方式

vLLM 0.29 的 `FusedMoEFactory` 返回 `MoERunner`，参数位于 `runner.routed_experts.w13_weight` 和 `w2_weight`。Mixtral 模型中，这个 runner 是 `MixtralMoE.experts`。

`process_weights_after_loading` 会根据后端转换权重布局或替换参数。因此绑定必须在 postprocess 完成后执行，并处于 eager 模式、CUDA Graph 捕获之前。当前适配器限定未重排的 Triton BF16 布局 `w13[E,2I,H]` / `w2[E,H,I]`，拒绝 FlashInfer 重排格式、量化以及未完成 postprocess 的模块。

核心 API 位于 [fluxmoe/integrations/vllm.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/integrations/vllm.py)：

```python
from fluxmoe.cuda_pager import CudaMoEPager
from fluxmoe.integrations.vllm import extract_cpu_weights, VllmPagedAdapter

# runners: 已完成 process_weights_after_loading 的顺序 MoERunner 列表。
w13_cpu, w2_cpu = extract_cpu_weights(runners)
pager = CudaMoEPager(w13_cpu, w2_cpu, cpu_fraction=0.125)
adapter = VllmPagedAdapter(runners, pager, (w13_cpu, w2_cpu))

# 保持 vLLM 自身 forward_context，并按原模型顺序调用每个 runner。
with adapter.iteration():
    output = model_forward(...)

pager.synchronize()  # 结果检查或生命周期边界，而非每层同步。
adapter.detach()    # 默认恢复 CPU 参数，释放 runner 对 VMM storage 的引用。
pager.close()       # 此前还需释放用户自己保存的 VMM/派生 Tensor 视图。
```

`model_forward(...)` 表示调用方已有前向函数；这段展示 adapter 使用关系，不是创建完整 LLM 的运行脚本。上面的命令行 smoke 才是本仓库提供的自包含真实运行入口。

绑定通过 `Parameter.data = VMM_view` 保留 Parameter 身份及加载器属性，并释放原 GPU storage 引用。适配器保存 CPU 恢复副本；默认 `detach()` 将参数恢复至 CPU，之后原 runner 不可直接继续 GPU 推理。显式 `detach(restore_to_device=True)` 会重新分配常驻 GPU 权重，需要额外显存预算。

forward pre-hook 调用 `pager.acquire()`，在当前计算 stream 等待 RAW 就绪事件；post-hook 调用 `pager.release()`，记录本层最后读事件，随后允许向复用块预取后续层。捕获模式、乱序/重复调用、重入以及绑定后参数地址被替换会明确报错。单次 `adapter.iteration()` 必须严格按顺序执行所有绑定层一次；异常后适配器不可继续使用，应同步、detach 并重建。

## 目前限制

- 本文的历史 smoke 覆盖实际 MoE 模块与真实分页器；它本身没有启动 `LLM.generate`、KV scheduler、注意力或 tokenizer。后续完整引擎入口已接入这些组件，见 [VLLM_ENGINE.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VLLM_ENGINE.md)。OpenAI HTTP server 仍未覆盖。
- 仅支持单 rank、eager、BF16、SiLU、未量化 Triton。TP/EP、CUDA Graph、EPLB、DBO、LoRA、shared-expert overlap、latent transforms 和 expert bias 被拒绝。
- bootstrap 从已加载的常驻权重提取 CPU 副本，不能避免初次加载完整 GPU 权重的峰值。低显存模型加载需要在 checkpoint loader 阶段注册分页 storage，并重新验证 postprocess。
- CPU 恢复副本属于验证/恢复工件；不能把它们排除后声称已经实现论文主机内存预算。
- 参数绑定后禁止热更新权重、重新运行 postprocess 或更改专家映射。地址检查能发现 storage 替换，不能为任意原地热更新提供一致性。
- 后续引擎入口使用随机 tiny Mixtral 检查点验证请求批处理与 KV 生命周期；预训练模型质量、图捕获、多卡和正式性能仍需单独实验，不能由本阶段 smoke 推断。

本次没有安装、升级或修改现有 `vllm_env`。
