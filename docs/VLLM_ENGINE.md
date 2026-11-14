# 完整 vLLM 推理流程：检查点、KV cache 与生成

本阶段把前一阶段的 [MoERunner 分页适配器](/run/media/steve/Lenovo/FluxMoE论文复现/docs/VLLM_INTEGRATION.md) 接入实际 vLLM 0.29 引擎。入口经过磁盘检查点加载、模型 postprocess、KV 分配、请求调度、attention、MoE、LM head 和 `LLM.generate`。全驻留与分页模式在独立进程中顺序运行，使用同一检查点、tokenizer、请求和采样配置。

本机 RTX 4060 已完成全驻留/分页成对验证，[比较报告](/run/media/steve/Lenovo/FluxMoE论文复现/results/engine_validation/comparison.json) 为 `passed`。这是小型随机模型上的执行链路与数值正确性证据，不是论文性能复现。

## 本机结果

原始记录为 [resident.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/engine_validation/resident.json)、[paged.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/engine_validation/paged.json)，完整日志为 [resident.log](/run/media/steve/Lenovo/FluxMoE论文复现/results/engine_validation/resident.log)、[paged.log](/run/media/steve/Lenovo/FluxMoE论文复现/results/engine_validation/paged.log)。

| 检查 | 实测结果 |
| --- | --- |
| 权重加载 | 两侧四层专家的形状和 SHA256 均与磁盘检查点独立拼接结果一致 |
| 主对照生成 | 11 个请求、176 个生成 token；token IDs 和停止原因一致 |
| 主对照概率 | 176 个生成 token logprob、200 个 prompt token logprob；两类最大绝对误差均为 0 |
| KV 写入 | 每侧第一层 attention 检查 454 次 slot 写入，BF16 逐 bit 一致，无失败 |
| 历史 KV 读取 | 每侧 165 次独立 FP32 SDPA 对照；无失败，无缺失历史 slot |
| 调度覆盖 | 每侧观测到 7 次 chunked prefill；热 prefix 请求命中 48 个缓存 token；64 次已写 slot 重写 |
| 移除 KV hooks 后 | 每侧另跑 3 个请求、48 个生成 token；与本侧先前相同请求的 token IDs 一致，生成及 prompt logprob 最大绝对误差均为 0 |

历史 attention 对照的最大绝对误差为 `0.0018382072448730469`，全部元素满足混合容差 `abs(actual-reference) <= 1e-3 + 1e-2 * abs(reference)`。这里没有声称所有元素的绝对误差都小于 `1e-3`；BF16 GPU attention 与 FP32 CPU 参考采用不同精度。

分页进程累计记录 107 次分页 iteration、2,140 次 GPU 解码和 1,284 次 H2D，时间线无截断。这些累计值包含引擎生命周期中的前向与附加核验，不是单位请求的性能测量。VMM 物理分配为 **8 MiB**，对应逻辑专家权重只有 **1.5 MiB**；本案例受 2 MiB 分配粒度影响，不能据此宣称节省显存。

首次启动探测曾在 FlashInfer sampler 的 JIT 编译中因未找到 `nvcc` / 默认 `/usr/local/cuda` 失败，记录保留于 [resident_init.log](/run/media/steve/Lenovo/FluxMoE论文复现/results/engine_probe/resident_init.log)。最终两侧统一设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，使用 vLLM 原生 GPU sampler；没有修改已有依赖或将推理转到 CPU。

## 可复跑入口

使用本项目现有 native CUDA 解码/VMM 库，以及 `/home/steve/vllm_env/bin/python` 中已安装的依赖；构建方法见 [CUDA 解码说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_CODEC.md) 和 [VMM 构建说明](/run/media/steve/Lenovo/FluxMoE论文复现/docs/CUDA_VMM.md)。以下命令不会安装依赖或下载模型。

```bash
cd '/run/media/steve/Lenovo/FluxMoE论文复现'

# CPU 构建并检查真实 HF safetensors 检查点；已存在时核验 hash 后复用。
/home/steve/vllm_env/bin/python -m fluxmoe.checkpoint models/tiny_mixtral

# 每次完整运行使用一个新的目录，保留先前成功或失败的证据。
/home/steve/vllm_env/bin/python -m fluxmoe.engine_validation \
  --model models/tiny_mixtral --output results/engine_validation_run01

# 只重新读取已有 JSON 并生成 comparison.json，不再次运行 GPU。
/home/steve/vllm_env/bin/python -m fluxmoe.engine_validation \
  --output results/engine_validation_run01 --compare-only

# 检查点的 CPU 回归测试。
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/steve/vllm_env/bin/python \
  -m unittest tests.test_checkpoint -v
```

完整入口先运行全驻留基线，再运行分页模式。输出包括 `resident.json` / `paged.json`、对应完整日志以及 `comparison.json`。检查点或运行失败会报错并保留记录；已有运行目录不会被重新执行覆盖。`--compare-only` 会重新生成该目录的比较报告。

## 小型检查点的含义

[models/tiny_mixtral](/run/media/steve/Lenovo/FluxMoE论文复现/models/tiny_mixtral) 是本地随机初始化的 **Mixtral 架构验证 fixture，不是预训练语言模型**。它有真实的 causal attention、RoPE、top-2 MoE、RMSNorm 与 LM head；生成内容只用于检查执行链路和数值一致性。

| 项目 | 固定配置 |
| --- | --- |
| decoder 层 / 每层专家数 | 4 / 4 |
| hidden / intermediate | 128 / 128 |
| attention heads / KV heads / head_dim | 4 / 2 / 32 |
| 词表 / 配置最大上下文 | 256 / 256 |
| dtype / 初始化 seed | BF16 / 27415 |
| 参数量 / 权重字节 | 1,051,776 / 2,103,552 |
| tokenizer | 本地 WordLevel，包含 BOS/EOS/PAD/UNK |

`fixture_manifest.json` 记录参数名、形状、逐 tensor SHA256、文件 SHA256、初始化种子、依赖版本及样例 prompt。构建过程使用 CPU，并通过公开的 HF `from_pretrained` 接口再次加载，核对全部 BF16 参数逐 bit 一致；不同配置、seed 或已损坏的目标目录会被拒绝覆盖。

当前 Transformers 5.17 的内存权重使用 `.mlp.experts.gate_up_proj` / `down_proj`，vLLM Mixtral 加载器使用经典 `.block_sparse_moe.experts.N.w1/w2/w3`。本项目导出后者，HF 5 的官方转换逻辑在加载时重新融合。`head_dim=null` 在两端由 hidden/heads 推导为 32，RoPE 参数保存在 `rope_parameters`，dtype 保存在 `dtype` 字段；这些配置与当前 vLLM 源码接口一致。

## Loader 与 worker extension

实现位于 [fluxmoe/integrations/engine.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/integrations/engine.py)。`register_engine_loaders()` 注册两种加载格式：

- `load_format="fluxmoe_resident"`：调用 vLLM `DefaultModelLoader`，保留全驻留权重，并安装相同的前向观测 hooks。
- `load_format="fluxmoe"`：调用同一原生加载器，在 kernel postprocess 完成后提取专家权重，创建 `CudaMoEPager` 并绑定稳定 VMM Tensor；随后才进入引擎 KV profiling/分配阶段。

`worker_extension_cls="fluxmoe.integrations.engine.FluxMoEWorkerExtension"` 使 worker 进程导入并注册加载器，并提供 `fluxmoe_status` / `fluxmoe_close` RPC。该路径没有改动已安装的 vLLM 文件。

最小 API 示例复用已固定的验证配置：

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from vllm import LLM, SamplingParams
from fluxmoe.checkpoint import build_tiny_moe_checkpoint, expected_vllm_expert_manifest
from fluxmoe.engine_validation import engine_config
from fluxmoe.integrations.engine import register_engine_loaders

model = Path("models/tiny_mixtral").resolve()
build_tiny_moe_checkpoint(model)
register_engine_loaders()
llm = LLM(
    model=str(model), tokenizer=str(model),
    load_format="fluxmoe",
    model_loader_extra_config={"cpu_fraction": 0.25, "prefetch": True},
    worker_extension_cls="fluxmoe.integrations.engine.FluxMoEWorkerExtension",
    **engine_config(),
)
try:
    status = llm.collective_rpc("fluxmoe_status")[0]
    assert status["loading"]["expert_weights"] == expected_vllm_expert_manifest(model)
    outputs = llm.generate(
        ["hello world"],
        SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, logprobs=5),
        use_tqdm=False,
    )
    print(outputs[0].outputs[0].token_ids)
finally:
    try:
        llm.collective_rpc("fluxmoe_close")
    finally:
        llm.llm_engine.engine_core.shutdown()
```

这段示例展示加载与生成 API；包含两侧对照、KV 诊断、原始数据与失败记录的完整入口是 `fluxmoe.engine_validation`。`fluxmoe_close` 是终止操作：它解除 hooks，将恢复权重放回 CPU 并释放分页资源，不能在调用后继续用同一引擎生成。

`model_loader_extra_config` 支持 `cpu_fraction`、`prefetch`、`max_trace_events` 以及 `checkpoint_format`。`cpu_fraction` 按全部专家 BF16 原始字节计算主机预算，选择单位是单个专家的完整 projection tensor（`gate_up` 或 `down`），不是整层。默认选择器优先 `down`，并按各层已分配字节平衡。本配置 `cpu_fraction=0.25` 选中 12 个 `down` tensor，经 pinned-host H2D 加载；其余 tensor 走 GPU 压缩源和解码路径。该设置用于覆盖两条加载路径，不是论文 CPU 内存预算的复现。

## 正确性证据

[engine_validation.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/engine_validation.py) 配置短文本混合请求、长 prompt 的 chunked prefill、重复 prefix 的冷/热请求以及 KV block 复用场景。所有请求使用 `temperature=0` 与 `ignore_eos=True`，从相同初始条件进行比较。实际覆盖和检查次数应读取运行记录。

权重检查独立于两侧 vLLM loader：`expected_vllm_expert_manifest(path)` 直接从磁盘 safetensors 读取各专家，按 NVIDIA unquantized Triton 的布局拼接 `w13=[w1;w3]` 并堆叠 `w2`，再与四层实际加载参数的形状和 SHA256 对照。这避免了“全驻留与分页同时装错权重但彼此相同”被误判通过。这个 oracle 只适用于当前布局，不能用来核对有权重重排的 FlashInfer 或分片模型。

[engine_checks.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/engine_checks.py) 要求两侧检查点与语义配置相同，prompt/token IDs、生成 token IDs 和停止原因一致，并核对所选生成 token 及 prompt token 的有限 logprob。当前默认 logprob 容差为 `atol=1e-5, rtol=1e-5`；这针对同一内核下的分页一致性，不是任意 BF16 算法之间的通用容差。它还要求实际 KV 写入与历史读取检查存在且无失败，缺失证据不能通过。

[kv_observer.py](/run/media/steve/Lenovo/FluxMoE论文复现/fluxmoe/kv_observer.py) 在**第一层 attention** 上观测真实 slot mapping、block table 与序列长度。它逐 bit 核对已分配 slot 中的 K/V 与当前输入，并针对有历史的单 token query，从实际 cache 读出历史 K/V，用独立 FP32 CPU attention 重算输出；默认 attention 容差为 `atol=1e-3, rtol=1e-2`。记录还包括分配是否保持稳定、chunked prefill、已写 slot 的覆盖与复用。整个 KV 池包含未初始化空间，因此不拿全池 hash 在独立进程之间做等值比较。

KV observer 的 `.cpu()` 与核验会主动同步；读取 pager 时间线也有边界同步。**本入口的耗时只能用于诊断，不能作为吞吐、延迟、通信计算重叠收益或论文加速比。** 后续性能实验需要移除观测 hooks，另行固定预热、负载、采样和测量边界。

## 显存预算与当前边界

验证配置使用固定 **1 MiB KV 池**，两侧一致。CUDA VMM 和 native 压缩后备的分配不完全属于 PyTorch caching allocator；直接依赖 vLLM 的自动 profiling 推导可用 KV 大小可能把这些分配变化混入估算。固定 KV 池使本次比较条件可核对，但它不是大模型服务的容量策略，也不证明在线 KV/专家显存再平衡。

当前支持单 GPU、单 rank、BF16、eager、未量化 Mixtral，MoE backend 强制 Triton；验证的 attention backend 为 `TRITON_ATTN`。TP/PP/DP/EP、CUDA Graph、编译、EPLB、DBO、LoRA、speculative decoding 和重入前向均不在支持范围，部分模式会在加载前明确拒绝。

**初次加载仍要求完整模型能常驻 GPU。** 当前 loader 先让 `DefaultModelLoader` 完成权重加载与 postprocess，再替换为 VMM storage。因此它证明适配正确性，尚不能让一个初次加载就超过显存的大模型成功启动。CPU 恢复副本、压缩源、索引、VMM 对齐、双缓冲和临时分配也需要计入账本；小 fixture 的 VMM 粒度开销可能超过其原始权重。

本阶段使用真实 vLLM 请求引擎，不包含 OpenAI HTTP server、预训练模型语言质量、长时间压力测试、多卡 A40 或原论文模型/数据集的性能矩阵。后续扩展需要在各自条件下重新验证。
