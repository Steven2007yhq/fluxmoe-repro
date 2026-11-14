# 静态容量估算

`fluxmoe.capacity.estimate_capacity` 只进行算术运算，不查询 GPU，不下载或加载权重，不分配相应容量。`fits_static_budget=true` 仅表示给定估算项之和没有超过预算，不代表部署可运行或真实峰值不会 OOM。

所有容量使用十进制 GB，即 `1 GB = 10^9 bytes`。模型参数量来自 FluxMoE v3 表 1/表 2 的取整数据；压缩比例取表 3 报告的压缩后存储占比。估算把参数均分到各 rank，无法验证实际 TP/EP 分片、复制或划分约束。

| model 参数 | 总参数 P | routed-expert 参数 E | MoE 层数 N | 压缩后比例 c |
| --- | ---: | ---: | ---: | ---: |
| `mixtral` | 47B | 45B | 32 | 0.676 |
| `qwen` | 80B | 77B | 48 | 0.679 |
| `glm` | 355B | 338B | 89 | 0.675 |

## 计算方式

BF16 每个参数 2 字节。对于 G 张 GPU、所有专家层均使用 pageable 存储、CPU 专家占比 f 的情况，每 rank 的 GPU 估算为：

```text
非专家 BF16                = 2(P-E)/G
完整 BF16 双层物化缓冲       = 4E/(NG)
GPU 压缩专家存储            = 2E*c*(1-f)/G
总计                       = 以上三项 + 每卡 KV/state + 每卡 workspace
```

双缓冲包含每层全部专家的 gate/up 和 down 张量，不能再乘活跃专家比例。GPU 压缩存储不包含卸载到 CPU 的专家；CPU 专家以未压缩 BF16 计入主存。当前估算不支持混合常驻层/pageable 层的逐层配置。

`resident=True` 对应全部 BF16 权重常驻的容量对照：

```text
每 rank GPU 总计 = 2P/G + 每卡 KV/state + 每卡 workspace
```

该模式没有 GPU 压缩副本、双层物化缓冲和 CPU 专家卸载；有效 CPU 占比为 0。CPU KV 卸载预算仍单列为 512 GB，与论文对所有系统的统一设置一致。

每卡容量预算为 `gpu_gb * budget_fraction`，默认预算比例为 0.9。`kv_gb_per_gpu` 和 `workspace_gb_per_gpu` 默认 0 方便查看权重/缓冲下界；**这不是可运行部署配置**。Qwen3-Next 包含混合 attention/state，且 KV 头可能在 TP ranks 间复制，不能简单将整模型 KV 字节数除以卡数。

## 设备容量来源

`GPU_PRESETS` 暴露下列规划值。函数接受显式 `gpu_gb`，不擅自把设备型号转成实际可用显存。

| preset | 每卡 gpu_gb | 来源及限制 |
| --- | ---: | --- |
| `a40` | 48.0 | 标称 48GB 按十进制处理的规划估算，本机未实测 |
| `l40s` | 48.3 | 论文 §6.1 报告的设备容量，本机未实测 |
| `h20` | 102.6 | 论文 §6.1 报告的设备容量，本机未实测 |

不能混用 GB/GiB、产品标称值和驱动报告值。上述数值不是 GPU 性能模型；本模块也没有模拟 PCIe、NUMA 或 GPU 间通信。

## 调用示例

在项目根目录运行：

```python
from fluxmoe.capacity import estimate_capacity

weights = estimate_capacity("mixtral", gpus=2, gpu_gb=48)
print(weights["used_gb_per_gpu"])       # 31.43 GB，尚未计入 KV/workspace
print(weights["headroom_gb_per_gpu"])   # 11.77 GB

deployment = estimate_capacity(
    "mixtral", gpus=2, gpu_gb=48,
    kv_gb_per_gpu=10, workspace_gb_per_gpu=2,
)
print(deployment["fits_static_budget"]) # False

glm = estimate_capacity("glm", gpus=8, gpu_gb=102.6, resident=True)
print(glm["headroom_gb_per_gpu"])        # 3.59 GB，权重以外预算很少
```

## 未计入的容量

压缩码表/索引、解压 scratch、激活和通信缓冲、页对齐与碎片、CUDA Graph 池、运行时元数据没有独立建模，必须通过 workspace 预算覆盖或实测。pageable 公式已包含持久 GPU 压缩存储；额外暂存的压缩副本和初始化/加载峰值没有计入。

CPU 专家存储为 `2E*f`（所有 ranks 合计），另列论文的 512 GB 主机 KV 预算。这两者之和不是整机 RAM 要求：未包含检查点加载/暂存副本、可选压缩主存副本、pinned-memory 管理开销、进程、文件缓存和 OS。GLM 在 f=0.125 时专家存储为 84.5 GB；与主机 KV 预算合计 596.5 GB，仍不能据此声称约 600 GB RAM 足够部署。

验证入口：`python3 -m unittest tests.test_capacity -v`。这些测试核对公式、输入和边界，不能替代真实分配及峰值采样。
