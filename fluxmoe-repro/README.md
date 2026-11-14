# FluxMoE 动机实验复现方案（路线 A）

> **目标不是复刻 7.2×，而是在自己的机器上验证它的因果链：**
> 「MoE 权重常驻挤占 KV cache 容量 → 高并发 / 长上下文时吞吐崩塌」。
>
> 论文：FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving（[arXiv 2604.02715](https://export.arxiv.org/abs/2604.02715)，v3 2026-09）
> 论文测试床：4×L40 48G + 2TiB DRAM，Mixtral-8×7B-Instruct / Qwen3-Next-80B-A3B，基于 vLLM v0.10.2。

---

## 0. 先定成功标准（否则你会陷入对齐数字的泥潭）

四条判据，**满足 ≥3 条即算复现成功**：

| 编号 | 判据 | 观测方式 |
|---|---|---|
| **S1** | 吞吐随并发先升、后转平或下降（出现拐点） | 吞吐-并发曲线 |
| **S2** | 拐点处 `num_preemptions` 开始增长，`gpu_cache_usage` 贴住 100% | Prometheus 指标 |
| **S3** | 提高 KV 预算后，拐点右移或消失 | 多组 KV 预算对照 |
| **S4** | 固定并发、context 变长 → 吞吐下降；KV 预算变大可缓解 | 上下文扫描 |

**绝对数字不重要。** 论文是 4×L40 + Qwen3-80B，你大概会是 4×A40 + Mixtral，吞吐差 2 倍完全正常。**趋势对就是成功，比值对不上是预期内的。**

---

## 1. 要验证的 4 个 Claim 与论文原始数字

| 编号 | 论文 claim | 论文数字 | 你能否验证 | 备注 |
|---|---|---|---|---|
| C1 | KV 预算 20GiB → 60GiB，吞吐提升 | +58.8%（Mixtral，batch 128，ctx 4096） | ✅ 主要目标 | 用 `gpu-memory-utilization` 扫预算 |
| C2 | 固定 20GiB 时 ctx 1024→4096 吞吐暴跌 | −66.7% | ✅ 可验证 | 你只能测 vLLM 侧，测不了 FluxMoE 侧 |
| C3 | vLLM 在 batch 128→256 出现**负 scaling** | −32.2%（Qwen3-80B，ctx 4096） | ⚠️ 模型不同，看**形状**即可 | 你的拐点位置会随模型/显存变 |
| C4 | ctx 1024→4096 吞吐下降 | −45.3%（batch 256） | ✅ | 同上 |

> 注意 C3/C4 论文是在 Qwen3-Next-80B 上测的。**你要复现的是"负 scaling 这个现象存在"，不是这两个百分数。**

---

## 2. 硬件分级与模型选择

先算清楚"你能撑多少 token"，再决定用哪个模型：

```
每 token KV 字节 = 2(K,V) × n_layers × n_kv_heads × head_dim × 2(bf16)
可容纳 token 数 = 可用 KV 显存 / 每 token 字节
```

| Tier | 硬件 | 推荐模型 | 权重 | KV/token | 可用 KV | 容量 | 能复现什么 |
|---|---|---|---|---|---|---|---|
| **0（强烈推荐先做）** | **1×24G 或 1×48G** | `allenai/OLMoE-1B-7B-0924-Instruct`（16 层、16 KV head、128 dim） | 13.8 GB | 128 KiB | 24G 卡约 9 GB | ~72k token | **完整现象**：ctx 4096 时 conc 16 就开始 preempt |
| 1 | 1×48G | `Qwen/Qwen2.5-7B-Instruct`（dense，28 层、4 KV head） | 15 GB | 56 KiB | ~33 GB | ~590k token | 稠密版的同一现象（做对照） |
| 2 | 2×48G | `mistralai/Mixtral-8x7B-Instruct-v0.1` TP=2 | 94 GB | 128 KiB | **≈0**（正好复现论文 Exp#2 的 OOM） | — | 复现"vLLM 装不下"这一条 |
| **3（主线）** | **4×48G** | `mistralai/Mixtral-8x7B-Instruct-v0.1` TP=4 | 94 GB | 128 KiB | 21–79 GB（随 util） | 172k–640k token | 论文主实验的动机部分 |

**关键结论：Tier 0 一张 24G 卡就能跑出声称的完整现象。** 不要等 4 卡到位再开始。

---

## 3. 实验设计

### 自变量（3 维）

| 变量 | 取值 | 控制方式 |
|---|---|---|
| **KV 预算** | 4 档：util = 0.60 / 0.70 / 0.80 / 0.90 | `--gpu-memory-utilization`（需重启服务） |
| **并发** | 16 / 32 / 64 / 128 / 256 | `--max-concurrency`（客户端，无需重启） |
| **上下文长度** | 1024 / 2048 / 4096 | `--random-input-len`（客户端，无需重启） |

### 因变量（5 个）

1. `output_throughput`（tok/s，主指标）
2. `mean_tpot_ms` / `p99_tpot_ms`（解码延迟）
3. `mean_ttft_ms`（首 token）
4. **`vllm:num_preemptions_total` 增量**（机制证据，最关键）
5. **`vllm:gpu_cache_usage_perc`**（KV 是否吃满）

### 控制变量（写进报告，别让审稿人问）

- 模型、dtype（bf16）、TP 数、`--max-num-seqs 256`、`--max-model-len 4096`
- 输出长度固定 256 token（`--ignore-eos` 强制）
- `num_prompts = 2 × concurrency`（保证进入稳态）
- 每档配置先跑 1 次 warmup 丢弃，再跑 3 次取中位数
- 关闭 prefix caching（随机 prompt 无复用，开着只会引入噪声）

### 为什么主实验用 random 而不是 ShareGPT

ShareGPT 的输入长度不受控，做二维扫描时长度会成为一个隐变量。**固定长度的 random 数据集才是干净的自变量控制**。ShareGPT 只用来做一次"真实性校验"（确认趋势在真实分布上方向一致）。

---

## 4. 测量：preemption 才是这个实验的灵魂

只报吞吐，你只是在做性能测试；**报 preemption，你才是在做机理论证**。

启动后每轮采集（脚本已实现）：

```bash
curl -s localhost:8000/metrics | grep -E \
  "vllm:num_preemptions_total|vllm:gpu_cache_usage_perc|vllm:num_requests_(running|waiting)"
```

另外从启动日志抓真实 KV 容量，它是你图的真实横轴：

```
GPU KV cache size: 640,000 tokens
```

---

## 5. 执行阶段、耗时与成本

| Phase | 内容 | 配置数 | 预估耗时 | 卡时 | 租卡成本* |
|---|---|---|---|---|---|
| 0 | 环境 smoke test（能启动、能出数） | 2 | 0.5 h | 0.5 | ¥0.5 |
| 1 | Tier 0 OLMoE 全网格（1 卡） | 4×3×5 = 60 | 1.5 h | 1.5 | ¥2 |
| 2 | Tier 1 Qwen2.5-7B 全网格（1 卡，对照） | 60 | 2 h | 2 | ¥2 |
| 3 | **Tier 3 Mixtral TP=4 全网格（主线）** | 60 | 5–7 h | 20–28 | ¥25–30 |
| 4 | 关键点复测（3 个配置 × 3 次） | 9 | 1 h | 4 | ¥5 |
| | **合计** | | **1.5–2 个工作日** | | **≈ ¥35–40** |

\* 按 ¥1.09/卡·小时（A40 最低报价，2026-11 采集）估算。

> 整个实验的租卡成本 **不到 50 块**。如果组里有 4 卡机器，就是两个通宵的事。这也是为什么"没有 4 张卡"从来不是做不了这个题的原因。

---

## 6. 目录与命令

```
fluxmoe-repro/
├── README.md              # 本文件
├── run_sweep.sh           # 主驱动（启动/跑/采样/落表）
├── plot_results.py        # 出图（--demo 可看预期形状）
└── results/
    ├── results.csv        # 每行 = 一个配置
    └── figs/fig1..4.png
```

```bash
# 环境
pip install "vllm==0.10.2" transformers pandas matplotlib

# 先看预期输出长什么样（不需要 GPU）
python plot_results.py --demo

# 正式跑（Linux / WSL / Git Bash）
MODEL=allenai/OLMoE-1B-7B-0924-Instruct TP=1 UTILS="0.55 0.65 0.75 0.85" \
  bash run_sweep.sh
```

---

## 7. 四张图与判读标准

| 图 | 内容 | 通过判据 | 对应 claim |
|---|---|---|---|
| **Fig 1** | 吞吐 vs 并发；线 = context；分面 = KV 预算 | 出现拐点，且 KV 预算越大拐点越靠右 | S1 / S3 / C1 |
| **Fig 2** | 吞吐 vs 实际 KV 容量（GiB），固定 ctx=4096、conc=256 | 单调上升后饱和 | C1 |
| **Fig 3** | preemption 增量 vs 并发（双轴：吞吐） | 吞吐拐点与 preemption 起跳点重合 | S2 |
| **Fig 4** | 固定 conc=256，吞吐 vs context（多 KV 预算） | context↑ 吞吐↓，KV 预算大时下降变缓 | S4 / C2 / C4 |

**Fig 3 是整份报告的核心。** 如果 preemption 起跳点和吞吐拐点对得上，你就把机制的因果链钉死了。

---

## 8. 结果对不上时的诊断表

| 症状 | 最可能原因 | 处理 |
|---|---|---|
| 吞吐随并发**一路平坦**，不下降 | 没撞上 KV 墙：`gpu-memory-utilization` 太高 / context 太短 | 降 util 到 0.60，或把 ctx 拉到 4096、conc 拉到 256 |
| preemption **恒为 0** | vLLM 在换出前先排队（`num_requests_waiting` 涨） | 看 waiting 队列；这本身也是"KV 不足"的证据，可换算成有效并发 |
| TP=4 比 TP=2 **还慢** | A40 无 NVLink，PCIe all-reduce 成为新瓶颈 | 记录该现象，主线改 TP=2 + 更小模型；TP 影响要在报告里点名 |
| 启动就 OOM | util 低于权重占比 | Mixtral 4 卡时 util ≥ 0.55；仍 OOM 则降 `--max-num-seqs` |
| 数值抖动 > 20% | 客户端成为瓶颈（单线程发送）/ 热降频 | 用 `--request-rate inf` + 多进程客户端；记录 GPU 温度与 clock |
| 拐点位置比预测**晚很多** | vLLM 的 KV 块预留 / `max-model-len` 过大 | 设 `--max-model-len 4096`，让 profiling 更贴近真实 |

---

## 9. 预期结果（先算好，用来验证你的测量是正常的）

以 Mixtral TP=4、4×48G 为例（每卡 48G，权重每卡约 23.5G）：

| util | 每卡可用 KV | 总 KV | 可容纳 token | ctx=4096 时撞墙并发 |
|---|---|---|---|---|
| 0.60 | ~5.3 GB | ~21 GB | ~172k | 约 42 |
| 0.70 | ~10.1 GB | ~42 GB | ~344k | 约 84 |
| 0.80 | ~14.9 GB | ~63 GB | ~516k | 约 126 |
| 0.90 | ~19.7 GB | ~79 GB | ~640k | 约 156 |

> **预测：util=0.90 时，conc 128（524k token）能跑满，conc 256（1.05M token）必然 preempt。**
> 如果你实测拐点落在 128–256 之间，说明测量是对的，可以直接往下写；如果完全不在这个量级，先查第 8 节的诊断表，不要急着怀疑论文。

---

## 10. 交付物

1. `results/results.csv`：60+ 行原始数据（含真实 KV 容量、preemption 增量）
2. 四张图（PNG + 一份画图脚本，别人能复跑）
3. **一页结论**：四条判据的通过情况 + 与论文数字的差异 + 差异归因（硬件/模型/vLLM 版本）
4. 组会 8 页：
   - P1 动机（FluxMoE 想解决什么）
   - P2 它的 claim 与测试床
   - P3 我的复现目标与判据
   - P4 实验设计（三维网格）
   - P5 Fig 1 + Fig 3（主结果）
   - P6 Fig 2 + Fig 4（补充）
   - P7 与论文的差异及归因
   - P8 下一步（往 L2 走：单模块复现，或做 phase diagram）

---

## 11. 五个必须避开的坑

1. **不要一上来跑 Mixtral。** 先用 OLMoE 在 1 张卡上把整条流水线跑通，再上大模型。
2. **不要只报一个 batch 点。** 论文自己承认 batch=32 时它比 vLLM 慢 36%，**拐点才是结论**。
3. **不要在换 util 时不重启。** `gpu-memory-utilization` 是启动参数，必须重启服务；重启间隔要留足加载时间（Mixtral 约 60–90 s）。
4. **不要忽略 TP 的影响。** A40 无 NVLink，TP=4 的通信开销会压低绝对吞吐，这要在报告里明确说明，否则你会误以为"复现失败"。
5. **不要只做一次。** 每个配置至少 3 次，用中位数；GPU 热降频在长时间扫描里非常明显。

---

## 12. 做完之后可以往哪走

- **升级为 phase diagram**：把 batch × context × KV 容量三维扫完，画出"哪里 preempt、哪里不 preempt"，这就是论文 future work #1 提到的 expert paging phase diagram 的一个切片。
- **接你自己的方向**：这套 harness 可以直接换成"KV cache 压缩/淘汰方法"的评测，你上一个项目（统一评测）就有着落了。
- **写成 tech report**：复现 + 差异归因 + 边界条件，是有价值的 artifact 类工作，也能作为你后续论文的 motivation 素材。
