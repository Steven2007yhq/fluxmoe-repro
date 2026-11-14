# A40 服务器只读审计与复现实验范围

审计日期：2026-10-31（America/Los_Angeles）。已通过 SSH 成功登录并完成只读检查，随后退出。未安装软件、未上传模型/项目、未启动 benchmark、未改变共享源码或目录权限。原始输出及实际执行的检查脚本保存在 [审计目录](/run/media/steve/Lenovo/FluxMoE论文复现/results/server_audit_20261031)。登录密码未写入这些产物。

## 已确认资源

| 项目 | 只读检查结果 | 对实验的含义 |
| --- | --- | --- |
| GPU | 4× NVIDIA A40，SM 8.6，ECC 开启；每卡驱动可见 46,068 MiB（44.99 GiB / 48.305 GB） | 不能把四卡合计容量当单卡连续显存 |
| 当前 GPU 占用 | 每卡 14 MiB used、45,477 MiB free（44.41 GiB），利用率 0%；没有 compute-app 进程，仅 Xorg | 检查时空闲，不代表已经为本项目预留 |
| GPU 拓扑 | 0↔1、2↔3 为 NV4；跨配对为 SYS | 此前“无 NVLink”假设已被实测推翻。两卡优先用一个配对；四卡需处理跨 NUMA/系统互联通信 |
| PCIe | 最大 Gen4×16；空闲快照当前 Gen1×16 | 当前省电链路状态不是负载下带宽；未运行 H2D/P2P/NCCL 带宽测试 |
| CUDA 能力 | 各卡 VMM=1、unified-addressing=1、async-engine-count=3；全部有向卡对 peer-access=1 | 支持继续验证 VMM/异步分页，能力标志不证明实际带宽、重叠收益或项目算子已通过 |
| CPU | 系统报告 AMD Ryzen Threadripper PRO 5995WX，128 cores / 256 logical CPUs，8 NUMA nodes | 这是 OS 报告，未用机箱/固件验证物理处理器配置；NUMA 绑定必须依据实测而非只看名称 |
| 主存 | 540,797,251,584 bytes total（503.66 GiB），528,231,120,896 bytes available（491.95 GiB） | 约 512 GiB 级机器，明显小于论文的 2 TB 节点；当前 session 的 memory.max 未设限 |
| NUMA 内存 | 节点 0、3、4、7 有内存，1、2、5、6 为 0；有效 memory cpuset 为 0,3-4,7 | GPU1/2 相邻 CPU 节点无本地内存，后续应查距离并实测 pinned-host 放置；当前未安装 numactl |
| `/data` | ext4，约 23.903 TB / 21.74 TiB 空闲；底层盘报告 WDC WUH722424ALE6L4、ROTA=1 | 容量充足；机械盘影响冷启动。稳态实验不应意外把逐 token 磁盘读取混入 CPU/GPU 专家物化路径 |
| 根目录 | SSD，约 700.6 GB / 652.52 GiB 空闲 | 当前 home 在根文件系统；大检查点仍应按导师要求放 `/data` |
| 驱动 / Toolkit | NVIDIA driver 595.80；`/usr/local/cuda → cuda-13.2`；nvcc 13.2.78、cudart 13.2.75；头文件齐全 | 工具链存在，当前 SSH PATH 没有 nvcc；不能把 PATH 未配置误报为没有 CUDA |
| Profilers | `/usr/local/cuda/bin/nsys` 2025.6.3.541；`ncu` 2026.1.1.0 | 工具可执行；尚未验证性能计数器权限或进行 profiler 采集 |

依据：[hardware.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/server_audit_20261031/hardware.json)、[software.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/server_audit_20261031/software.json)、[extra.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/server_audit_20261031/extra.json)、[driver.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/server_audit_20261031/driver.json)。驱动属性查询没有创建模型、分配模型张量或发射 kernel；其进程退出后再次核对各卡恢复 14 MiB used / 45,477 MiB free。

## 环境和目录的实际前置条件

当前账号为 `bks-agent`，只属于同名组。`/data` 与 `/data/share` 是 `wangyc` 组可写，账号不在该组，也没有额外 ACL；`os.access(..., W_OK)` 均为 false。因此当前不能在这里新建本项目的模型/结果目录。后续需要管理员提供一个归该账号可写的专用子目录，例如 `/data/bks-agent/fluxmoe`。本轮没有尝试创建文件来探测权限，也未修改权限。

当前账号的 PATH 和常见安装位置未发现可复用的 Conda/Mamba 环境。系统 Python 为 3.10.12，当前解释器没有 pip、torch、vLLM、SGLang、Transformers 或 Triton 包元数据；这只说明本次可见账号环境，不能据此断言其他账号没有环境。

`/data/share/sglang` 是只读可访问源码，commit 为 `4b1941e8891fdef3d99d77a71e537a836561a85b`。该 checkout 的 `python/pyproject.toml` 要求 Python≥3.11、torch==2.14.1；存在源码不表示已经安装或能直接用系统 Python 运行。当前 FluxMoE 项目接入的是 vLLM 0.29.0，SGLang 适配是独立后续工作。

后续部署应使用专用环境并固定完整依赖，不拷贝本机 Python 3.14 环境二进制作为服务器环境。vLLM 0.29 的官方安装说明也建议使用新环境并保持 CUDA/PyTorch 二进制配置匹配，服务器 A40 的 SM8.6 满足其列出的最低 GPU 能力要求。[vLLM 0.29 GPU 安装说明](https://docs.vllm.ai/en/v0.29.0/getting_started/installation/gpu/)

服务器驱动与 CUDA 13.2 的对应版本要求相符；但这不等于任意 CUDA 13.4 PTX 都能直接迁移运行。应在目标机为 `sm_86` 重新构建项目 native 库，并先完成最小正确性检查。CUDA minor compatibility 对 PTX 和新增功能存在限制。[CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/)、[NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

## 模型范围：资源可规划与实现可执行分别判断

下表是算术规划，不是实测分配。参数量/压缩比例沿用论文取整数据，实际每卡容量使用本次驱动报告；90% 预算约 **43.475 GB/卡**。举例另留每卡 8 GB KV/state 和 2 GB workspace；这 10 GB 只是规划假设，未证明足够。完整计算在 [capacity_estimates.json](/run/media/steve/Lenovo/FluxMoE论文复现/results/server_audit_20261031/capacity_estimates.json)。

| 目标 | 权重及分页双缓冲下界，GB/卡 | 加示例 8+2 GB 后 | 判定 |
| --- | ---: | ---: | --- |
| 单卡小型 Mixtral | 依实际 checkpoint 计算 | 需实际量峰值 | 最先迁移现有加载/KV/生成正确性入口；本机 4060 已通过，服务器尚未运行 |
| 2×A40，Mixtral-8×7B，12.5% CPU 专家 | 31.43 | 41.43 | 主线受限显存实验资源可规划；需要合法 TP/EP、流式超显存加载和运行态峰值验证 |
| 4×A40，Mixtral-8×7B 全驻留 | 23.50 | 33.50 | 适合建立全驻留对照、质量验证及容量实验；项目仍需补多 rank |
| 4×A40，Qwen3-Next-80B，全 GPU 压缩专家 | 29.25 | 39.25 | 资源上可进一步规划；当前不支持 Qwen 架构、多 rank 和 hybrid state，不能直接运行 |
| 4×A40，Qwen3-Next-80B 全驻留 | 40.00 | 50.00 | 权重本身接近预算，示例运行态不够；应先核验实际 KV/state，不能保证大批次基线可部署 |
| 4×A40，GLM-4.5，12.5% CPU 专家 | 112.11 | 122.11 | 明显超过本机预算，无法按论文设置做完整端到端主实验；可读取小部分真实权重做分布/压缩微实验 |

现有 loader 仍先将完整模型加载到一张 GPU，再换为分页权重；引擎还显式拒绝 TP/EP、多 rank、KV offload 和非 Mixtral 架构。因此上表的多卡项是后续实现目标，不能直接拿当前入口启动原检查点。

CPU 主存还必须另算。论文使用 512 GB aggregate host-KV，加 12.5% CPU 专家后，Mixtral/Qwen/GLM 的静态主存项分别为 523.25/531.25/596.5 GB。对本机当前 528.23 GB available，Mixtral仅余约4.98 GB，Qwen和GLM已超；这还没计 CPU 恢复副本、checkpoint 暂存、压缩构建、进程和 pinned 管理。若“512”按 GiB，则 KV 池本身已超过本机可见总内存。

本平台应先设计统一较小的 host-KV 预算（例如按峰值核验后选择128或256 GB），对所有方法使用相同预算并标为 A40 移植条件。不得声称完整复刻论文的 512 GB host-KV / 2 TB RAM 条件。

## 最终图表交付要求

已将用户要求固定为：**每组复现实验必须交付与论文对应的柱状图、曲线图或 CDF，而非只交日志和数值 JSON。** 全图号、原条件与实现缺口见 [PAPER_EXPERIMENT_MAP.md](/run/media/steve/Lenovo/FluxMoE论文复现/docs/PAPER_EXPERIMENT_MAP.md)。

| 优先级 | 论文对应 | 本平台拟交付图 | 开始正式测量前的条件 |
| --- | --- | --- | --- |
| 1 | 表3、图11 | 真实 BF16 专家的压缩存储比例、解码吞吐/延迟柱图；exponent/mantissa 分布图 | 专用环境、可写数据目录、真实权重 revision、A40 kernel 正确性与无诊断同步的测量入口 |
| 2 | 图5 / Exp#2 | Mixtral 两卡的吞吐、TTFT、TPOT 分组柱图，分别扫描 batch 与输出长度 | 超显存流式加载、TP/EP、同预算的 FluxMoE-H/vLLM-O/KT 对照、正式请求与 KV 驱动 |
| 2 | 图6、表4/5 / Exp#4 | forward/active-batch 柱图，CPU H2D/解码/物化和暴露 stall 的机制图 | 同预算、真实多 stream 完成依赖，性能测量与 profiler 分开 |
| 3 | 图7/8 / Exp#5–6 | CI/AG 扫描及 planner 开关的逐 iteration 吞吐曲线 | 真实 GPU planner、安全释放和 KV 预算联动；图8多次 launch 模拟需保持论文统计边界 |
| 3 | 图9/14 / Exp#7 | 多请求率下的 E2E latency CDF | 在线服务驱动、固定到达轨迹、预热/成功/失败及排空协议 |
| 配套 | 表7 / Exp#8 | 预训练 Mixtral 的 WikiText-2 perplexity 对照表，可附柱图 | 同 checkpoint/tokenizer，原评分窗口协议，4卡合法分片，不能用随机 tiny 代替 |

每张最终图提供 PNG 和矢量 PDF/SVG，附图号、平台差异、原始 JSONL/CSV、作图入口和统计口径。正式柱图按论文五次独立运行协议报告均值与95%置信区间；曲线和CDF按相应实验单位统计。OOM/超时/未支持点单独标记，不能画成0吞吐参与加速比；论文数字与本机测量分列。没有收益也如实呈现。

本轮只读审计没有性能样本，所以没有绘制声称复现收益的图。此时可确认的是资源、环境缺口和出图范围；下一开发阶段应依次解决可写数据目录、隔离环境、单卡 A40 正确性、超显存加载、多卡与公平基线，再采集正式数据。
