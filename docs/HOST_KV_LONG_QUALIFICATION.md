# 四卡 BF16 长负载 KV 换出资格

入口使用 Mixtral-8x7B-Instruct-v0.1 的固定离线权重和 ShareGPT 请求清单。每次尝试依次运行两个新引擎，各执行 32 个请求、每个请求 4,096 个输出 token。两个引擎均为四张 A40、TP2/PP2/EP2/DP1、BF16、eager、TRITON_ATTN、LBHNC 和 native OffloadingConnector。参考回放使用独立引擎和新缓存，沿用同一 native 调度与传输协议。此资格验证 host KV 正确性和同布局可重复性。两卡 FluxMoE-H/B、resident-only KV 参考和正式性能属于其他实验。

## 验收

- 同一请求用独立 cache salt 和实际 token 历史绑定。至少一个全由解码生成的逻辑块须经历真实 GPU→host 完成、旧 GPU 块释放、host→GPU 完成及实际 attention 读回。四个 rank 各保存其 16 层的原始 GPU、host、恢复后 GPU 与使用时的 KV 位型。审计从原件重算逐字节相等，跨请求缓存命中和重算不能代替这条证据。
- 两个引擎分别完成全部请求和四 rank CUDA/PP fence，并保留原父进程记录的 worker 退出码。只有第一侧完成、取证通过并退出，才启动参考。对照核验有序 InputBatch、实际 token/位置和 KV 布局、完整生成历史、六个输出位置（1、512、1,024、2,048、3,072、4,096）及取证请求的首次恢复位置。对照保留完整词表 logits 和 32 层采样激活。
- GPU 池默认 4,096 个 16-token 块，约 2 GiB/rank。CPU 池共同上限 18 GiB 聚合。启动核对四张目标卡空闲、192 GiB 可用 RAM、32 GiB 工作盘余量和 host KV 共享内存。执行保留 160 GiB 归属进程 RSS 之和与 16 GiB 工作目录上限。共享映射可能计入多个进程的 RSS。证据采集与 native 完成事件等待用于正确性诊断。

## 在四卡目标机执行

从源码根目录运行，使用已经安装契约所列 vLLM 0.29.0 和 Torch 的目标 Python。准备阶段读取随源码分发的请求清单与 vLLM 源码契约。模型必须是现有的离线目录，包含 `download_manifest.json`、19 个 safetensors 分片及 tokenizer。预检逐文件复算模型 SHA256。

```bash
python -m fluxmoe.host_kv_qualification prepare \
  --checkpoint /home/bks-agent/fluxmoe-data/mixtral-full-offline-eba92302.2jipyb54 \
  --gpu-uuids \
    GPU-e81a0988-c3ff-59b8-8589-bee2898ca21e \
    GPU-52772f42-b234-dfc4-8f1d-dd535086e8ee \
    GPU-49a16f24-98bf-794a-4ad0-e62d6dff5dec \
    GPU-ea48857b-eef3-58d7-2b1d-02dc57f56d46 \
  --output /tmp/fluxmoe-host-kv-long.json

work=/home/bks-agent/fluxmoe-runs/host-kv-long-001
python -m fluxmoe.host_kv_execution run \
  --config /tmp/fluxmoe-host-kv-long.json --work "$work" --timeout 7200
```

长任务可在已有 tmux 会话中运行。从另一个终端用 `status` 检查同一工作目录。每个工作目录只接受一次启动，失败或中断也保留该尝试。7200 秒是两次模型执行和预检的合计上限。它是本轮执行预算，实际耗时有待四卡运行。执行状态和日志分别在 `control/execution.json` 和 `control/worker.log`。成功要求监督器状态为 `success`，且 `result.json` 为 `HOST_KV_LONG_QUALIFICATION_PASSED`。

```bash
python -m fluxmoe.host_kv_execution status --work "$work"
python -m fluxmoe.host_kv_execution export \
  --work "$work" --archive "${work}.tar.gz"
```

进程组闭合后可导出成功或失败原件。将归档和同名追加 `.receipt.json` 的回执拷回本机，并从目标机导出输出保留回执 SHA256。恢复目录必须是新目录。

```bash
python -m fluxmoe.host_kv_execution recover \
  --archive host-kv-long-001.tar.gz \
  --receipt host-kv-long-001.tar.gz.receipt.json \
  --receipt-sha256 "$receipt_sha256" \
  --output ./received-host-kv-long-001

python -m fluxmoe.host_kv_qualification audit --work ./received-host-kv-long-001
```

`recover` 校验回执、清单、卷和所有成员。完整成功归档可继续用 `audit` 重算数值与逐位证据，后者需要 NumPy。失败归档可直接检查日志、状态和已经提交的证据文件。

## 核验状态

本机验证使用 CPU 夹具及已安装 vLLM 源码。上述四卡真实负载仍待运行。历史 A1 的短资格保存在原归档。
