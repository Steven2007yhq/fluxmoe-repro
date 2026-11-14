"""Rebuild inspectable summaries and Chrome traces from saved raw records."""
from __future__ import annotations

import json
from pathlib import Path


def chrome_trace(result: dict) -> dict:
    return {
        "displayTimeUnit": "ms",
        "metadata": {"timing": "SIMULATED, NOT MEASURED", "backend": result.get("backend")},
        "traceEvents": [
            {"name": event["name"], "cat": event["metadata"].get("kind", "operation"),
             "ph": "X", "ts": event["start_ms"] * 1000, "dur": event["duration_ms"] * 1000,
             "pid": 1, "tid": event["stream"], "args": {**event["metadata"], "dependencies": event["dependencies"]}}
            for event in result.get("trace", [])
        ],
    }


def render_report(results: list[dict]) -> str:
    lines = [
        "# FluxMoE 独立实现：模拟验证记录", "",
        "这些结果来自真实 BF16 编解码、真实小型 MoE 数值计算和离散事件调度。",
        "**表中时间全部是模拟时间，不是 A40、L40S、H20 或本机 GPU 的性能测量。**", "",
        "| 实验 | 策略 | 预取 | batch | 状态 | 模拟时间 ms | 模拟等待 ms | GPU 预算内字节 | 数值误差 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, result in enumerate(results):
        config = result.get("config", {})
        metrics = result.get("metrics")
        if metrics is None:
            elapsed, stall, memory, error = "—", "—", "—", "—"
        else:
            elapsed = f"{metrics['modeled_elapsed_ms']:.6f}"
            stall = f"{metrics['modeled_exposed_stall_ms']:.6f}"
            memory = str(max(m["device_total"] for m in result["memory_by_iteration"]))
            error = str(result["correctness"]["max_abs_error"])
        lines.append(f"| {result.get('experiment_id', i)} | {config.get('strategy', '')} | {config.get('prefetch', '')} | {config.get('batch', '')} | {result['status']} | {elapsed} | {stall} | {memory} | {error} |")
    lines.extend([
        "", "预算内字节是逻辑分配账本，包含压缩格式头、物理权重缓冲、router、固定 KV 和 workspace；不是进程 RSS 或 nvidia-smi。",
        "OOM 保留为状态，时间与吞吐缺失，不填 0 或用于计算加速比。",
        "各策略使用相同预算上限；整层舍入可能使实际 CPU/GPU 占用不同，详见原始 JSON。",
        "驻留策略是相同 toy MoE 的参考路径；这里的策略不是 vLLM/KTransformers 产品实现。", "",
        "## 实现边界", "",
        "- 当前模拟器为单 rank，未运行原始三个大模型、真实请求队列、注意力或 TP/EP 通信。",
        "- CPU 参考以 BF16 存储权重、Python float 累加；可选 PyTorch oracle 独立核对数值。",
        "- H2D 和解码分别为共享串行资源，可以与计算重叠；未建模 SM/DRAM/KV 换入换出竞争。",
        "- 规划器验证元数据版本与决策；迁移新副本的准备开销和瞬时双份内存未纳入时间/容量模型。",
        "- 本次配置中的 bandwidth、compute time 均为显式假设，不能外推论文加速比。", "",
    ])
    return "\n".join(lines)


def load_results(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else [value]
