"""Static, decimal-GB capacity estimates; no device inspection or allocation.

This arithmetic is a feasibility filter, not an execution or OOM guarantee. The
source model sizes are rounded paper values and parameters are divided evenly
over ranks. Actual tensor-parallel replication and allocator costs can differ.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelSpec:
    name: str
    total_parameters: int
    expert_parameters: int
    moe_layers: int
    compressed_fraction: float


MODEL_SPECS = {
    "mixtral": ModelSpec("Mixtral-8x7B-Instruct", 47_000_000_000, 45_000_000_000, 32, 0.676),
    "qwen": ModelSpec("Qwen3-Next-80B-A3B-Instruct", 80_000_000_000, 77_000_000_000, 48, 0.679),
    "glm": ModelSpec("GLM-4.5", 355_000_000_000, 338_000_000_000, 89, 0.675),
}

GPU_PRESETS = {
    "a40": {
        "gpu_gb": 48.0,
        "source": "nominal A40 48 GB treated as decimal GB for a conservative planning estimate; not measured",
    },
    "l40s": {
        "gpu_gb": 48.3,
        "source": "FluxMoE paper v3 section 6.1 reported device capacity, decimal GB; not measured here",
    },
    "h20": {
        "gpu_gb": 102.6,
        "source": "FluxMoE paper v3 section 6.1 reported device capacity, decimal GB; not measured here",
    },
}

_ALIASES = {
    "mixtral-8x7b": "mixtral",
    "mixtral-8x7b-instruct": "mixtral",
    "mixtral-8x7b-instruct-v0.1": "mixtral",
    "qwen3-next": "qwen",
    "qwen3-next-80b-a3b": "qwen",
    "qwen3-next-80b-a3b-instruct": "qwen",
    "glm-4.5": "glm",
}


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def estimate_capacity(
    model: str,
    gpus: int,
    gpu_gb: float,
    cpu_fraction: float = 0.125,
    kv_gb_per_gpu: float = 0,
    workspace_gb_per_gpu: float = 0,
    resident: bool = False,
    budget_fraction: float = 0.9,
) -> dict:
    """Estimate GPU weight/buffer footprint plus explicit runtime allowances.

    ``cpu_fraction`` is the fraction of routed-expert BF16 parameters stored on
    the host; the remaining expert weights have compressed GPU backing. The
    pageable estimate includes two BF16 layer buffers and assumes all layers are
    pageable. In ``resident`` mode all BF16 weights remain on GPU and the CPU
    expert fraction is zero, regardless of the validated ``cpu_fraction`` input.

    Host KV budget is the paper's 512 GB aggregate budget, separately reported.
    Returned GB values use 1 GB = 1e9 bytes, never GiB.
    """
    if not isinstance(model, str):
        raise ValueError("model must be mixtral, qwen, glm, or a supported full model name")
    key = model.strip().lower().replace("×", "x")
    key = _ALIASES.get(key, key)
    if key not in MODEL_SPECS:
        raise ValueError(f"unknown model {model!r}; choose mixtral, qwen, or glm")
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus <= 0:
        raise ValueError("gpus must be a positive integer")
    if not isinstance(resident, bool):
        raise ValueError("resident must be a boolean")
    gpu_gb = _finite_number("gpu_gb", gpu_gb)
    cpu_fraction = _finite_number("cpu_fraction", cpu_fraction)
    kv_gb_per_gpu = _finite_number("kv_gb_per_gpu", kv_gb_per_gpu)
    workspace_gb_per_gpu = _finite_number("workspace_gb_per_gpu", workspace_gb_per_gpu)
    budget_fraction = _finite_number("budget_fraction", budget_fraction)
    if gpu_gb <= 0:
        raise ValueError("gpu_gb must be positive")
    if not 0 <= cpu_fraction <= 1:
        raise ValueError("cpu_fraction must be between 0 and 1")
    if kv_gb_per_gpu < 0 or workspace_gb_per_gpu < 0:
        raise ValueError("KV and workspace allowances must be nonnegative")
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be greater than 0 and at most 1")

    spec = MODEL_SPECS[key]
    p = spec.total_parameters / 1e9
    e = spec.expert_parameters / 1e9
    effective_cpu_fraction = 0.0 if resident else cpu_fraction
    breakdown = {
        "resident_bf16_gb_per_gpu": 2 * p / gpus if resident else 0.0,
        "nonexpert_bf16_gb_per_gpu": 0.0 if resident else 2 * (p - e) / gpus,
        "double_layer_buffer_gb_per_gpu": 0.0 if resident else 4 * e / (spec.moe_layers * gpus),
        "compressed_expert_gb_per_gpu": 0.0 if resident else 2 * e * spec.compressed_fraction * (1 - cpu_fraction) / gpus,
        "kv_gb_per_gpu": kv_gb_per_gpu,
        "workspace_gb_per_gpu": workspace_gb_per_gpu,
    }
    used = sum(breakdown.values())
    budget = gpu_gb * budget_fraction
    if not math.isfinite(used) or not math.isfinite(budget):
        raise ValueError("capacity arithmetic exceeds finite numeric range")
    caveats = [
        "Static estimate only: no model loading, GPU allocation, kernel execution, or peak-memory measurement occurred.",
        "Model parameter counts are rounded paper values; equal per-rank division does not prove actual TP/EP shard fit, replication, or partition divisibility.",
        "Default zero KV and workspace allowances do not describe a runnable deployment.",
        "Compression codebooks/indices, decode scratch, activation/communication buffers, allocator/page rounding, CUDA graph pools, and runtime metadata are omitted unless covered by the workspace allowance.",
        "GPU compressed backing and two BF16 layer buffers are included in pageable mode; extra transient compressed copies and checkpoint-loading peaks are not.",
        "Host expert storage is uncompressed BF16; the 512 GB host KV tier is an aggregate paper budget, not measured RSS or a complete host-RAM requirement.",
        "Host loading/staging copies, compressed host copies if retained, pinned-memory overhead, processes, and OS memory are excluded.",
        "Device capacity must use consistent decimal GB; nominal product labels and reported available capacity are not interchangeable.",
        "Capacity fit does not establish throughput, link bandwidth, topology compatibility, or reproduction of paper performance.",
    ]
    if key == "qwen":
        caveats.append(
            "Qwen3-Next has hybrid attention/state and TP-dependent KV-head replication; do not obtain per-GPU KV/state bytes by dividing a whole-model KV estimate by GPU count."
        )
    if resident:
        caveats.append("Resident mode ignores cpu_fraction for placement and includes neither compressed expert backing nor double-layer materialization buffers.")
    else:
        caveats.append("All routed-expert layers are assumed pageable; mixed resident/pageable placements require a separate calculation.")

    return {
        "kind": "static_capacity_estimate",
        "units": "decimal GB (1 GB = 1e9 bytes)",
        "model": spec.name,
        "model_key": key,
        "gpus": gpus,
        "gpu_gb": gpu_gb,
        "budget_fraction": budget_fraction,
        "resident": resident,
        "requested_cpu_fraction": cpu_fraction,
        "effective_cpu_fraction": effective_cpu_fraction,
        "compressed_fraction": spec.compressed_fraction,
        "budget_gb_per_gpu": budget,
        "used_gb_per_gpu": used,
        "headroom_gb_per_gpu": budget - used,
        "fits_static_budget": used <= budget,
        "breakdown": breakdown,
        "host_expert_gb": 2 * e * effective_cpu_fraction,
        "host_kv_budget_gb": 512.0,
        "host_expert_plus_kv_budget_gb": 2 * e * effective_cpu_fraction + 512.0,
        "caveats": caveats,
    }
