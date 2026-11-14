"""Execute real toy MoE arithmetic through a modeled asynchronous paging path.

Event durations are inputs to a discrete-event model. Python execution time is
never reported as GPU performance. Device memory accounting is a logical ledger;
the independent host oracle and fixture files are kept outside that ledger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

from .codec import compress_bf16, decompress_bf16
from .events import Event, Timeline
from .model import ModelConfig, ToyMoE
from .paging import PagedTensorPool
from .planner import Planner, ResidencyState, TensorKey, TensorSpec, initial_bandwidth_placement


STRATEGIES = ("resident", "compressed", "hybrid", "whole_layer_cpu", "whole_layer_hybrid")


@dataclass(frozen=True)
class SimulationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: str = "hybrid"
    batch: int = 4
    iterations: int = 4
    remaining_batch: int = 16
    prefetch: bool = True
    cpu_fraction: float = 0.125
    # Deliberately scaled to small tensors, NOT calibrated A40 specifications.
    cpu_bandwidth_gbps: float = 0.008
    decompression_bandwidth_gbps: float = 0.080
    compressed_read_bandwidth_gbps: float = 0.200
    launch_ms: float = 0.005
    expert_ms_per_token: float = 0.030
    nonexpert_ms: float = 0.030
    gpu_budget_bytes: int = 2_000_000
    kv_bytes: int = 8192
    workspace_bytes: int = 4096
    planner_enabled: bool = False
    control_interval: int = 16
    adjustment_granularity: float = 0.25
    restore_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        for name in ("batch", "iterations", "remaining_batch", "gpu_budget_bytes", "control_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.remaining_batch < self.batch:
            raise ValueError("remaining_batch must be >= batch")
        for name in ("kv_bytes", "workspace_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("cpu_bandwidth_gbps", "decompression_bandwidth_gbps", "compressed_read_bandwidth_gbps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("launch_ms", "expert_ms_per_token", "nonexpert_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.cpu_fraction) or not 0 <= self.cpu_fraction <= 1:
            raise ValueError("cpu_fraction must be in [0, 1]")
        if not isinstance(self.prefetch, bool) or not isinstance(self.planner_enabled, bool):
            raise ValueError("prefetch and planner_enabled must be boolean")
        if self.planner_enabled and self.strategy != "hybrid":
            raise ValueError("online tensor planning is supported only for the hybrid strategy")
        Planner(self.control_interval, self.adjustment_granularity, self.restore_threshold)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SimulationConfig:
        data = dict(values)
        if "model" in data:
            data["model"] = ModelConfig(**data["model"])
        return cls(**data)


class BudgetExceeded(RuntimeError):
    """The logical device/host allocation exceeds the configured budget."""


def key_name(key: TensorKey) -> str:
    return f"layer{key.layer}/expert{key.expert}/{key.projection}"


class FluxMoESimulator:
    """One GPU rank's homogeneous layer pipeline and iteration metadata.

    CPU copies share one exclusive H2D resource; both decompression streams
    share one decoder resource. Compute overlaps those resources. This limits
    aggregate bandwidth without claiming to model SM/DRAM contention.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.model = ToyMoE(config.model)
        self.compressed = {key: compress_bf16(value) for key, value in self.model.weights.items()}
        self.specs = [TensorSpec(key, len(raw), len(self.compressed[key]))
                      for key, raw in self.model.weights.items()]
        total_raw = sum(spec.raw_bytes for spec in self.specs)
        self.cpu_budget = math.floor(total_raw * config.cpu_fraction)
        residents: set[int] = set()
        cpu_keys: set[TensorKey] = set()
        if config.strategy == "resident":
            residents = set(range(config.model.layers))
        elif config.strategy == "hybrid":
            cpu_keys = set(initial_bandwidth_placement(
                self.specs, config.cpu_bandwidth_gbps,
                config.decompression_bandwidth_gbps, self.cpu_budget,
            ))
        elif config.strategy in ("whole_layer_cpu", "whole_layer_hybrid"):
            remaining = self.cpu_budget
            for layer in reversed(range(config.model.layers)):
                group = [spec for spec in self.specs if spec.key.layer == layer]
                volume = sum(spec.raw_bytes for spec in group)
                if volume <= remaining:
                    cpu_keys.update(spec.key for spec in group)
                    remaining -= volume
            if config.strategy == "whole_layer_cpu":
                residents = set(range(config.model.layers)) - {key.layer for key in cpu_keys}
        self.state = ResidencyState(self.specs, residents, cpu_keys, self.cpu_budget)
        self.pool = PagedTensorPool(self.model.layer_sizes(), residents)
        self.timeline = Timeline()
        self.planner = Planner(config.control_interval, config.adjustment_granularity, config.restore_threshold)
        for key, raw in self.model.weights.items():
            if key.layer in residents:
                self.pool.write(key.layer, key.expert, key.projection, raw)
        self._has_run = False
        self.initial_memory = self.memory()
        self._check_memory()

    def memory(self, backends: Any = None) -> dict[str, int]:
        backends = self.state.backends if backends is None else backends
        compressed = sum(len(self.compressed[key]) for key, backend in backends.items() if backend == "compressed")
        cpu = sum(len(self.model.weights[key]) for key, backend in backends.items() if backend == "cpu")
        router_bytes = self.config.model.layers * self.config.model.experts * self.config.model.hidden * 2
        total = self.pool.physical_bytes + compressed + router_bytes + self.config.kv_bytes + self.config.workspace_bytes
        return {
            "physical_weight_buffers": self.pool.physical_bytes,
            "compressed_backing_including_codec_headers": compressed,
            "uncompressed_router_weights": router_bytes,
            "kv_fixed_reservation": self.config.kv_bytes,
            "workspace_reservation": self.config.workspace_bytes,
            "device_total": total,
            "device_budget": self.config.gpu_budget_bytes,
            "host_experts": cpu,
            "host_expert_budget": self.cpu_budget,
        }

    def _check_memory(self, backends: Any = None) -> None:
        memory = self.memory(backends)
        if memory["device_total"] > memory["device_budget"]:
            raise BudgetExceeded(f"device requires {memory['device_total']} bytes, budget {memory['device_budget']}")
        if memory["host_experts"] > memory["host_expert_budget"]:
            raise BudgetExceeded("host expert budget exceeded")

    def _load_layer(self, layer: int, snapshot: Any, iteration: int,
                    issue_after: Event | None) -> list[Event]:
        if layer in self.state.resident_layers:
            return []
        events = []
        for projection in ("gate_up", "down"):
            block = self.pool.block(layer, projection)
            barriers = [event for event in (issue_after, block.last_read) if event is not None]
            for backend in ("compressed", "cpu"):
                keys = [key for key, value in snapshot.backends.items()
                        if key.layer == layer and key.projection == projection and value == backend]
                if not keys:
                    continue
                raw_bytes = sum(len(self.model.weights[key]) for key in keys)
                compressed_bytes = sum(len(self.compressed[key]) for key in keys)
                if backend == "cpu":
                    duration = raw_bytes / (self.config.cpu_bandwidth_gbps * 1_000_000)
                else:
                    duration = max(
                        raw_bytes / (self.config.decompression_bandwidth_gbps * 1_000_000),
                        compressed_bytes / (self.config.compressed_read_bandwidth_gbps * 1_000_000),
                    )
                duration += self.config.launch_ms
                readers = [self.state.metadata.begin_read(snapshot, key) for key in keys]

                def materialize(keys=tuple(keys), readers=tuple(readers), backend=backend):
                    for key, token in zip(keys, readers):
                        data = self.model.weights[key] if backend == "cpu" else decompress_bf16(self.compressed[key])
                        self.pool.write(key.layer, key.expert, key.projection, data)
                        self.state.metadata.complete_read(token)

                event = self.timeline.submit(
                    f"{backend}:layer{layer}:{projection}", duration,
                    f"load/{projection}/{backend}", wait_for=barriers, action=materialize,
                    resource="h2d_link" if backend == "cpu" else "decoder",
                    metadata={"kind": "load", "layer": layer, "iteration": iteration,
                              "projection": projection, "backend": backend,
                              "metadata_version": snapshot.version, "raw_bytes": raw_bytes,
                              "issued_ms": max(self.timeline.now_ms, issue_after.end_ms if issue_after is not None else self.timeline.now_ms),
                              "source_bytes": raw_bytes if backend == "cpu" else compressed_bytes,
                              "tensor_count": len(keys), "block_id": block.id},
                )
                # Check against the prior reader before future readers are installed.
                self.pool.assert_write_safe(layer, projection, event.start_ms)
                events.append(event)
        return events

    def run(self) -> dict[str, Any]:
        if self._has_run:
            raise RuntimeError("create a new simulator for each independent run")
        self._has_run = True
        records, planner_records, memory_records, retired_records = [], [], [], []
        address_map = {key: self.pool.tensor(key.layer, key.expert, key.projection).virtual_address
                       for key in self.model.weights}
        all_outputs = []
        total_stall = 0.0
        for iteration in range(1, self.config.iterations + 1):
            self.state.publish_iteration()
            retired = self.state.metadata.drain_reclaimable()
            retired_records.extend({"key": key_name(copy.key), "backend": copy.backend,
                                    "generation": copy.generation, "version": copy.retired_at_version}
                                   for copy in retired)
            self._check_memory()
            memory_records.append(self.memory())
            snapshot = self.state.snapshot()
            started = self.timeline.now_ms
            inputs = self.model.inputs(self.config.batch, iteration)
            activation = {"value": inputs}
            routing: list[list[int]] = []
            ready: dict[int, list[Event]] = {}
            compute: list[Event] = []
            if self.config.prefetch:
                for layer in range(min(2, self.config.model.layers)):
                    ready[layer] = self._load_layer(layer, snapshot, iteration, None)
            iteration_stall = 0.0
            layer_load_ms = []
            for layer in range(self.config.model.layers):
                previous = compute[-1] if compute else None
                if not self.config.prefetch:
                    ready[layer] = self._load_layer(layer, snapshot, iteration, previous)
                nonexpert = self.timeline.submit(
                    f"nonexpert:layer{layer}", self.config.nonexpert_ms, "compute",
                    wait_for=() if previous is None else (previous,),
                    metadata={"kind": "nonexpert", "layer": layer, "iteration": iteration},
                )

                def forward(layer=layer):
                    activation["value"], counts = self.model.forward_layer(layer, activation["value"], self.pool.read)
                    routing.append(counts)

                event = self.timeline.submit(
                    f"expert:layer{layer}", self.config.batch * self.config.expert_ms_per_token,
                    "compute", wait_for=(nonexpert, *ready[layer]), action=forward,
                    metadata={"kind": "compute", "layer": layer, "iteration": iteration,
                              "metadata_version": snapshot.version},
                )
                iteration_stall += max(0.0, event.start_ms - nonexpert.end_ms)
                compute.append(event)
                for projection in ("gate_up", "down"):
                    self.pool.mark_read(layer, projection, event)
                layer_load_ms.append(
                    max(load.end_ms for load in ready[layer]) - min(load.metadata["issued_ms"] for load in ready[layer])
                    if ready[layer] else 0.0
                )
                if self.config.prefetch and layer + 2 < self.config.model.layers:
                    ready[layer + 2] = self._load_layer(layer + 2, snapshot, iteration, event)
            self.timeline.run_until(compute[-1])
            expected = self.model.forward(inputs)
            error = max(abs(x - y) for row, ref in zip(activation["value"], expected) for x, y in zip(row, ref))
            if error != 0.0:
                raise AssertionError(f"paged output differs from resident oracle: {error}")
            if any(self.pool.tensor(key.layer, key.expert, key.projection).virtual_address != address
                   for key, address in address_map.items()):
                raise AssertionError("virtual identity changed within an engine launch")
            all_outputs.append(activation["value"])
            total_stall += iteration_stall
            records.append({"iteration": iteration, "metadata_version": snapshot.version,
                            "modeled_elapsed_ms": self.timeline.now_ms - started,
                            "modeled_exposed_stall_ms": iteration_stall,
                            "layer_materialization_ms": layer_load_ms,
                            "max_abs_error": error, "routing_counts": routing})
            if self.config.planner_enabled:
                decision = self.planner.observe(
                    self.state, iteration=iteration, active_batch=self.config.batch,
                    remaining_batch=self.config.remaining_batch,
                    forward_ms=self.timeline.now_ms - started, load_ms=layer_load_ms,
                )
                applied = False
                reason = decision.reason
                if decision.changes and not decision.requires_relaunch and iteration < self.config.iterations:
                    proposed = dict(self.state.backends)
                    proposed.update(decision.changes)
                    try:
                        self._check_memory(proposed)
                    except BudgetExceeded:
                        reason = "rejected_device_budget"
                    else:
                        self.planner.apply(self.state, decision)
                        applied = True
                planner_records.append({
                    "iteration": iteration, "action": decision.action, "reason": reason,
                    "rho": decision.rho if decision.rho is None or math.isfinite(decision.rho) else "infinity",
                    "keys": [key_name(key) for key in decision.keys], "applied_for_next_iteration": applied,
                    "requires_relaunch": decision.requires_relaunch,
                })
        self.timeline.run_until()
        trace = self.timeline.trace()
        load_events = [event for event in trace if event["metadata"].get("kind") == "load"]
        output_hash = hashlib.sha256(json.dumps(all_outputs, allow_nan=False).encode()).hexdigest()
        elapsed = self.timeline.total_ms
        return {
            "schema_version": 1, "status": "passed", "backend": "cpu_reference_discrete_event_simulator",
            "measurement_kind": "modeled_time_not_gpu_benchmark", "config": asdict(self.config),
            "model_fingerprint": self.model.fingerprint(), "output_sha256": output_hash,
            "correctness": {"reference": "resident BF16 weights, Python float arithmetic",
                            "max_abs_error": max(record["max_abs_error"] for record in records),
                            "stable_virtual_identities": True},
            "metrics": {"modeled_elapsed_ms": elapsed, "modeled_exposed_stall_ms": total_stall,
                        "executed_token_vectors": self.config.batch * self.config.iterations,
                        "modeled_vectors_per_second": self.config.batch * self.config.iterations * 1000 / elapsed if elapsed else None,
                        "h2d_raw_bytes": sum(event["metadata"]["raw_bytes"] for event in load_events if event["metadata"]["backend"] == "cpu"),
                        "decoded_raw_bytes": sum(event["metadata"]["raw_bytes"] for event in load_events if event["metadata"]["backend"] == "compressed")},
            "pool": self.pool.summary(), "initial_memory": self.initial_memory,
            "memory_by_iteration": memory_records, "iterations": records,
            "planner": planner_records, "reclaimed_copies": retired_records, "trace": trace,
            "limitations": [
                "Single rank, synthetic MoE; no attention, tokenizer, trained checkpoint, TP/EP collective, or serving scheduler.",
                "BF16 weights are exact; activations accumulate as Python floats, not a BF16 CUDA kernel.",
                "All timings and bandwidths are model inputs; defaults are scaled toy values, not A40 measurements.",
                "Shared H2D and decoder resources serialize loads; SM, DRAM and KV-swap contention are not modeled.",
                "KV capacity is a fixed reservation; remaining_batch is a control input, not a simulated request queue.",
                "Device ledger includes codec headers and two layer buffers; Python oracle copies and backend metadata are outside it.",
                "Planner migration preparation time and transient old/new backing-copy allocation peaks are not modeled; metadata lifetime is verified separately.",
                "No in-launch layer remapping or real vLLM integration; ordinary tensor metadata changes publish next iteration.",
            ],
        }


def run_simulation(config: SimulationConfig) -> dict[str, Any]:
    try:
        return FluxMoESimulator(config).run()
    except BudgetExceeded as exc:
        return {"schema_version": 1, "status": "oom", "backend": "cpu_reference_discrete_event_simulator",
                "measurement_kind": "modeled_time_not_gpu_benchmark", "config": asdict(config),
                "reason": str(exc), "metrics": None}
