"""Rank-local FluxMoE residency control and a CPU-testable metadata lifetime model.

The switching rule, granularity and placement order follow paper sections 4.2--4.3.
The paper does not specify the resident-layer selection order; this implementation
reclaims the highest numbered resident layer first.  Construction establishes the
restoration floor: only recorded reclamations can be undone, in LIFO order.

This module models allocation lifetime; it does not allocate GPU/host storage or
perform CUDA synchronization.  A real backend must prepare new storage before
publishing metadata and translate completed reads into stream-local CUDA events.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping


BACKENDS = frozenset(("resident", "compressed", "cpu"))


@dataclass(frozen=True, order=True)
class TensorKey:
    layer: int
    expert: int
    projection: str

    def __post_init__(self) -> None:
        if self.layer < 0 or self.expert < 0:
            raise ValueError("layer and expert indices must be nonnegative")
        if self.projection not in ("gate_up", "down"):
            raise ValueError("projection must be 'gate_up' or 'down'")


@dataclass(frozen=True)
class TensorSpec:
    key: TensorKey
    raw_bytes: int
    compressed_bytes: int

    def __post_init__(self) -> None:
        if self.raw_bytes <= 0 or self.compressed_bytes < 0:
            raise ValueError("raw_bytes must be positive and compressed_bytes nonnegative")


@dataclass(frozen=True)
class BackendSnapshot:
    version: int
    backends: Mapping[TensorKey, str]


@dataclass(frozen=True)
class ReadToken:
    sequence: int
    version: int
    key: TensorKey
    generation: int


@dataclass(frozen=True)
class RetiredCopy:
    key: TensorKey
    backend: str
    generation: int
    retired_at_version: int


class VersionedBackends:
    """Publish immutable iteration metadata and defer retired-copy reclamation.

    Register every read before publication.  ``publish`` means all old-iteration
    prefetches have been issued, so beginning additional reads on an old snapshot
    is an error.  Reads already registered may finish after publication.  Storage
    generations distinguish an old compressed copy from a newly restored copy.
    The simulator owns a single-threaded event loop; this object is not a threaded
    atomic-publication implementation.
    """

    def __init__(self, backends: Mapping[TensorKey, str]) -> None:
        values = dict(backends)
        if any(value not in BACKENDS for value in values.values()):
            raise ValueError("unknown backend")
        self._snapshot = BackendSnapshot(0, MappingProxyType(values))
        self._pending: dict[TensorKey, str] | None = None
        self._generation = {key: 0 for key in values}
        self._next_generation = 1
        self._next_reader = 0
        self._readers: dict[int, ReadToken] = {}
        self._retired: list[RetiredCopy] = []
        self._ready: list[RetiredCopy] = []

    def snapshot(self) -> BackendSnapshot:
        return self._snapshot

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def pending_backends(self) -> Mapping[TensorKey, str] | None:
        return None if self._pending is None else MappingProxyType(self._pending)

    @property
    def retired_pending(self) -> tuple[RetiredCopy, ...]:
        return tuple(self._retired)

    def stage(self, changes: Mapping[TensorKey, str]) -> None:
        if self.has_pending:
            raise RuntimeError("publish the pending metadata before staging another update")
        if not changes:
            return
        next_values = dict(self._snapshot.backends)
        for key, backend in changes.items():
            if key not in next_values:
                raise KeyError(key)
            if backend not in BACKENDS:
                raise ValueError(f"unknown backend: {backend}")
            next_values[key] = backend
        if next_values != dict(self._snapshot.backends):
            self._pending = next_values

    def publish(self) -> BackendSnapshot:
        if self._pending is None:
            return self._snapshot
        version = self._snapshot.version + 1
        for key, backend in self._pending.items():
            previous = self._snapshot.backends[key]
            if backend != previous:
                self._retired.append(
                    RetiredCopy(key, previous, self._generation[key], version)
                )
                self._generation[key] = self._next_generation
                self._next_generation += 1
        self._snapshot = BackendSnapshot(version, MappingProxyType(self._pending))
        self._pending = None
        self._collect()
        return self._snapshot

    def begin_read(self, snapshot: BackendSnapshot, key: TensorKey) -> ReadToken:
        if snapshot is not self._snapshot:
            raise RuntimeError("cannot issue new reads using a retired or foreign snapshot")
        if key not in snapshot.backends:
            raise KeyError(key)
        token = ReadToken(self._next_reader, snapshot.version, key, self._generation[key])
        self._next_reader += 1
        self._readers[token.sequence] = token
        return token

    def complete_read(self, token: ReadToken) -> None:
        if self._readers.get(token.sequence) is not token:
            raise ValueError("unknown or already completed read token")
        del self._readers[token.sequence]
        self._collect()

    def _collect(self) -> None:
        active = {(reader.key, reader.generation) for reader in self._readers.values()}
        waiting = []
        for retired in self._retired:
            if (retired.key, retired.generation) in active:
                waiting.append(retired)
            else:
                self._ready.append(retired)
        self._retired = waiting

    def drain_reclaimable(self) -> tuple[RetiredCopy, ...]:
        """Return copies that can now be freed, exactly once per generation."""
        ready = tuple(self._ready)
        self._ready.clear()
        return ready


@dataclass(frozen=True)
class _Reclamation:
    before: Mapping[TensorKey, str]


class ResidencyState:
    """Rank-local expert backing state with whole-layer resident invariants."""

    def __init__(
        self,
        specs: Iterable[TensorSpec],
        resident_layers: Iterable[int] = (),
        cpu_keys: Iterable[TensorKey] = (),
        cpu_budget_bytes: int | None = None,
    ) -> None:
        items = tuple(specs)
        values = {spec.key: spec for spec in items}
        if len(values) != len(items) or not values:
            raise ValueError("specs must contain unique keys and must not be empty")
        self.specs: Mapping[TensorKey, TensorSpec] = MappingProxyType(values)
        self.layers = tuple(sorted({key.layer for key in values}))
        residents, cpu = set(resident_layers), set(cpu_keys)
        if not residents.issubset(self.layers) or not cpu.issubset(values):
            raise ValueError("initial placement references unknown layers or tensors")
        if any(key.layer in residents for key in cpu):
            raise ValueError("resident layers cannot also contain CPU-backed tensors")
        if cpu_budget_bytes is not None and cpu_budget_bytes < 0:
            raise ValueError("cpu_budget_bytes must be nonnegative")
        self.cpu_budget_bytes = cpu_budget_bytes
        initial = {
            key: "resident" if key.layer in residents else "cpu" if key in cpu else "compressed"
            for key in values
        }
        self._validate(initial)
        self.metadata = VersionedBackends(initial)
        self._history: list[_Reclamation] = []

    def snapshot(self) -> BackendSnapshot:
        return self.metadata.snapshot()

    @property
    def backends(self) -> Mapping[TensorKey, str]:
        return self.snapshot().backends

    @property
    def resident_layers(self) -> frozenset[int]:
        return frozenset(key.layer for key, backend in self.backends.items() if backend == "resident")

    @property
    def cpu_bytes(self) -> int:
        return self.cpu_bytes_for(self.backends)

    def cpu_bytes_for(self, backends: Mapping[TensorKey, str]) -> int:
        return sum(self.specs[key].raw_bytes for key, value in backends.items() if value == "cpu")

    def _validate(self, backends: Mapping[TensorKey, str]) -> None:
        for layer in self.layers:
            placements = {value for key, value in backends.items() if key.layer == layer}
            if "resident" in placements and placements != {"resident"}:
                raise ValueError("uncompressed residency must cover the complete layer")
        if self.cpu_budget_bytes is not None and self.cpu_bytes_for(backends) > self.cpu_budget_bytes:
            raise ValueError("CPU expert budget exceeded")

    def stage(self, changes: Mapping[TensorKey, str], *, between_launches: bool = False) -> None:
        next_values = dict(self.backends)
        for key, backend in changes.items():
            if key not in next_values:
                raise KeyError(key)
            if backend not in BACKENDS:
                raise ValueError(f"unknown backend: {backend}")
            if backend != next_values[key] and "resident" in (backend, next_values[key]):
                if not between_launches:
                    raise RuntimeError("layer residency remapping is allowed only between engine launches")
            next_values[key] = backend
        self._validate(next_values)
        self.metadata.stage(changes)

    def publish_iteration(self) -> BackendSnapshot:
        """Publish after old-iteration prefetch issuance, before new prefetches."""
        return self.metadata.publish()


@dataclass(frozen=True)
class PlannerDecision:
    action: str
    keys: tuple[TensorKey, ...]
    changes: Mapping[TensorKey, str]
    rho: float | None
    reason: str
    requires_relaunch: bool
    source_version: int


class ResidencyPlanner:
    """Paper Algorithm 2 with explicit publication and engine-launch boundaries."""

    def __init__(self, control_interval: int = 16, adjustment_granularity: float = 0.25,
                 restore_threshold: float = 0.9) -> None:
        if control_interval < 1:
            raise ValueError("control_interval must be positive")
        if not math.isfinite(adjustment_granularity) or not 0 < adjustment_granularity <= 1:
            raise ValueError("adjustment_granularity must lie in (0, 1]")
        if not math.isfinite(restore_threshold) or not 0 <= restore_threshold <= 1:
            raise ValueError("restore_threshold must lie in [0, 1]")
        self.control_interval = control_interval
        self.adjustment_granularity = adjustment_granularity
        self.restore_threshold = restore_threshold

    @staticmethod
    def _decision(state: ResidencyState, action: str, reason: str,
                  rho: float | None = None, changes: Mapping[TensorKey, str] | None = None,
                  requires_relaunch: bool = False) -> PlannerDecision:
        chosen = dict(changes or {})
        return PlannerDecision(action, tuple(chosen), MappingProxyType(chosen), rho, reason,
                               requires_relaunch, state.snapshot().version)

    def observe(
        self, state: ResidencyState, *, iteration: int, active_batch: int,
        remaining_batch: int, forward_ms: float, load_ms: float | Iterable[float],
        phase: str = "decode", profiles_complete: bool = True,
        between_launches: bool = False,
    ) -> PlannerDecision:
        """Plan from a completed interval; this does not mutate the state.

        ``iteration`` is a one-based completed decoding iteration counter.  Layer
        decisions may be returned while inside a launch, but are marked deferred;
        ``apply`` refuses them until explicitly given ``between_launches=True``.
        """
        if iteration < 0 or active_batch < 0 or remaining_batch < active_batch:
            raise ValueError("invalid iteration or batch counts")
        if phase not in ("decode", "prefill"):
            raise ValueError("phase must be 'decode' or 'prefill'")
        skip = None
        if remaining_batch == 0:
            skip = "finished"
        elif phase != "decode":
            skip = "prefill_disabled"
        elif active_batch == 0:
            skip = "no_active_sequences"
        elif iteration == 0 or iteration % self.control_interval:
            skip = "not_control_interval"
        elif not profiles_complete:
            skip = "incomplete_profiles"
        elif state.metadata.has_pending:
            skip = "metadata_pending_publication"
        if skip:
            return self._decision(state, "none", skip)
        loads = [load_ms] if isinstance(load_ms, (int, float)) else list(load_ms)
        if not math.isfinite(forward_ms) or forward_ms < 0:
            raise ValueError("forward_ms must be finite and nonnegative")
        if any(not math.isfinite(value) or value < 0 for value in loads):
            raise ValueError("load times must be finite and nonnegative")
        total_load = sum(loads)
        rho = math.inf if total_load == 0 else (remaining_batch / active_batch) * forward_ms / total_load
        if rho > 1 and active_batch < remaining_batch:
            if state.resident_layers:
                layer = max(state.resident_layers)
                changes = {key: "compressed" for key in sorted(state.specs)
                           if key.layer == layer}
                reason = "reclaim_resident_layer" if between_launches else "deferred_until_engine_relaunch"
                return self._decision(state, "reclaim_layer", reason, rho, changes, True)
            changes = self._offload(state)
            return self._decision(state, "offload" if changes else "none",
                                  "offload_pageable_tensors" if changes else "no_offload_within_budget",
                                  rho, changes)
        if rho < self.restore_threshold:
            if not state._history:
                return self._decision(state, "none", "at_initial_residency_floor", rho)
            changes = state._history[-1].before
            relaunch = any("resident" in (state.backends[key], value) for key, value in changes.items())
            reason = "restore_last_reclamation"
            if relaunch and not between_launches:
                reason = "deferred_until_engine_relaunch"
            return self._decision(state, "restore", reason, rho, changes, relaunch)
        return self._decision(state, "none", "dead_zone_or_no_batch_pressure", rho)

    def _offload(self, state: ResidencyState) -> dict[TensorKey, str]:
        down = [key for key, backend in state.backends.items()
                if backend == "compressed" and key.projection == "down"]
        projection = "down" if down else "gate_up"
        available = down or [key for key, backend in state.backends.items()
                             if backend == "compressed" and key.projection == "gate_up"]
        if not available:
            return {}
        expert_counts = {layer: len({key.expert for key in state.specs if key.layer == layer})
                         for layer in state.layers}
        if len(set(expert_counts.values())) != 1:
            raise ValueError("planner granularity requires homogeneous per-layer expert counts")
        count = math.ceil(self.adjustment_granularity * next(iter(expert_counts.values()))
                          / (2 if projection == "gate_up" else 1))
        volume = {layer: sum(state.specs[key].raw_bytes for key, value in state.backends.items()
                            if key.layer == layer and value == "cpu") for layer in state.layers}
        total = sum(volume.values())
        selected: dict[TensorKey, str] = {}
        for _ in range(min(count, len(available))):
            fitting = [key for key in available if state.cpu_budget_bytes is None or
                       total + state.specs[key].raw_bytes <= state.cpu_budget_bytes]
            if not fitting:
                break
            key = min(fitting, key=lambda key: (volume[key.layer], -key.layer, -key.expert))
            selected[key] = "cpu"
            available.remove(key)
            total += state.specs[key].raw_bytes
            volume[key.layer] += state.specs[key].raw_bytes
        return selected

    def apply(self, state: ResidencyState, decision: PlannerDecision,
              *, between_launches: bool = False) -> None:
        """Stage an observed decision; the caller publishes at the next boundary."""
        if decision.source_version != state.snapshot().version:
            raise RuntimeError("decision was computed from a stale metadata version")
        if not decision.changes:
            return
        if state.metadata.has_pending:
            raise RuntimeError("another metadata update is awaiting publication")
        if decision.action not in ("reclaim_layer", "offload", "restore"):
            raise ValueError("unsupported planner action")
        if decision.action == "restore" and (
            not state._history or dict(state._history[-1].before) != dict(decision.changes)
        ):
            raise RuntimeError("restoration must reverse the latest reclamation")
        before = MappingProxyType({key: state.backends[key] for key in decision.changes})
        state.stage(decision.changes, between_launches=between_launches)
        if decision.action == "restore":
            state._history.pop()
        else:
            state._history.append(_Reclamation(before))


# Short alias useful to callers implementing Algorithm 2.
Planner = ResidencyPlanner


def initial_bandwidth_placement(
    specs: Iterable[TensorSpec], cpu_bandwidth: float, gpu_bandwidth: float,
    cpu_budget_bytes: int | None = None,
) -> tuple[TensorKey, ...]:
    """Whole-down-tensor realization of paper Eq. 7--8, with a host budget cap.

    Bandwidths must use the same units and represent effective materialization
    rates, not nominal GPU memory bandwidth.  The paper's recipe applies when
    CPU share is at most one third; larger shares require another placement rule.
    Budget clipping balances layers, ties favoring descending layer/expert IDs.
    """
    if (not math.isfinite(cpu_bandwidth) or not math.isfinite(gpu_bandwidth)
            or cpu_bandwidth < 0 or gpu_bandwidth <= 0):
        raise ValueError("bandwidths require finite CPU >= 0 and GPU > 0")
    if cpu_budget_bytes is not None and cpu_budget_bytes < 0:
        raise ValueError("cpu_budget_bytes must be nonnegative")
    share = cpu_bandwidth / (cpu_bandwidth + gpu_bandwidth)
    if share > 1 / 3:
        raise ValueError("down-only paper placement requires CPU share <= 1/3")
    items = tuple(specs)
    values = {spec.key: spec for spec in items}
    if len(values) != len(items):
        raise ValueError("duplicate tensor keys")
    layers = sorted({key.layer for key in values})
    targets = {}
    for layer in layers:
        downs = {key.expert for key in values if key.layer == layer and key.projection == "down"}
        ups = {key.expert for key in values if key.layer == layer and key.projection == "gate_up"}
        if downs != ups:
            raise ValueError("each expert requires both projection tensor types")
        for expert in downs:
            if values[TensorKey(layer, expert, "gate_up")].raw_bytes != 2 * values[TensorKey(layer, expert, "down")].raw_bytes:
                raise ValueError("Eq. 8 requires gate/up tensors twice the size of down tensors")
        targets[layer] = math.ceil(3 * len(downs) * share)
    selected: list[TensorKey] = []
    count = {layer: 0 for layer in layers}
    volume = {layer: 0 for layer in layers}
    total = 0
    available = {key for key in values if key.projection == "down"}
    while available:
        fitting = [key for key in available if count[key.layer] < targets[key.layer]
                   and (cpu_budget_bytes is None or total + values[key].raw_bytes <= cpu_budget_bytes)]
        if not fitting:
            break
        key = min(fitting, key=lambda key: (volume[key.layer], -key.layer, -key.expert))
        selected.append(key)
        count[key.layer] += 1
        volume[key.layer] += values[key].raw_bytes
        total += values[key].raw_bytes
        available.remove(key)
    return tuple(selected)
