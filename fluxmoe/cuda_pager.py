"""Real single-rank GPU paging: VMM aliases, direct CUDA decode and pinned H2D.

Setup validates/compresses/uploads immutable backing once. The steady-state path
only submits stream waits, decoder kernels, async copies and events. It never
copies weights back to CPU and never synchronizes a device or stream per layer.
The API is an inference-only, sequential-layer protocol, without CUDA graphs.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from .codec import compress_bf16
from .cuda_codec import CudaCompressedTensor
from .cuda_vmm import CudaPagedArena
from .planner import TensorKey


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CudaMoEPager requires a CUDA-enabled PyTorch environment") from exc
    return torch


class PagingProtocolError(RuntimeError):
    """A caller violated the layer/iteration or storage lifetime protocol."""


class CudaMoEPager:
    """Materialize homogeneous BF16 experts into two buffers per projection.

    Input shapes: w13[layer] = [experts, 2*intermediate, hidden],
    w2[layer] = [experts, hidden, intermediate]. Inputs must be CPU tensors.
    All rank-local experts are loaded; routing is not consulted by the pager.

    Calling order: begin_iteration, acquire(0), enqueue compute, release(0), ...,
    end_iteration. The consumer must enqueue *all* weight readers on the acquired
    stream before release. External side streams require their own join first.
    """

    def __init__(self, w13: list[Any], w2: list[Any], *, cpu_fraction: float = 0.125,
                 cpu_keys: Iterable[TensorKey] | None = None, device: int = 0,
                 prefetch: bool = True, max_trace_events: int = 10000):
        torch = _torch()
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("device must be a nonnegative integer")
        if not torch.cuda.is_available() or device >= torch.cuda.device_count():
            raise RuntimeError("requested CUDA device is unavailable; no CPU fallback")
        if not isinstance(prefetch, bool):
            raise ValueError("prefetch must be boolean")
        if isinstance(max_trace_events, bool) or not isinstance(max_trace_events, int) or max_trace_events < 0:
            raise ValueError("max_trace_events must be a nonnegative integer")
        if not math.isfinite(cpu_fraction) or not 0 <= cpu_fraction <= 1:
            raise ValueError("cpu_fraction must be in [0, 1]")
        if not w13 or len(w13) != len(w2):
            raise ValueError("matching nonempty w13/w2 layer lists are required")
        for tensor in [*w13, *w2]:
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16:
                raise ValueError("setup weights must be CPU torch.bfloat16 tensors")
            if tensor.ndim != 3 or not tensor.is_contiguous() or tensor.requires_grad:
                raise ValueError("weights must be contiguous, 3D and inference-only")
        e, twice_i, h = w13[0].shape
        if e < 1 or h < 1 or twice_i < 2 or twice_i % 2:
            raise ValueError("invalid fused gate/up shape")
        i = twice_i // 2
        if any(tuple(t.shape) != (e, twice_i, h) for t in w13) or any(tuple(t.shape) != (e, h, i) for t in w2):
            raise ValueError("all layers must have matching homogeneous expert shapes")
        self.device, self.layers, self.experts = device, len(w13), e
        self.hidden, self.intermediate = h, i
        self.prefetch_enabled = prefetch
        self.max_trace_events = max_trace_events
        self._torch = torch
        self._closed = False
        self._closing = False
        self._failed = False
        self._active = False
        self._iteration = 0
        self._next_layer = 0
        self._acquired: tuple[int, Any, Any] | None = None
        self._ready: dict[int, list[Any]] = {}
        self._slot_owner: list[int | None] = [None, None]
        self._last_read: list[Any | None] = [None, None]
        self._arenas: dict[str, CudaPagedArena] = {}
        self._views: dict[str, list[Any]] = {}
        self._sources: dict[TensorKey, CudaCompressedTensor] = {}
        self._host: dict[TensorKey, Any] = {}
        self._streams: dict[tuple[str, str], Any] = {}
        self._consumer_streams: dict[int, Any] = {}
        self._records: list[dict[str, Any]] = []
        self._counters = {"decode_launches": 0, "h2d_copies": 0, "raw_waits": 0,
                          "war_waits": 0, "iterations": 0, "trace_events_dropped": 0}
        self._raw_sizes = {"gate_up": w13[0][0].numel() * 2, "down": w2[0][0].numel() * 2}
        total_raw = self.layers * e * sum(self._raw_sizes.values())
        self.cpu_budget_bytes = math.floor(cpu_fraction * total_raw)
        keys = {TensorKey(layer, expert, projection)
                for layer in range(self.layers) for expert in range(e) for projection in ("gate_up", "down")}
        selected: set[TensorKey] = set()
        if cpu_keys is not None:
            selected = set(cpu_keys)
            if not selected <= keys:
                raise ValueError("unknown CPU expert tensor")
        else:
            # Whole tensors, down first; balanced by raw CPU bytes per layer.
            volume = [0] * self.layers
            remaining = self.cpu_budget_bytes
            for projection in ("down", "gate_up"):
                available = {key for key in keys if key.projection == projection}
                while available and remaining >= self._raw_sizes[projection]:
                    key = min(available, key=lambda k: (volume[k.layer], -k.layer, -k.expert))
                    selected.add(key)
                    available.remove(key)
                    remaining -= self._raw_sizes[projection]
                    volume[key.layer] += self._raw_sizes[projection]
        if sum(self._raw_sizes[key.projection] for key in selected) > self.cpu_budget_bytes:
            raise ValueError("CPU placement exceeds configured BF16 host budget")
        self.cpu_keys = frozenset(selected)
        try:
            with torch.cuda.device(device):
                for projection, weights in (("gate_up", w13), ("down", w2)):
                    arena = CudaPagedArena(self.layers, weights[0].numel() * 2, device)
                    self._arenas[projection] = arena
                    self._views[projection] = [arena.tensor(layer, tuple(weights[layer].shape), torch.bfloat16)
                                               for layer in range(self.layers)]
                    for backend in ("compressed", "cpu"):
                        self._streams[projection, backend] = torch.cuda.Stream(device=device)
                    for layer, weight in enumerate(weights):
                        for expert in range(e):
                            key = TensorKey(layer, expert, projection)
                            if key in selected:
                                self._host[key] = weight[expert].clone().pin_memory()
                            else:
                                raw = weight[expert].view(torch.uint8).numpy().tobytes()
                                self._sources[key] = CudaCompressedTensor(compress_bf16(raw), device=device)
                self._anchor = torch.cuda.Event(enable_timing=True)
                self._anchor.record(torch.cuda.current_stream(device))
                for stream in self._streams.values():
                    stream.wait_event(self._anchor)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _open(self) -> None:
        if self._closed or self._closing:
            raise PagingProtocolError("pager is closed or closing")
        if self._failed:
            raise PagingProtocolError("pager failed during submission; synchronize and close it")

    def _record(self, record: dict[str, Any]) -> None:
        if len(self._records) < self.max_trace_events:
            self._records.append(record)
        else:
            self._counters["trace_events_dropped"] += 1

    def _layer(self, layer: int) -> None:
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < self.layers:
            raise ValueError("layer index out of range")

    def _stream(self, stream: Any = None) -> Any:
        stream = self._torch.cuda.current_stream(self.device) if stream is None else stream
        if stream.device.index != self.device:
            raise ValueError("compute stream belongs to a different device")
        return stream

    def layer_tensors(self, layer: int) -> tuple[Any, Any]:
        self._open()
        self._layer(layer)
        return self._views["gate_up"][layer], self._views["down"][layer]

    def begin_iteration(self) -> None:
        self._open()
        if self._active:
            raise PagingProtocolError("previous iteration is still active")
        with self._torch.cuda.device(self.device):
            if self._torch.cuda.is_current_stream_capturing():
                raise PagingProtocolError("CUDA graph capture is not supported by this pager")
        self._iteration += 1
        self._active = True
        self._next_layer = 0
        self._acquired = None
        self._ready = {}
        self._slot_owner = [None, None]
        if self.prefetch_enabled:
            for layer in range(min(2, self.layers)):
                self.prefetch(layer)

    def prefetch(self, layer: int) -> None:
        self._open()
        self._layer(layer)
        if not self._active:
            raise PagingProtocolError("prefetch requires an active iteration")
        if layer in self._ready:
            return
        if layer < self._next_layer or layer > self._next_layer + 1:
            raise PagingProtocolError("prefetch exceeds the two-layer window")
        slot = layer % 2
        if self._slot_owner[slot] is not None:
            raise PagingProtocolError("cannot overwrite a prefetched or acquired layer before release")
        try:
            self._enqueue_prefetch(layer)
        except BaseException:
            self._failed = True
            raise

    def _enqueue_prefetch(self, layer: int) -> None:
        slot = layer % 2
        self._slot_owner[slot] = layer
        ready = []
        torch = self._torch
        with torch.cuda.device(self.device):
            for projection in ("gate_up", "down"):
                for backend in ("compressed", "cpu"):
                    keys = [TensorKey(layer, expert, projection) for expert in range(self.experts)
                            if (TensorKey(layer, expert, projection) in self.cpu_keys) == (backend == "cpu")]
                    if not keys:
                        continue
                    stream = self._streams[projection, backend]
                    previous = self._last_read[slot]
                    if previous is not None:
                        stream.wait_event(previous)
                        self._counters["war_waits"] += 1
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        for key in keys:
                            destination = self._views[projection][layer][key.expert]
                            if backend == "cpu":
                                destination.copy_(self._host[key], non_blocking=True)
                                self._counters["h2d_copies"] += 1
                            else:
                                self._sources[key].decode_into(destination.data_ptr(), stream.cuda_stream,
                                                               self._raw_sizes[projection])
                                self._counters["decode_launches"] += 1
                        end.record(stream)
                    ready.append(end)
                    self._record({"kind": "load", "backend": backend, "projection": projection,
                                          "layer": layer, "slot": slot, "iteration": self._iteration,
                                          "stream": int(stream.cuda_stream), "raw_bytes": len(keys) * self._raw_sizes[projection],
                                          "war_dependency": previous is not None,
                                          "start_event": start, "end_event": end})
        self._ready[layer] = ready

    def acquire(self, layer: int, compute_stream: Any = None) -> tuple[Any, Any]:
        self._open()
        self._layer(layer)
        if not self._active or self._acquired is not None or layer != self._next_layer:
            raise PagingProtocolError("acquire must follow sequential layers and the previous release")
        stream = self._stream(compute_stream)
        self._consumer_streams[int(stream.cuda_stream)] = stream
        if layer not in self._ready:
            self.prefetch(layer)
        for event in self._ready[layer]:
            stream.wait_event(event)
            self._counters["raw_waits"] += 1
        start = self._torch.cuda.Event(enable_timing=True)
        start.record(stream)
        self._acquired = (layer, stream, start)
        return self.layer_tensors(layer)

    def release(self, layer: int, compute_stream: Any = None) -> None:
        self._open()
        stream = self._stream(compute_stream)
        if self._acquired is None or self._acquired[0] != layer or self._acquired[1].cuda_stream != stream.cuda_stream:
            raise PagingProtocolError("release must match the acquired layer and compute stream")
        done = self._torch.cuda.Event(enable_timing=True)
        done.record(stream)
        self._record({"kind": "compute", "layer": layer, "iteration": self._iteration,
                              "stream": int(stream.cuda_stream), "slot": layer % 2,
                              "raw_dependencies": len(self._ready[layer]),
                              "start_event": self._acquired[2], "end_event": done})
        self._last_read[layer % 2] = done
        self._slot_owner[layer % 2] = None
        self._acquired = None
        self._next_layer += 1
        if self.prefetch_enabled and layer + 2 < self.layers:
            self.prefetch(layer + 2)

    def end_iteration(self) -> None:
        self._open()
        if not self._active or self._acquired is not None or self._next_layer != self.layers:
            raise PagingProtocolError("all layers must be acquired and released before ending an iteration")
        self._active = False
        self._counters["iterations"] += 1

    def synchronize(self) -> None:
        """Explicit validation/cleanup boundary, not used in the layer hot path."""
        if self._closed:
            return
        errors = self._fence_and_check()
        if errors:
            self._failed = True
            raise RuntimeError("GPU paging validation failed: " + "; ".join(str(error) for error in errors)) from errors[0]

    def _fence_and_check(self) -> list[Exception]:
        errors = []
        for stream in [*self._streams.values(), *self._consumer_streams.values()]:
            try:
                stream.synchronize()
            except Exception as error:
                errors.append(error)
        for source in self._sources.values():
            try:
                source.check()
            except Exception as error:
                errors.append(error)
        return errors

    def memory(self) -> dict[str, Any]:
        return {"vmm_physical_bytes": sum(arena.physical_bytes for arena in self._arenas.values()),
                "vmm_logical_weight_bytes": sum(arena.logical_bytes for arena in self._arenas.values()),
                "vmm_granularity_bytes": {name: arena.granularity for name, arena in self._arenas.items()},
                "compressed_device_source_bytes": sum(source.device_bytes for source in self._sources.values()),
                "pinned_host_expert_bytes": sum(value.numel() * value.element_size() for value in self._host.values()),
                "host_expert_budget_bytes": self.cpu_budget_bytes,
                "note": "excludes test oracles, activations, torch allocator pools and event/driver metadata"}

    def trace(self) -> list[dict[str, Any]]:
        """Read actual CUDA timestamps after explicit end-to-end synchronization."""
        self.synchronize()
        output = []
        for record in self._records:
            start, end = record["start_event"], record["end_event"]
            output.append({**{key: value for key, value in record.items() if not key.endswith("_event")},
                           "start_ms": self._anchor.elapsed_time(start),
                           "end_ms": self._anchor.elapsed_time(end),
                           "duration_ms": start.elapsed_time(end)})
        return output

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        errors = self._fence_and_check()
        self._views.clear()
        # Explicitly refuse freeing arena storage still held by a bound module.
        if any(arena.live_tensor_leases for arena in self._arenas.values()):
            raise PagingProtocolError("VMM tensors still have external owners; detach bound modules and drop views, then retry close")
        for source in self._sources.values():
            try:
                source.close()
            except Exception as error:
                errors.append(error)
        self._sources.clear()
        self._host.clear()
        for arena in self._arenas.values():
            try:
                arena.close()
            except Exception as error:
                errors.append(error)
        self._arenas.clear()
        self._closed = True
        if errors:
            raise RuntimeError("GPU paging cleanup completed with errors: " + "; ".join(str(error) for error in errors)) from errors[0]

    def __enter__(self) -> CudaMoEPager:
        self._open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
