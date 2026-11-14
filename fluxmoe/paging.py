"""CPU byte-buffer model of fixed virtual identities and layer-buffer aliasing.

The virtual addresses below are stable integer identifiers, never actual CUDA
addresses. Each pageable projection uses two real bytearray buffers. All aliases
are established at construction; ordinary layer execution never remaps them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .events import Event


class HazardError(RuntimeError):
    """An alias was read before materialization or overwritten before last use."""


@dataclass(frozen=True)
class TensorHandle:
    layer: int
    expert: int
    projection: str
    virtual_address: int
    size_bytes: int
    offset: int
    block_id: int


@dataclass(eq=False)
class PhysicalBlock:
    id: int
    projection: str
    data: bytearray
    owners: list[tuple[int, int, str] | None]
    last_read: Event | None = None

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class PagedTensorPool:
    """Represent homogeneous MoE layers with two physical blocks per projection.

    ``layer_sizes[layer][projection][expert]`` is a tensor byte length. Layers
    must have matching expert counts and sizes, with gate_up and down tensors.
    Resident layers each get independent physical blocks. Nonresident layers,
    sorted by layer index, alias alternating physical buffers. Tensor identities
    have distinct page-aligned virtual identifiers regardless of physical alias.
    """

    PROJECTIONS = ("gate_up", "down")

    def __init__(
        self,
        layer_sizes: dict[int, dict[str, list[int]]],
        resident_layers: set[int] | None = None,
    ) -> None:
        if not layer_sizes:
            raise ValueError("at least one layer is required")
        if any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 for layer in layer_sizes):
            raise ValueError("layer indices must be nonnegative integers")
        self.resident_layers = frozenset(resident_layers or ())
        if not self.resident_layers.issubset(layer_sizes):
            raise ValueError("resident_layers contains an unknown layer")
        self.layer_sizes: dict[int, dict[str, tuple[int, ...]]] = {}
        reference = None
        for layer in sorted(layer_sizes):
            sizes = layer_sizes[layer]
            if set(sizes) != set(self.PROJECTIONS):
                raise ValueError("each layer must contain exactly gate_up and down")
            normalized = {projection: tuple(sizes[projection]) for projection in self.PROJECTIONS}
            if not normalized["gate_up"] or len(normalized["gate_up"]) != len(normalized["down"]):
                raise ValueError("projections must have matching nonzero expert counts")
            if any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for values in normalized.values() for size in values
            ):
                raise ValueError("tensor byte lengths must be positive integers")
            if reference is not None and normalized != reference:
                raise ValueError("all layers must have homogeneous tensor sizes")
            reference = normalized
            self.layer_sizes[layer] = normalized
        self._blocks: list[PhysicalBlock] = []
        self._mapping: dict[tuple[int, str], PhysicalBlock] = {}
        self._handles: dict[tuple[int, int, str], TensorHandle] = {}
        pageable = [layer for layer in self.layer_sizes if layer not in self.resident_layers]
        for projection in self.PROJECTIONS:
            sizes = self.layer_sizes[next(iter(self.layer_sizes))][projection]
            if pageable:
                shared = [self._new_block(projection, sizes) for _ in range(2)]
                for ordinal, layer in enumerate(pageable):
                    self._mapping[layer, projection] = shared[ordinal % 2]
            for layer in sorted(self.resident_layers):
                self._mapping[layer, projection] = self._new_block(projection, sizes)
        next_address = 0x100000000
        for layer, sizes in self.layer_sizes.items():
            for projection in self.PROJECTIONS:
                offset = 0
                for expert, size in enumerate(sizes[projection]):
                    self._handles[layer, expert, projection] = TensorHandle(
                        layer=layer, expert=expert, projection=projection,
                        virtual_address=next_address, size_bytes=size, offset=offset,
                        block_id=self._mapping[layer, projection].id,
                    )
                    next_address += ((size + 4095) // 4096) * 4096
                    offset += size

    def _new_block(self, projection: str, sizes: tuple[int, ...]) -> PhysicalBlock:
        block = PhysicalBlock(
            id=len(self._blocks), projection=projection,
            data=bytearray(sum(sizes)), owners=[None] * len(sizes),
        )
        self._blocks.append(block)
        return block

    @property
    def physical_bytes(self) -> int:
        return sum(block.size_bytes for block in self._blocks)

    @property
    def logical_bytes(self) -> int:
        return sum(handle.size_bytes for handle in self._handles.values())

    def tensor(self, layer: int, expert: int, projection: str) -> TensorHandle:
        try:
            return self._handles[layer, expert, projection]
        except KeyError:
            raise KeyError(f"unknown tensor: layer={layer}, expert={expert}, projection={projection}") from None

    def block(self, layer: int, projection: str) -> PhysicalBlock:
        try:
            return self._mapping[layer, projection]
        except KeyError:
            raise KeyError(f"unknown layer/projection: {layer}/{projection}") from None

    def write(self, layer: int, expert: int, projection: str, data: bytes) -> None:
        """Materialize bytes at write-event completion, after checking WAR at start.

        The caller must check ``assert_write_safe(..., write_event.start_ms)``.
        This method cannot infer the start of a write from its completion callback.
        """
        handle = self.tensor(layer, expert, projection)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("tensor contents must be bytes-like")
        contents = bytes(data)
        if len(contents) != handle.size_bytes:
            raise ValueError(f"expected {handle.size_bytes} bytes, got {len(contents)}")
        block = self.block(layer, projection)
        block.data[handle.offset : handle.offset + handle.size_bytes] = contents
        block.owners[expert] = (layer, expert, projection)

    def read(self, layer: int, expert: int, projection: str) -> bytes:
        handle = self.tensor(layer, expert, projection)
        block = self.block(layer, projection)
        if block.owners[expert] != (layer, expert, projection):
            raise HazardError(
                f"RAW/alias hazard: tensor {layer}/{expert}/{projection} is not materialized; "
                f"physical owner is {block.owners[expert]}"
            )
        return bytes(block.data[handle.offset : handle.offset + handle.size_bytes])

    def assert_write_safe(self, layer: int, projection: str, at_ms: float) -> None:
        at = float(at_ms)
        if not math.isfinite(at) or at < 0:
            raise ValueError("at_ms must be finite and nonnegative")
        block = self.block(layer, projection)
        if block.last_read is not None and block.last_read.end_ms > at:
            raise HazardError(
                f"WAR hazard: block {block.id} is read until {block.last_read.end_ms} ms, "
                f"but write starts at {at} ms"
            )

    def mark_read(self, layer: int, projection: str, event: Event) -> None:
        if not isinstance(event, Event) or event._timeline is None:
            raise ValueError("last_read must be a scheduled Event")
        block = self.block(layer, projection)
        if block.last_read is not None and block.last_read._timeline is not event._timeline:
            raise ValueError("a physical block cannot mix events from different timelines")
        if block.last_read is None or event.end_ms >= block.last_read.end_ms:
            block.last_read = event

    def summary(self) -> dict:
        return {
            "backend": "cpu_bytearray_model",
            "virtual_addresses": "stable identifiers, not CUDA virtual addresses",
            "layers": len(self.layer_sizes),
            "resident_layers": sorted(self.resident_layers),
            "physical_blocks": len(self._blocks),
            "physical_bytes": self.physical_bytes,
            "logical_bytes": self.logical_bytes,
            "tensors": len(self._handles),
            "mapping": [
                {"layer": layer, "projection": projection, "block_id": block.id}
                for (layer, projection), block in sorted(self._mapping.items())
            ],
        }
