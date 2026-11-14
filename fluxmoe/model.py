"""A small, executable MoE oracle; BF16 weights and Python float accumulation.

This is a numerical correctness fixture, not a replacement for a pretrained LLM.
Routing and expert arithmetic are identical for resident and paged execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
from typing import Callable

from .codec import bf16_to_floats, floats_to_bf16
from .planner import TensorKey

TensorReader = Callable[[int, int, str], bytes]


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 6
    experts: int = 4
    hidden: int = 16
    intermediate: int = 32
    top_k: int = 2
    seed: int = 17

    def __post_init__(self) -> None:
        for name in ("layers", "experts", "hidden", "intermediate", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k > self.experts:
            raise ValueError("top_k cannot exceed experts")


def _dot(row: list[float], vector: list[float]) -> float:
    return sum(a * b for a, b in zip(row, vector))


def _silu(x: float) -> float:
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    z = math.exp(x)
    return x * z / (1.0 + z)


class ToyMoE:
    """Fused gate/up, SwiGLU, down projection, top-k gating, residual.

    Each invocation materializes *all* local experts. No future routing is used
    for prefetch. The host oracle intentionally keeps original weights for
    comparison; these are not part of the simulated device memory footprint.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        rng = random.Random(config.seed)
        self.weights: dict[TensorKey, bytes] = {}
        self.routers: list[list[list[float]]] = []
        h, m = config.hidden, config.intermediate
        for layer in range(config.layers):
            self.routers.append([
                bf16_to_floats(floats_to_bf16(rng.gauss(0, 1 / math.sqrt(h)) for _ in range(h)))
                for _ in range(config.experts)
            ])
            for expert in range(config.experts):
                for projection, count, scale in (
                    ("gate_up", 2 * m * h, 0.3 / math.sqrt(h)),
                    ("down", h * m, 0.3 / math.sqrt(m)),
                ):
                    self.weights[TensorKey(layer, expert, projection)] = floats_to_bf16(
                        rng.gauss(0, scale) for _ in range(count)
                    )

    def reader(self, layer: int, expert: int, projection: str) -> bytes:
        return self.weights[TensorKey(layer, expert, projection)]

    def inputs(self, batch: int, iteration: int = 0) -> list[list[float]]:
        rng = random.Random(self.config.seed + 1009 * (iteration + 1))
        return [[rng.uniform(-1, 1) for _ in range(self.config.hidden)] for _ in range(batch)]

    def layer_sizes(self) -> dict[int, dict[str, list[int]]]:
        return {
            layer: {
                projection: [len(self.reader(layer, expert, projection)) for expert in range(self.config.experts)]
                for projection in ("gate_up", "down")
            }
            for layer in range(self.config.layers)
        }

    def forward_layer(
        self, layer: int, inputs: list[list[float]], reader: TensorReader,
    ) -> tuple[list[list[float]], list[int]]:
        c = self.config
        if not 0 <= layer < c.layers:
            raise ValueError("invalid layer")
        if any(len(x) != c.hidden or any(not math.isfinite(v) for v in x) for x in inputs):
            raise ValueError("invalid input shape or nonfinite input")
        # Decode all expert tensors, including experts that receive zero tokens.
        matrices = []
        for expert in range(c.experts):
            gu = bf16_to_floats(reader(layer, expert, "gate_up"))
            down = bf16_to_floats(reader(layer, expert, "down"))
            if len(gu) != 2 * c.intermediate * c.hidden or len(down) != c.hidden * c.intermediate:
                raise ValueError("expert tensor shape mismatch")
            matrices.append((
                [gu[j * c.hidden:(j + 1) * c.hidden] for j in range(2 * c.intermediate)],
                [down[j * c.intermediate:(j + 1) * c.intermediate] for j in range(c.hidden)],
            ))
        counts = [0] * c.experts
        outputs = []
        for x in inputs:
            scores = [_dot(row, x) for row in self.routers[layer]]
            selected = sorted(range(c.experts), key=lambda e: (-scores[e], e))[:c.top_k]
            peak = max(scores[e] for e in selected)
            probabilities = [math.exp(scores[e] - peak) for e in selected]
            total = sum(probabilities)
            out = list(x)
            for e, probability in zip(selected, probabilities):
                counts[e] += 1
                gu, down = matrices[e]
                intermediate = [
                    _silu(_dot(gu[j], x)) * _dot(gu[j + c.intermediate], x)
                    for j in range(c.intermediate)
                ]
                weight = probability / total
                for j, row in enumerate(down):
                    out[j] += weight * _dot(row, intermediate)
            outputs.append(out)
        return outputs, counts

    def forward(self, inputs: list[list[float]], reader: TensorReader | None = None) -> list[list[float]]:
        value = inputs
        for layer in range(self.config.layers):
            value, _ = self.forward_layer(layer, value, reader or self.reader)
        return value

    def fingerprint(self) -> str:
        digest = hashlib.sha256(repr(asdict(self.config)).encode())
        for key in sorted(self.weights):
            digest.update(repr(key).encode())
            digest.update(self.weights[key])
        return digest.hexdigest()
