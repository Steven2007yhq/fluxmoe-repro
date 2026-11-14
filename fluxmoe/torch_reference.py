"""Optional, independent PyTorch numerical reference for the synthetic MoE.

Weights retain their exact BF16 bit patterns.  Routing, expert matrix products,
and residual accumulation use float64 so comparisons isolate the BF16 storage
path from additional arithmetic rounding.  This is a correctness oracle, not a
BF16-kernel performance benchmark or a vLLM/model-quality reproduction.

PyTorch is imported only when a reference function is called.  Requesting CUDA
without a usable CUDA device is an error; no operation silently falls back.
"""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Any, Sequence

from .planner import TensorKey

if TYPE_CHECKING:
    from .model import ToyMoE


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The optional PyTorch reference requires an installed torch package") from exc
    return torch


def _device(torch: Any, device: str) -> Any:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports no usable CUDA device")
    return torch.device(device)


def bf16_bytes_to_tensor(raw: bytes, *, device: str = "cpu") -> Any:
    """Interpret little-endian BF16 bytes without using the project's codec.

    Explicit uint16 unpacking preserves all bit patterns, including signed zero,
    infinities and NaN payloads, independently of host byte order.
    """
    torch = _torch()
    target = _device(torch, device)
    if len(raw) % 2:
        raise ValueError("BF16 storage must contain an even number of bytes")
    words = struct.unpack(f"<{len(raw) // 2}H", raw)
    return torch.tensor(words, dtype=torch.uint16, device=target).view(torch.bfloat16)


def tensor_to_bf16_bytes(tensor: Any) -> bytes:
    """Export exact BF16 storage bits, without any floating-point conversion."""
    torch = _torch()
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
        raise ValueError("expected a torch.bfloat16 tensor")
    words = tensor.detach().contiguous().view(torch.uint16).reshape(-1).cpu().tolist()
    return struct.pack(f"<{len(words)}H", *words)


def torch_forward(model: "ToyMoE", inputs: Sequence[Sequence[float]],
                  device: str = "cpu") -> list[list[float]]:
    """Evaluate all MoE layers with independently implemented Torch operations.

    Stable descending argsort reproduces the reference's lowest-expert-index
    tie break.  Expert contributions accumulate in per-token selected-rank order.
    All expert tensors are decoded, even for experts receiving no tokens.
    """
    torch = _torch()
    target = _device(torch, device)
    config = model.config
    rows = [list(row) for row in inputs]
    if any(len(row) != config.hidden or any(not math.isfinite(value) for value in row)
           for row in rows):
        raise ValueError("invalid input shape or nonfinite input")
    if not rows:
        return []
    with torch.no_grad():
        value = torch.tensor(rows, dtype=torch.float64, device=target)
        for layer in range(config.layers):
            router = torch.tensor(model.routers[layer], dtype=torch.float64, device=target)
            if tuple(router.shape) != (config.experts, config.hidden):
                raise ValueError("router shape mismatch")
            gate_up = []
            down = []
            for expert in range(config.experts):
                gu = bf16_bytes_to_tensor(model.weights[TensorKey(layer, expert, "gate_up")], device=device)
                dw = bf16_bytes_to_tensor(model.weights[TensorKey(layer, expert, "down")], device=device)
                if gu.numel() != 2 * config.intermediate * config.hidden or dw.numel() != config.hidden * config.intermediate:
                    raise ValueError("expert tensor shape mismatch")
                gate_up.append(gu.to(torch.float64).reshape(2 * config.intermediate, config.hidden))
                down.append(dw.to(torch.float64).reshape(config.hidden, config.intermediate))
            all_gate_up = torch.stack(gate_up)
            all_down = torch.stack(down)
            scores = value @ router.T
            selected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :config.top_k]
            probabilities = torch.softmax(torch.gather(scores, 1, selected), dim=1)
            result = value.clone()
            for choice in range(config.top_k):
                experts = selected[:, choice]
                projected = torch.bmm(all_gate_up[experts], value.unsqueeze(-1)).squeeze(-1)
                gate, up = projected.split(config.intermediate, dim=1)
                intermediate = torch.nn.functional.silu(gate) * up
                contribution = torch.bmm(all_down[experts], intermediate.unsqueeze(-1)).squeeze(-1)
                result = result + probabilities[:, choice].unsqueeze(-1) * contribution
            value = result
        return value.cpu().tolist()


def verify_torch_reference(model: "ToyMoE", batch: int = 8, device: str = "cpu",
                           tolerance: float = 1e-10) -> dict[str, Any]:
    """Compare deterministic synthetic inputs against Python's float64 oracle.

    The absolute tolerance covers matrix-reduction ordering differences; it is
    specific to this small bounded-weight fixture, not a model-quality threshold.
    CUDA's final host transfer synchronizes results before error calculation.
    """
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 0:
        raise ValueError("batch must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    inputs = model.inputs(batch)
    reference = model.forward(inputs)
    observed = torch_forward(model, inputs, device=device)
    errors = [abs(actual - expected) for actual_row, expected_row in zip(observed, reference)
              for actual, expected in zip(actual_row, expected_row)]
    max_abs = max(errors, default=0.0)
    finite = all(math.isfinite(value) for row in observed for value in row)
    shape_matches = len(observed) == len(reference) and all(
        len(actual) == len(expected) for actual, expected in zip(observed, reference)
    )
    return {
        "passed": finite and shape_matches and max_abs <= tolerance,
        "max_abs_error": max_abs,
        "max_abs": max_abs,
        "dtype": "torch.float64",
        "weight_dtype": "torch.bfloat16",
        "device": device,
        "tolerance": tolerance,
        "batch": batch,
        "layers": model.config.layers,
        "reference": "independent PyTorch matrix operations vs Python float arithmetic",
        "scope": "synthetic MoE correctness only",
    }
