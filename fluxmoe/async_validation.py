"""Bounded real-GPU MoE paging validation, without a pretrained checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .cli import provenance, write_json
from .cuda_pager import CudaMoEPager


def moe_layer(value: Any, router: Any, w13: Any, w2: Any, top_k: int = 2) -> Any:
    """GPU top-k/SwiGLU/residual fixture; BF16 weights, FP32 arithmetic.

    This is intentionally the same computation in resident and paged runs;
    actual vLLM Triton kernel equivalence is checked by the separate adapter.
    """
    import torch
    scores = value @ router.T
    selected = torch.argsort(scores, descending=True, stable=True, dim=1)[:, :top_k]
    probabilities = torch.softmax(torch.gather(scores, 1, selected), dim=1)
    fused, down = w13.float(), w2.float()
    result = value.clone()
    for position in range(top_k):
        expert = selected[:, position]
        projected = torch.bmm(fused[expert], value.unsqueeze(-1)).squeeze(-1)
        gate, up = projected.chunk(2, dim=1)
        activation = torch.nn.functional.silu(gate) * up
        contribution = torch.bmm(down[expert], activation.unsqueeze(-1)).squeeze(-1)
        result = result + probabilities[:, position:position + 1] * contribution
    return result


def run_validation(*, layers: int = 5, iterations: int = 5, batch: int = 8,
                   experts: int = 4, hidden: int = 64, intermediate: int = 96,
                   cpu_fraction: float = 0.25, prefetch: bool = True,
                   seed: int = 260402715) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this validation does not fall back to CPU")
    for value in (layers, iterations, batch, experts, hidden, intermediate):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("all model and execution dimensions must be positive integers")
    generator = torch.Generator().manual_seed(seed)
    w13 = [torch.randn(experts, 2 * intermediate, hidden, generator=generator).mul_(0.04).to(torch.bfloat16)
           for _ in range(layers)]
    w2 = [torch.randn(experts, hidden, intermediate, generator=generator).mul_(0.04).to(torch.bfloat16)
          for _ in range(layers)]
    routers = [torch.randn(experts, hidden, generator=generator).mul_(0.1).cuda() for _ in range(layers)]
    inputs = [torch.randn(batch, hidden, generator=generator).mul_(0.2).cuda() for _ in range(iterations)]
    resident13, resident2 = [value.cuda() for value in w13], [value.cuda() for value in w2]
    expected = []
    top_k = min(2, experts)
    with torch.no_grad():
        for value in inputs:
            for layer in range(layers):
                value = moe_layer(value, routers[layer], resident13[layer], resident2[layer], top_k)
            expected.append(value.cpu())
    del resident13, resident2
    pager = CudaMoEPager(w13, w2, cpu_fraction=cpu_fraction, prefetch=prefetch)
    observed, materialized = [], []
    pointers = [tuple(tensor.data_ptr() for tensor in pager.layer_tensors(layer)) for layer in range(layers)]
    compute_stream = torch.cuda.Stream()
    compute_stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.no_grad(), torch.cuda.stream(compute_stream):
            for iteration, value in enumerate(inputs):
                pager.begin_iteration()
                for layer in range(layers):
                    first, second = pager.acquire(layer, compute_stream)
                    # Keep independent device snapshots to inspect only after all
                    # iterations have been submitted; this is verification work.
                    materialized.append((iteration, layer, first.clone(), second.clone()))
                    value = moe_layer(value, routers[layer], first, second, top_k)
                    pager.release(layer, compute_stream)
                observed.append(value)
                pager.end_iteration()
        del first, second
        pager.synchronize()
        outputs = []
        for iteration, (value, reference) in enumerate(zip(observed, expected)):
            actual = value.cpu()
            error = (actual - reference).abs().max().item()
            equal = torch.equal(actual, reference)
            outputs.append({"iteration": iteration, "bitwise_equal": equal, "max_abs_error": error,
                            "output_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
            if not equal:
                raise AssertionError(f"paged MoE differs from resident in iteration {iteration}: {error}")
        for iteration, layer, first, second in materialized:
            if not torch.equal(first.cpu().view(torch.uint16), w13[layer].view(torch.uint16)):
                raise AssertionError(f"gate/up BF16 bits differ: iteration {iteration}, layer {layer}")
            if not torch.equal(second.cpu().view(torch.uint16), w2[layer].view(torch.uint16)):
                raise AssertionError(f"down BF16 bits differ: iteration {iteration}, layer {layer}")
        del first, second
        final_pointers = [tuple(tensor.data_ptr() for tensor in pager.layer_tensors(layer)) for layer in range(layers)]
        if final_pointers != pointers:
            raise AssertionError("VMM addresses changed")
        trace = pager.trace()
        overlap = any(
            max(load["start_ms"], compute["start_ms"]) < min(load["end_ms"], compute["end_ms"])
            for load in trace if load["kind"] == "load"
            for compute in trace if compute["kind"] == "compute"
        )
        result = {
            "status": "passed", "kind": "real_async_gpu_paging_correctness",
            "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "config": {"layers": layers, "iterations": iterations, "batch": batch,
                       "experts": experts, "hidden": hidden, "intermediate": intermediate,
                       "cpu_fraction": cpu_fraction, "prefetch": prefetch, "seed": seed},
            "comparisons": outputs, "weight_bitwise_checks": len(materialized) * 2,
            "stable_vmm_pointers": True, "pointers": pointers, "memory": pager.memory(),
            "counters": pager.counters, "trace": trace,
            "observed_load_compute_interval_overlap": overlap,
            "timing_scope": "Real CUDA events from a correctness run; includes weight-snapshot verification kernels and host issue gaps; not a performance benchmark.",
            "limitations": [
                "Single GPU, fixed expert backing placement, eager execution; no CUDA Graphs or TP/EP.",
                "CPU source encoding/indexing and persistent GPU-source upload happen once at setup.",
                "BF16 weight storage with FP32 fixture arithmetic; real vLLM kernel validation is a separate entry point.",
                "Host originals and GPU output/weight snapshots are verification fixtures outside the pager allocation ledger.",
                "VMM page rounding is measured; tiny layers do not demonstrate memory savings.",
                "Event interval overlap does not establish an end-to-end speedup.",
            ],
            "provenance": provenance(),
        }
    finally:
        # Error paths must not leave local aliases preventing safe arena teardown.
        first = second = None
        pager.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real asynchronous GPU weight paging correctness validation")
    parser.add_argument("--output", default="results/async_paging")
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--cpu-fraction", type=float, default=0.25)
    parser.add_argument("--no-prefetch", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = run_validation(layers=args.layers, iterations=args.iterations, batch=args.batch,
                                cpu_fraction=args.cpu_fraction, prefetch=not args.no_prefetch)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        write_json(output / "result.json", {"status": "failed", "error": str(exc), "provenance": provenance()})
        return 1
    write_json(output / "result.json", result)
    write_json(output / "trace.json", {"displayTimeUnit": "ms", "traceEvents": [
        {"ph": "X", "name": f"{event['kind']}:layer{event['layer']}", "cat": event["kind"],
         "pid": 1, "tid": str(event["stream"]), "ts": event["start_ms"] * 1000,
         "dur": event["duration_ms"] * 1000,
         "args": {key: value for key, value in event.items() if key not in ("start_ms", "end_ms", "duration_ms")}}
        for event in result["trace"]]})
    print(json.dumps({key: result[key] for key in ("status", "device", "config", "comparisons", "counters",
                                                   "weight_bitwise_checks", "stable_vmm_pointers")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
