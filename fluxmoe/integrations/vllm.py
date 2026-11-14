"""Eager, single-rank vLLM 0.29 unquantized Triton MoE paging adapter.

Attach after ``process_weights_after_loading`` and before graph capture. This
adapter changes live parameter storage and wraps the actual MoERunner forward;
it does not implement an LLM engine, checkpoint loader, or multi-rank serving.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
from typing import Any, Iterable


SUPPORTED_VLLM_VERSION = "0.29.0"


def installed_vllm_version() -> str:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("vLLM is not installed in this interpreter; use the existing vllm_env") from exc


def _torch() -> Any:
    import torch
    return torch


def validate_weight_layout(w13: Any, w2: Any, *, require_cuda: bool = False) -> tuple[int, int, int]:
    """Validate the unshuffled BF16 gate/up and down layout used by Triton."""
    torch = _torch()
    if not isinstance(w13, torch.Tensor) or not isinstance(w2, torch.Tensor):
        raise TypeError("expert weights must be torch tensors")
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("only unquantized BF16 expert weights are supported")
    if w13.ndim != 3 or w2.ndim != 3 or any(size <= 0 for size in (*w13.shape, *w2.shape)):
        raise ValueError("expert weights must have positive rank-3 shapes")
    experts, fused, hidden = w13.shape
    if fused % 2 or tuple(w2.shape) != (experts, hidden, fused // 2):
        raise ValueError("expected w13[E,2I,H] and w2[E,H,I]")
    if not w13.is_contiguous() or not w2.is_contiguous():
        raise ValueError("only contiguous unshuffled Triton weights are supported")
    if w13.device != w2.device:
        raise ValueError("expert weight devices must match")
    if require_cuda and w13.device.type != "cuda":
        raise ValueError("bound expert views must be CUDA tensors")
    return experts, hidden, fused // 2


def inspect_runner(runner: Any) -> dict[str, Any]:
    """Reject unsupported configurations before replacing any parameter storage."""
    routed = getattr(runner, "routed_experts", None)
    if routed is None:
        raise ValueError("vLLM 0.29 MoERunner.routed_experts is required")
    method = getattr(routed, "quant_method", None)
    if method is None or type(method).__name__ != "UnquantizedFusedMoEMethod":
        raise ValueError("only UnquantizedFusedMoEMethod is supported")
    if getattr(routed, "quant_config", None) is not None:
        raise ValueError("quantized MoE is unsupported")
    backend = getattr(getattr(method, "unquantized_backend", None), "name", None)
    if backend != "TRITON":
        raise ValueError(f"requires unshuffled TRITON backend, found {backend!r}")
    if getattr(method, "moe_kernel", None) is None:
        raise ValueError("call process_weights_after_loading before installing the pager")
    moe = getattr(runner, "moe_config", None)
    parallel = getattr(moe, "moe_parallel_config", None)
    if moe is None or parallel is None:
        raise ValueError("runner has no MoE parallel configuration")
    for field in ("tp_size", "dp_size", "ep_size", "pcp_size", "sp_size"):
        if getattr(parallel, field, 1) != 1:
            raise ValueError(f"single-rank adapter requires {field}=1")
    for field in ("enable_eplb", "use_ep", "is_sequence_parallel", "use_all2all_kernels"):
        if bool(getattr(parallel, field, False)):
            raise ValueError(f"{field} is unsupported")
    if bool(getattr(runner, "enable_dbo", False)):
        raise ValueError("dual batch overlap is unsupported")
    if getattr(runner, "_shared_experts", None) is not None:
        raise ValueError("shared-expert overlap is unsupported")
    if getattr(runner, "routed_input_transform", None) is not None or getattr(runner, "routed_output_transform", None) is not None:
        raise ValueError("latent expert transforms are unsupported")
    if bool(getattr(moe, "is_lora_enabled", False)) or bool(getattr(moe, "has_bias", False)):
        raise ValueError("LoRA and expert bias are unsupported")
    activation = getattr(getattr(routed, "activation", None), "value", None)
    if activation != "silu":
        raise ValueError("only SiLU/SwiGLU experts are supported")
    shape = validate_weight_layout(routed.w13_weight, routed.w2_weight)
    return {"runner_type": f"{type(runner).__module__}.{type(runner).__name__}",
            "layer_name": getattr(runner, "layer_name", ""), "backend": backend,
            "experts": shape[0], "hidden": shape[1], "intermediate": shape[2],
            "weight_dtype": "bfloat16", "tp": 1, "ep": 1}


def extract_cpu_weights(runners: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    """Snapshot postprocessed weights for pager construction; may synchronize D2H.

    This bootstrap starts from already-loaded modules. It does not avoid the
    checkpoint-loading GPU peak; an engine loader hook is separate work.
    """
    w13, w2 = [], []
    for runner in runners:
        inspect_runner(runner)
        routed = runner.routed_experts
        w13.append(routed.w13_weight.detach().to("cpu").contiguous().clone())
        w2.append(routed.w2_weight.detach().to("cpu").contiguous().clone())
    if not w13:
        raise ValueError("at least one MoERunner is required")
    return w13, w2


@dataclass
class _Binding:
    runner: Any
    original_cpu_w13: Any
    original_cpu_w2: Any
    original_device: Any
    pointer_w13: int
    pointer_w2: int
    shape: tuple[int, int, int]


class VllmPagedAdapter:
    """Bind a CudaMoEPager to a sequence of actual eager MoERunner modules.

    The pager is borrowed, never closed by this adapter. Detach the adapter and
    discard any external tensor views before closing the pager. Parameter
    objects/weight-loader attributes remain unchanged; only their data changes.
    Original backups are CPU tensors, not retained resident GPU allocations.
    """

    def __init__(self, runners: Iterable[Any], pager: Any,
                 original_cpu_weights: tuple[list[Any], list[Any]] | None = None):
        version = installed_vllm_version()
        if version.split("+")[0] != SUPPORTED_VLLM_VERSION:
            raise RuntimeError(f"adapter inspected for vLLM {SUPPORTED_VLLM_VERSION}; found {version}")
        self.runners = tuple(runners)
        if not self.runners:
            raise ValueError("at least one MoERunner is required")
        self.pager, self.version = pager, version
        self._bindings: list[_Binding] = []
        self._handles: list[Any] = []
        self._active = False
        self._failed = False
        self._attached = False
        self._next_layer = 0
        self._inflight: int | None = None
        self.layer_reports = [inspect_runner(runner) for runner in self.runners]
        if len({(row["experts"], row["hidden"], row["intermediate"]) for row in self.layer_reports}) != 1:
            raise ValueError("this pager adapter requires homogeneous expert layers")
        if original_cpu_weights is None:
            original_cpu_weights = extract_cpu_weights(self.runners)
        originals13, originals2 = original_cpu_weights
        if len(originals13) != len(self.runners) or len(originals2) != len(self.runners):
            raise ValueError("CPU backup count does not match runners")
        views = []
        torch = _torch()
        # Validate all inputs first, so unsupported layers cannot partially bind.
        for layer, runner in enumerate(self.runners):
            first, second = pager.layer_tensors(layer)
            shape = validate_weight_layout(first, second, require_cuda=True)
            routed = runner.routed_experts
            if first.device != routed.w13_weight.device or second.device != routed.w2_weight.device:
                raise ValueError("VMM views and loaded vLLM parameters must use the same CUDA device")
            if tuple(first.shape) != tuple(routed.w13_weight.shape) or tuple(second.shape) != tuple(routed.w2_weight.shape):
                raise ValueError("VMM view shapes do not match the vLLM expert parameters")
            validate_weight_layout(originals13[layer], originals2[layer])
            if originals13[layer].device.type != "cpu" or originals2[layer].device.type != "cpu":
                raise ValueError("restoration backups must reside on CPU")
            if tuple(originals13[layer].shape) != tuple(first.shape) or tuple(originals2[layer].shape) != tuple(second.shape):
                raise ValueError("CPU backups do not match the expert parameter shapes")
            if not isinstance(routed.w13_weight, torch.nn.Parameter) or not isinstance(routed.w2_weight, torch.nn.Parameter):
                raise TypeError("vLLM expert weights must be Parameters")
            views.append((first, second, shape))
        try:
            for layer, (runner, (first, second, shape)) in enumerate(zip(self.runners, views)):
                routed = runner.routed_experts
                record = _Binding(runner, originals13[layer], originals2[layer], routed.w13_weight.device,
                                  first.data_ptr(), second.data_ptr(), shape)
                self._bindings.append(record)
                routed.w13_weight.data = first
                routed.w2_weight.data = second
                self._handles.append(runner.register_forward_pre_hook(self._pre_hook(layer)))
                self._handles.append(runner.register_forward_hook(self._post_hook(layer), always_call=True))
            self._attached = True
        except BaseException:
            for handle in self._handles:
                handle.remove()
            for record in self._bindings:
                record.runner.routed_experts.w13_weight.data = record.original_cpu_w13.to(record.original_device)
                record.runner.routed_experts.w2_weight.data = record.original_cpu_w2.to(record.original_device)
            self._bindings.clear()
            raise

    def _pre_hook(self, layer: int):
        def hook(module: Any, args: Any) -> None:
            if not self._active or self._failed or layer != self._next_layer or self._inflight is not None:
                raise RuntimeError("MoE paging requires one ordered eager forward inside adapter.iteration()")
            torch = _torch()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("CUDA Graph capture is not supported by this eager adapter")
            record = self._bindings[layer]
            routed = module.routed_experts
            if routed.w13_weight.data_ptr() != record.pointer_w13 or routed.w2_weight.data_ptr() != record.pointer_w2:
                raise RuntimeError("vLLM replaced expert storage after binding; detach and reconstruct the pager")
            self.pager.acquire(layer, compute_stream=torch.cuda.current_stream(record.original_device))
            self._inflight = layer
        return hook

    def _post_hook(self, layer: int):
        def hook(module: Any, args: Any, output: Any) -> None:
            # always_call also fires after a failed prehook; no read was issued then.
            if self._inflight != layer:
                return
            torch = _torch()
            self.pager.release(layer, compute_stream=torch.cuda.current_stream(self._bindings[layer].original_device))
            self._inflight = None
            self._next_layer += 1
            if output is None:
                self._failed = True
        return hook

    @contextmanager
    def iteration(self):
        if not self._attached or self._active or self._failed:
            raise RuntimeError("adapter is detached, already executing, or failed")
        self._active = True
        self._next_layer = 0
        try:
            self.pager.begin_iteration()
            yield self
            if self._failed or self._next_layer != len(self.runners) or self._inflight is not None:
                raise RuntimeError("iteration did not execute every bound MoE layer exactly once")
            self.pager.end_iteration()
        except BaseException:
            self._failed = True
            self.pager.synchronize()
            raise
        finally:
            self._active = False

    def detach(self, *, restore_to_device: bool = False) -> None:
        """Remove hooks and restore backups, after queued readers finish.

        By default parameters move to CPU, releasing VMM references without
        allocating a resident model on GPU. Such runners cannot resume GPU
        inference until moved to CUDA. restore_to_device=True explicitly makes
        new resident CUDA copies and requires that additional memory budget.
        """
        if self._active:
            raise RuntimeError("cannot detach during an active iteration")
        if not self._attached:
            return
        self.pager.synchronize()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for record in self._bindings:
            destination = record.original_device if restore_to_device else "cpu"
            routed = record.runner.routed_experts
            routed.w13_weight.data = record.original_cpu_w13.to(destination)
            routed.w2_weight.data = record.original_cpu_w2.to(destination)
        self._bindings.clear()
        self._attached = False

    def summary(self) -> dict[str, Any]:
        return {"adapter": "vllm-0.29-eager-single-rank-unquantized-triton", "version": self.version,
                "attached": self._attached, "failed": self._failed, "layers": self.layer_reports,
                "original_gpu_storage_retained": False,
                "cpu_restore_backup_bytes": sum((b.original_cpu_w13.numel() + b.original_cpu_w2.numel()) * 2 for b in self._bindings),
                "scope": "MoERunner module integration; not an LLM engine or checkpoint loader"}


def run_vllm_smoke(*, layers: int = 4, iterations: int = 3, batch: int = 16,
                   experts: int = 4, hidden: int = 64, intermediate: int = 128,
                   cpu_fraction: float = 0.125, seed: int = 260402715) -> dict[str, Any]:
    """Run actual vLLM MoERunners on small random weights; no model downloads.

    This intentionally initializes/destroys a single-rank distributed context;
    invoke it in a dedicated process, never inside a running serving worker.
    Triton JIT is allowed. Returned timings are not serving-performance claims.
    """
    for name, number in (("layers", layers), ("iterations", iterations), ("batch", batch),
                         ("experts", experts), ("hidden", hidden), ("intermediate", intermediate)):
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError(f"{name} must be a positive integer")
    torch = _torch()
    if installed_vllm_version().split("+")[0] != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"real-runner smoke requires vLLM {SUPPORTED_VLLM_VERSION}")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required; no CPU fallback")
    if torch.distributed.is_initialized():
        raise RuntimeError("smoke requires an isolated process without an existing distributed group")
    import gc
    import hashlib
    import tempfile
    from pathlib import Path
    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
    from vllm.config.device import DeviceConfig
    from vllm.config.kernel import KernelConfig
    from vllm.config.scheduler import SchedulerConfig
    from vllm.distributed import (
        init_distributed_environment, ensure_model_parallel_initialized,
        destroy_model_parallel, destroy_distributed_environment,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
    from ..cuda_pager import CudaMoEPager

    config = VllmConfig(
        device_config=DeviceConfig(device="cuda"),
        scheduler_config=SchedulerConfig(max_model_len=256, is_encoder_decoder=False,
                                         max_num_batched_tokens=max(batch, 32), max_num_seqs=min(batch, 32)),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE),
        kernel_config=KernelConfig(moe_backend="triton"),
        optimization_level=0,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    original13 = [torch.randn(experts, 2 * intermediate, hidden, generator=generator).mul_(0.02).to(torch.bfloat16)
                  for _ in range(layers)]
    original2 = [torch.randn(experts, hidden, intermediate, generator=generator).mul_(0.02).to(torch.bfloat16)
                 for _ in range(layers)]
    router_cpu = [torch.randn(hidden, experts, generator=generator).mul_(0.1) for _ in range(layers)]
    inputs_cpu = [torch.randn(batch, hidden, generator=generator).mul_(0.2).to(torch.bfloat16)
                  for _ in range(iterations)]
    runners: list[Any] = []
    adapter = None
    pager = None
    with tempfile.TemporaryDirectory(prefix="fluxmoe-vllm-dist-") as temporary:
        try:
            with set_current_vllm_config(config):
                init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                             distributed_init_method=(Path(temporary) / "rendezvous").as_uri(),
                                             backend="nccl")
                ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
                init_workspace_manager(torch.device("cuda:0"))
                with torch.device("cuda"):
                    for layer in range(layers):
                        runner = FusedMoEFactory(
                            num_experts=experts, top_k=min(2, experts), hidden_size=hidden,
                            intermediate_size=intermediate, params_dtype=torch.bfloat16,
                            renormalize=True, quant_config=None, tp_size=1, dp_size=1, pcp_size=1,
                            prefix=f"fluxmoe_smoke.layers.{layer}.experts",
                        )
                        routed = runner.routed_experts
                        with torch.no_grad():
                            routed.w13_weight.copy_(original13[layer])
                            routed.w2_weight.copy_(original2[layer])
                        routed.quant_method.process_weights_after_loading(routed)
                        runners.append(runner)
                inputs = [value.cuda() for value in inputs_cpu]
                routers = [value.cuda() for value in router_cpu]
                initial_pointers = [(runner.routed_experts.w13_weight.data_ptr(), runner.routed_experts.w2_weight.data_ptr())
                                    for runner in runners]

                def forward(value):
                    with set_forward_context(None, config, num_tokens=batch), torch.no_grad():
                        for layer, runner in enumerate(runners):
                            logits = value.float() @ routers[layer]
                            value = value + runner(value, logits)
                    return value

                # This resident baseline also warms the actual Triton kernel.
                expected = [forward(value).clone() for value in inputs]
                torch.cuda.synchronize()
                source_weights = extract_cpu_weights(runners)
                pager = CudaMoEPager(*source_weights, cpu_fraction=cpu_fraction, device=0, prefetch=True)
                adapter = VllmPagedAdapter(runners, pager, original_cpu_weights=source_weights)
                binding_summary = adapter.summary()
                bound_pointers = [(runner.routed_experts.w13_weight.data_ptr(), runner.routed_experts.w2_weight.data_ptr())
                                  for runner in runners]
                if any(before == after for before, after in zip(initial_pointers, bound_pointers)):
                    raise AssertionError("binding did not change resident weight addresses to VMM addresses")
                observed = []
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for value in inputs:
                        with adapter.iteration():
                            observed.append(forward(value).clone())
                pager.synchronize()
                stream.synchronize()
                comparisons = []
                for iteration, (actual, reference) in enumerate(zip(observed, expected)):
                    equal = torch.equal(actual, reference)
                    error = (actual.float() - reference.float()).abs().max().item()
                    comparisons.append({"iteration": iteration, "bitwise_equal": equal, "max_abs_error": error})
                if not all(item["bitwise_equal"] for item in comparisons):
                    raise AssertionError(f"paged vLLM outputs differ from resident baseline: {comparisons}")
                final_pointers = [(runner.routed_experts.w13_weight.data_ptr(), runner.routed_experts.w2_weight.data_ptr())
                                  for runner in runners]
                if final_pointers != bound_pointers:
                    raise AssertionError("vLLM expert pointers changed during paged execution")
                trace = pager.trace()
                source_root = Path(vllm.__file__).parent
                source_paths = [source_root / relative for relative in (
                    "model_executor/layers/fused_moe/runner/moe_runner.py",
                    "model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
                    "model_executor/layers/fused_moe/experts/triton_moe.py",
                )]
                result = {
                    "status": "passed", "kind": "real_vllm_moerunner_paging_correctness",
                    "vllm_version": vllm.__version__, "torch_version": torch.__version__,
                    "torch_cuda_build": torch.version.cuda, "device": torch.cuda.get_device_name(),
                    "config": {"layers": layers, "iterations": iterations, "batch": batch, "experts": experts,
                               "hidden": hidden, "intermediate": intermediate, "cpu_fraction": cpu_fraction,
                               "seed": seed, "tp": 1, "ep": 1, "compile": "NONE", "graphs": "NONE"},
                    "binding": binding_summary, "comparisons": comparisons,
                    "stable_vmm_pointers": True, "vmm_pointers": bound_pointers,
                    "pager_trace": trace,
                    "vllm_source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                    "limitations": ["Synthetic BF16 MoE layers, no checkpoint/tokenizer/attention/serving scheduler.",
                                    "Single rank, eager-only Triton backend; no TP/EP/CUDA Graph/EPLB/quantization.",
                                    "Bootstrap begins from resident weights and keeps CPU restoration fixtures.",
                                    "Correctness run with JIT/warmup; no paper throughput or memory-peak claim."],
                }
                adapter.detach()
                adapter = None
                gc.collect()
                pager.close()
                pager = None
                return result
        finally:
            if adapter is not None:
                adapter.detach()
            if pager is not None:
                gc.collect()
                pager.close()
            runners.clear()
            reset_workspace_manager()
            destroy_model_parallel()
            destroy_distributed_environment()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Real vLLM eager MoERunner paging correctness smoke")
    parser.add_argument("--output", default="results/vllm_paging_smoke.json")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        result = run_vllm_smoke(layers=args.layers, iterations=args.iterations, batch=args.batch)
        code = 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = {"status": "failed", "kind": "real_vllm_moerunner_paging_correctness", "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "pager_trace"}, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
