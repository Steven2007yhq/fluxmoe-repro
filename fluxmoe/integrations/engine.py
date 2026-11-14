"""Checkpoint-to-generation vLLM 0.29 integration for eager single-GPU Mixtral.

Set ``worker_extension_cls`` to :class:`FluxMoEWorkerExtension` and use
``load_format='fluxmoe'`` (paged) or ``'fluxmoe_resident'`` (resident oracle).
Both loaders delegate the real checkpoint loading and kernel postprocessing to
vLLM. Paging is attached before the worker profiles/allocates its KV cache.

Bootstrap currently loads the complete model on CUDA first. This validates the
engine path on a fitting model; it does not enable loading an oversized model.
There are no changes to the installed vLLM package, graph capture, or CPU
fallbacks. The CPU-only configuration and hook tests are not GPU evidence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import math
import sys
from typing import Any

from .vllm import (
    SUPPORTED_VLLM_VERSION, VllmPagedAdapter, extract_cpu_weights,
    inspect_runner, installed_vllm_version,
)


@dataclass(frozen=True)
class EnginePagingOptions:
    cpu_fraction: float = 0.125
    prefetch: bool = True
    max_trace_events: int = 10000
    checkpoint_format: str = "auto"

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "EnginePagingOptions":
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown FluxMoE loader options: {sorted(unknown)}")
        result = cls(**values)
        if isinstance(result.cpu_fraction, bool) or not isinstance(result.cpu_fraction, (int, float)) or not math.isfinite(result.cpu_fraction) or not 0 <= result.cpu_fraction <= 1:
            raise ValueError("cpu_fraction must be a finite number in [0, 1]")
        if not isinstance(result.prefetch, bool):
            raise ValueError("prefetch must be boolean")
        if isinstance(result.max_trace_events, bool) or not isinstance(result.max_trace_events, int) or result.max_trace_events < 0:
            raise ValueError("max_trace_events must be a nonnegative integer")
        if result.checkpoint_format not in {"auto", "safetensors", "pt"}:
            raise ValueError("checkpoint_format must be auto, safetensors, or pt")
        return result


def validate_engine_config(config: Any, model_config: Any) -> None:
    """Fail before loading or replacing weights for unsupported engine modes."""
    if not bool(getattr(model_config, "enforce_eager", False)):
        raise ValueError("FluxMoE engine requires enforce_eager=True")
    if getattr(model_config, "quantization", None) is not None:
        raise ValueError("quantized engine models are unsupported")
    if str(getattr(model_config, "dtype", "torch.bfloat16")) not in {"torch.bfloat16", "bfloat16"}:
        raise ValueError("FluxMoE engine requires BF16 model weights")
    if bool(getattr(model_config, "enable_sleep_mode", False)):
        raise ValueError("sleep mode does not manage FluxMoE VMM allocations")
    architectures = getattr(getattr(model_config, "hf_config", None), "architectures", ())
    if architectures != ["MixtralForCausalLM"]:
        raise ValueError("engine integration currently supports only MixtralForCausalLM")
    parallel = config.parallel_config
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"single-GPU engine requires {name}=1")
    for name in ("enable_expert_parallel", "enable_eplb", "enable_dbo"):
        if bool(getattr(parallel, name, False)):
            raise ValueError(f"{name} is unsupported")
    if getattr(parallel, "ubatch_size", 0) not in (0, 1):
        raise ValueError("microbatch overlap is unsupported")
    compilation = config.compilation_config
    for name in ("mode", "cudagraph_mode"):
        value = getattr(compilation, name, None)
        if getattr(value, "value", value) != 0:
            raise ValueError(f"FluxMoE requires compilation_config.{name}=NONE")
    for name in ("lora_config", "speculative_config", "kv_transfer_config", "weight_transfer_config"):
        if getattr(config, name, None) is not None:
            raise ValueError(f"{name} is unsupported")
    if getattr(config.kernel_config, "moe_backend", None) != "triton":
        raise ValueError("FluxMoE requires kernel_config.moe_backend='triton'")
    scheduler = getattr(config, "scheduler_config", None)
    if bool(getattr(scheduler, "async_scheduling", False)):
        raise ValueError("this engine integration requires async_scheduling=False")
    cache = getattr(config, "cache_config", None)
    if getattr(cache, "cache_dtype", "auto") not in {"auto", "bfloat16"}:
        raise ValueError("quantized or non-BF16 KV caches are outside this engine integration")
    if getattr(cache, "kv_offloading_size", None) is not None:
        raise ValueError("KV offloading is unsupported")
    offload = getattr(config, "offload_config", None)
    if (getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0) != 0 or
            getattr(getattr(offload, "prefetch", None), "offload_group_size", 0) != 0):
        raise ValueError("native vLLM weight offloading cannot be combined with FluxMoE")
    device = getattr(getattr(config, "device_config", None), "device", "cuda")
    load_device = getattr(getattr(config, "load_config", None), "device", None)
    if str(load_device or device).split(":")[0] != "cuda":
        raise ValueError("FluxMoE engine requires CUDA checkpoint loading")


def _expert_runners(model: Any) -> list[Any]:
    # modules() preserves the actual decoder order for the explicitly supported
    # Mixtral architecture. A shared/repeated MoERunner would violate the protocol.
    result = [module for module in model.modules()
              if type(module).__name__ == "MoERunner" and hasattr(module, "routed_experts")]
    if not result:
        raise ValueError("loaded Mixtral contains no vLLM 0.29 MoERunners")
    for runner in result:
        inspect_runner(runner)
    return result


def _tensor_sha256(tensor: Any) -> str:
    import torch
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def _weight_manifest(runners: list[Any], weights: tuple[list[Any], list[Any]]) -> list[dict[str, Any]]:
    return [{"layer": index, "name": getattr(runner, "layer_name", ""),
             "w13_shape": list(w13.shape), "w2_shape": list(w2.shape),
             "w13_sha256": _tensor_sha256(w13), "w2_sha256": _tensor_sha256(w2)}
            for index, (runner, w13, w2) in enumerate(zip(runners, *weights))]


def _forward_observation(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    value = kwargs.get("input_ids")
    if value is None and args:
        value = args[0]
    if value is None:
        value = kwargs.get("inputs_embeds")
    row: dict[str, Any] = {"tokens": int(value.shape[0]) if value is not None else None,
                           "phase": "profile_or_unclassified", "max_query_len": None}
    try:
        from vllm.forward_context import get_forward_context
        metadata = get_forward_context().attn_metadata
    except (ImportError, AssertionError):
        return row
    if isinstance(metadata, dict) and metadata:
        first = next(iter(metadata.values()))
        maximum = getattr(first, "max_query_len", None)
        row["max_query_len"] = int(maximum) if maximum is not None else None
        row["phase"] = ("prefill_or_mixed" if maximum is not None and maximum > 1 else
                        "single_token_query" if maximum == 1 else "attention_metadata_present")
        # A one-token prompt also has max_query_len=1. Do not claim that every
        # such call is decoding without inspecting the sequence lengths.
    return row


class FluxMoEModelController:
    """Own root-forward hooks and the optional pager for one loaded model."""

    def __init__(self, model: Any, *, adapter: Any = None, pager: Any = None,
                 loading_report: dict[str, Any] | None = None, max_observations: int = 2048):
        if (adapter is None) != (pager is None):
            raise ValueError("adapter and pager must be supplied together")
        self.model, self.adapter, self.pager = model, adapter, pager
        self.loading_report = dict(loading_report or {})
        self.max_observations = max_observations
        self.observations: list[dict[str, Any]] = []
        self.forward_counts: Counter[str] = Counter()
        self._iteration: Any = None
        self._executing = False
        self.closed = False
        self._handles: list[Any] = []
        try:
            self._handles.append(model.register_forward_pre_hook(self._before, with_kwargs=True))
            self._handles.append(model.register_forward_hook(self._after, with_kwargs=True, always_call=True))
        except BaseException:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            raise

    def _before(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self.closed or self._executing:
            raise RuntimeError("FluxMoE root model is closed or already executing")
        self._executing = True
        try:
            if self.adapter is not None:
                context = self.adapter.iteration()
                context.__enter__()
                self._iteration = context
            row = _forward_observation(args, kwargs)
            self.forward_counts["started"] += 1
            self.forward_counts[row["phase"]] += 1
            if len(self.observations) < self.max_observations:
                self.observations.append(row)
        except BaseException:
            self._executing = False
            if self._iteration is not None:
                context, self._iteration = self._iteration, None
                context.__exit__(*sys.exc_info())
            raise

    def _after(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        if not self._executing:
            return  # always_call also executes when a prehook failed.
        exception = sys.exc_info()
        try:
            if self._iteration is not None:
                context, self._iteration = self._iteration, None
                context.__exit__(*exception)
            self.forward_counts["failed" if exception[0] is not None else "completed"] += 1
        except BaseException:
            self.forward_counts["failed"] += 1
            raise
        finally:
            self._executing = False

    def status(self, *, include_trace: bool = False) -> dict[str, Any]:
        result = {"mode": "paged" if self.pager is not None else "resident", "closed": self.closed,
                  "loading": self.loading_report, "forward_counts": dict(self.forward_counts),
                  "forward_observations": list(self.observations)}
        if self.pager is not None and not self.closed:
            result.update(pager_memory=self.pager.memory(), pager_counters=self.pager.counters,
                          adapter=self.adapter.summary())
            if include_trace:
                result["pager_trace"] = self.pager.trace()
        return result

    def close(self) -> None:
        """Terminal teardown: restore expert parameters to CPU, then free VMM."""
        if self.closed:
            return
        if self._executing:
            raise RuntimeError("cannot close an executing model")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self.adapter is not None:
            self.adapter.detach(restore_to_device=False)
            self.pager.close()
        self.closed = True


def _kv_status(worker: Any, *, hash_kv: bool = False) -> dict[str, Any]:
    runner = worker.model_runner
    rows = []
    for index, tensor in enumerate(getattr(runner, "kv_caches", [])):
        row = {"index": index, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
               "device": str(tensor.device), "data_ptr": tensor.data_ptr(),
               "bytes": tensor.numel() * tensor.element_size()}
        if hash_kv:
            row["sha256"] = _tensor_sha256(tensor)
        rows.append(row)
    config = getattr(runner, "kv_cache_config", None)
    cache = getattr(worker, "cache_config", None)
    groups = [group for collection in getattr(runner, "attn_groups", []) for group in collection]
    return {"tensors": rows, "tensor_bytes": sum(row["bytes"] for row in rows),
            "num_blocks": getattr(config, "num_blocks", None),
            "num_gpu_blocks": getattr(cache, "num_gpu_blocks", None),
            "block_size": getattr(cache, "block_size", None),
            "attention_backends": sorted({group.backend.__name__ for group in groups}),
            "hash_note": "Hashes include unused/uninitialized slots; only before/after changes within the same worker indicate writes. Do not compare whole-cache hashes across processes." if hash_kv else None}


class FluxMoEWorkerExtension:
    """Mixin loaded by vLLM before model loading; callable using collective_rpc."""

    def fluxmoe_status(self, include_trace: bool = False, hash_kv: bool = False) -> dict[str, Any]:
        model = self.model_runner.get_model()
        controller = getattr(model, "_fluxmoe_controller", None)
        if controller is None:
            # Default resident loader users may start observing before generate.
            controller = FluxMoEModelController(model, loading_report={"observer_attached_after_load": True})
            model._fluxmoe_controller = controller
        result = controller.status(include_trace=include_trace)
        result["kv_cache"] = _kv_status(self, hash_kv=hash_kv)
        return result

    def fluxmoe_close(self) -> dict[str, Any]:
        controller = getattr(self.model_runner.get_model(), "_fluxmoe_controller", None)
        if controller is not None:
            controller.close()
        return {"closed": True, "inference_can_resume": False}


_REGISTERED_LOADERS: tuple[type, type] | None = None


def register_engine_loaders() -> tuple[type, type]:
    """Register custom formats in this process, without patching vLLM files."""
    global _REGISTERED_LOADERS
    if _REGISTERED_LOADERS is not None:
        return _REGISTERED_LOADERS
    version = installed_vllm_version()
    if version.split("+")[0] != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"engine integration inspected for vLLM {SUPPORTED_VLLM_VERSION}; found {version}")
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    class EngineModelLoader(BaseModelLoader):
        paged = True

        def __init__(self, load_config: Any):
            super().__init__(load_config)
            self.options = EnginePagingOptions.from_dict(load_config.model_loader_extra_config)
            delegate_config = replace(load_config, load_format=self.options.checkpoint_format,
                                      model_loader_extra_config={})
            self.delegate = DefaultModelLoader(delegate_config)

        def download_model(self, model_config: Any) -> None:
            self.delegate.download_model(model_config)

        def load_weights(self, model: Any, model_config: Any) -> None:
            if hasattr(model, "_fluxmoe_controller"):
                raise RuntimeError("in-place weight reload after FluxMoE binding is unsupported")
            self.delegate.load_weights(model, model_config)

        def load_model(self, vllm_config: Any, model_config: Any, prefix: str = "") -> Any:
            validate_engine_config(vllm_config, model_config)
            model = self.delegate.load_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
            runners = _expert_runners(model)
            weights = extract_cpu_weights(runners)
            report = {"checkpoint_format": self.options.checkpoint_format,
                      "vllm_version": version, "checkpoint_loaded_by": "DefaultModelLoader",
                      "attached_after_kernel_postprocess": True, "attached_before_kv_initialization": True,
                      "bootstrap_requires_resident_model": True,
                      "expert_weights": _weight_manifest(runners, weights)}
            pager = adapter = None
            try:
                if self.paged:
                    import torch
                    from ..cuda_pager import CudaMoEPager
                    pager = CudaMoEPager(*weights, cpu_fraction=self.options.cpu_fraction,
                                        prefetch=self.options.prefetch,
                                        max_trace_events=self.options.max_trace_events,
                                        device=torch.cuda.current_device())
                    adapter = VllmPagedAdapter(runners, pager, original_cpu_weights=weights)
                    # Detached resident CUDA storages can otherwise remain in
                    # the caching allocator during vLLM's subsequent KV budget.
                    torch.cuda.empty_cache()
                model._fluxmoe_controller = FluxMoEModelController(
                    model, adapter=adapter, pager=pager, loading_report=report)
                return model
            except BaseException as original:
                # Preserve the load/bind error even if an asynchronous CUDA
                # error or a surviving tensor lease also prevents teardown.
                for resource, cleanup in ((adapter, lambda: adapter.detach(restore_to_device=False)),
                                          (pager, lambda: pager.close())):
                    if resource is not None:
                        try:
                            cleanup()
                        except BaseException as cleanup_error:
                            message = f"FluxMoE setup cleanup also failed: {cleanup_error!r}"
                            if hasattr(original, "add_note"):
                                original.add_note(message)
                            else:  # Python 3.10 remains supported by this package.
                                import warnings
                                warnings.warn(message, RuntimeWarning)
                raise

    class ResidentModelLoader(EngineModelLoader):
        paged = False

    register_model_loader("fluxmoe")(EngineModelLoader)
    register_model_loader("fluxmoe_resident")(ResidentModelLoader)
    _REGISTERED_LOADERS = EngineModelLoader, ResidentModelLoader
    return _REGISTERED_LOADERS


# Importing the worker extension in a spawned worker installs the same registry
# entries there. Plain Python and torch-only environments remain importable.
if importlib.util.find_spec("vllm") is not None:
    register_engine_loaders()
