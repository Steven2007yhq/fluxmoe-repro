"""Synchronous, first-attention-layer KV diagnostics for the tiny vLLM fixture.

Install/status/remove are callables accepted by ``LLM.collective_rpc``. This is
deliberately diagnostic instrumentation, not a performance measurement path.
Only ordinary BF16 causal Triton attention with optional GQA is supported.
"""
from __future__ import annotations

import copy
from typing import Any


class KVObserver:
    def __init__(self, layer: Any, *, attention_atol: float = 1e-3,
                 attention_rtol: float = 1e-2) -> None:
        import torch
        impl = layer.impl
        if type(impl).__name__ != "TritonAttentionImpl":
            raise ValueError("KV observer requires TRITON_ATTN (TritonAttentionImpl)")
        if layer.kv_cache.dtype != torch.bfloat16 or layer.kv_cache.ndim != 4:
            raise ValueError("KV observer requires a BF16 Triton cache with shape [B,H,N,2D]")
        if (impl.alibi_slopes is not None or impl.sliding_window != (-1, -1)
                or impl.logits_soft_cap != 0 or getattr(impl, "sinks", None) is not None
                or getattr(layer, "query_quant", None) is not None):
            raise ValueError("KV observer supports ordinary full causal attention only")
        if layer.num_heads % layer.num_kv_heads or layer.head_size_v != layer.head_size:
            raise ValueError("KV observer requires compatible GQA heads and equal K/V head sizes")
        self.layer = layer
        self.pending: dict[str, Any] | None = None
        self.written_slots: set[int] = set()
        self.data: dict[str, Any] = {
            "layer_name": layer.layer_name,
            "scope": "first attention layer; every written token and every single-token query with history",
            "diagnostic_synchronization": True,
            "prefill_forwards": 0, "decode_forwards": 0, "chunked_prefill_forwards": 0,
            "cache_write_checks": 0, "cache_write_failures": 0,
            "history_read_checks": 0, "history_read_failures": 0,
            "history_missing_slots": 0, "rewritten_slots": 0,
            "attention_atol": attention_atol, "attention_rtol": attention_rtol,
            "max_attention_abs_error": 0.0, "steps": [],
        }
        self.handles = [layer.register_forward_pre_hook(self.before, with_kwargs=True),
                        layer.register_forward_hook(self.after, with_kwargs=True)]

    def before(self, layer: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        metadata = context.attn_metadata
        if isinstance(metadata, dict):
            metadata = metadata[layer.layer_name]
        elif isinstance(metadata, list):
            raise ValueError("KV observer does not support microbatch metadata")
        if metadata is None:
            self.pending = None
            return  # vLLM profiling/warmup has no real cache operations.
        if not metadata.causal:
            raise ValueError("KV observer requires causal attention")
        tensors = [args[index] if len(args) > index else kwargs[name]
                   for index, name in enumerate(("query", "key", "value"))]
        query, key, value = tensors
        slots = context.slot_mapping[layer.layer_name][:metadata.num_actual_tokens].cpu().tolist()
        query_starts = metadata.query_start_loc.cpu().tolist()
        lengths = metadata.seq_lens.cpu().tolist()
        if len(query_starts) != len(lengths) + 1:
            raise ValueError("attention sequence/query metadata lengths do not agree")
        self.pending = {
            "query": query.detach().reshape(-1, layer.num_heads, layer.head_size).cpu(),
            "key": key.detach().reshape(-1, layer.num_kv_heads, layer.head_size).cpu(),
            "value": value.detach().reshape(-1, layer.num_kv_heads, layer.head_size).cpu(),
            "slots": slots, "query_starts": query_starts, "seq_lens": lengths,
            "block_table": metadata.block_table.cpu().tolist(),
        }

    def after(self, layer: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        import torch
        import torch.nn.functional as functional
        current, self.pending = self.pending, None
        if current is None:
            return
        cache = layer.kv_cache
        _, _, block_size, _ = cache.shape
        starts, lengths = current["query_starts"], current["seq_lens"]
        query_lengths = [end - start for start, end in zip(starts, starts[1:])]
        prefill = any(length > 1 for length in query_lengths)
        decode = any(query == 1 and length > 1 for query, length in zip(query_lengths, lengths))
        self.data["prefill_forwards"] += int(prefill)
        self.data["decode_forwards"] += int(decode)
        self.data["chunked_prefill_forwards"] += int(any(
            1 < query < length for query, length in zip(query_lengths, lengths)))
        valid = [(index, slot) for index, slot in enumerate(current["slots"]) if slot >= 0]
        valid_indices = [index for index, _ in valid]
        slot_ids = [slot for _, slot in valid]
        # Read only assigned slots; never hash or compare uninitialized cache.
        if valid:
            blocks = torch.tensor([slot // block_size for slot in slot_ids], device=cache.device)
            offsets = torch.tensor([slot % block_size for slot in slot_ids], device=cache.device)
            actual = cache[blocks, :, offsets, :].cpu()
            expected = torch.cat((current["key"][valid_indices], current["value"][valid_indices]), dim=-1)
            if expected.dtype != torch.bfloat16:
                raise ValueError("KV observer requires BF16 input keys and values")
            matched = (actual.view(torch.uint16) == expected.view(torch.uint16)).flatten(1).all(dim=1)
            self.data["cache_write_checks"] += len(valid)
            self.data["cache_write_failures"] += int((~matched).sum().item())
            self.data["rewritten_slots"] += sum(slot in self.written_slots for slot in slot_ids)

        actual_output = output.detach().reshape(-1, layer.num_heads, layer.head_size).float().cpu()
        step_errors: list[float] = []
        for request_index, (query_len, seq_len) in enumerate(zip(query_lengths, lengths)):
            if query_len != 1 or seq_len <= 1:
                continue
            block_ids = current["block_table"][request_index]
            historical_slots = [block_ids[position // block_size] * block_size + position % block_size
                                for position in range(seq_len)]
            missing = sum(slot not in self.written_slots for slot in historical_slots[:-1])
            self.data["history_missing_slots"] += missing
            blocks = torch.tensor([slot // block_size for slot in historical_slots], device=cache.device)
            offsets = torch.tensor([slot % block_size for slot in historical_slots], device=cache.device)
            history = cache[blocks, :, offsets, :].float().cpu()
            k, v = history.split(layer.head_size, dim=-1)
            repeats = layer.num_heads // layer.num_kv_heads
            k = k.transpose(0, 1).repeat_interleave(repeats, dim=0).unsqueeze(0)
            v = v.transpose(0, 1).repeat_interleave(repeats, dim=0).unsqueeze(0)
            position = starts[request_index]
            q = current["query"][position].float()[None, :, None, :]
            # This query is the final position; it can attend to the entire
            # historical sequence. is_causal=True would mask it as position 0.
            expected_output = functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False, scale=layer.impl.scale)[0, :, 0, :]
            actual = actual_output[position]
            difference = (actual - expected_output).abs()
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected_output).all())
            error = float(difference.max().item()) if finite else float("inf")
            matched = finite and bool(torch.allclose(actual, expected_output,
                atol=self.data["attention_atol"], rtol=self.data["attention_rtol"]))
            self.data["history_read_checks"] += 1
            self.data["history_read_failures"] += int(not matched or missing > 0)
            self.data["max_attention_abs_error"] = max(self.data["max_attention_abs_error"], error)
            step_errors.append(error)

        self.written_slots.update(slot_ids)
        self.data["steps"].append({
            "query_lens": query_lengths, "seq_lens": lengths,
            "written_slots": slot_ids,
            "active_block_ids": [row[:(length + block_size - 1) // block_size]
                                 for row, length in zip(current["block_table"], lengths)],
            "history_attention_abs_errors": step_errors,
        })

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install_kv_observer(worker: Any) -> dict[str, Any]:
    """Install after LLM construction and before the first test request."""
    if getattr(worker, "_fluxmoe_kv_observer", None) is not None:
        raise RuntimeError("KV observer is already installed")
    from vllm.model_executor.layers.attention import Attention
    model = worker.model_runner.get_model()
    layer = next((module for module in model.modules() if isinstance(module, Attention)), None)
    if layer is None:
        raise RuntimeError("loaded model has no vLLM Attention layer")
    worker._fluxmoe_kv_observer = KVObserver(layer)
    return kv_observer_status(worker)


def kv_observer_status(worker: Any) -> dict[str, Any]:
    observer = getattr(worker, "_fluxmoe_kv_observer", None)
    if observer is None:
        raise RuntimeError("KV observer is not installed")
    return copy.deepcopy(observer.data)


def remove_kv_observer(worker: Any) -> dict[str, Any]:
    result = kv_observer_status(worker)
    worker._fluxmoe_kv_observer.remove()
    del worker._fluxmoe_kv_observer
    return result
