"""Four-rank BF16 Mixtral capture and offline same-layout replay comparison.

Use identical diagnostic capture in two fresh engines. The comparison binds
ordered real InputBatch rows, padding, KV tensor layout, output checkpoints,
and the first forward that consumes each selected restored block. Physical
cache block IDs and internal request ID suffixes are local to each engine.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import re


VOCAB_SIZE = 32000
HIDDEN_SIZE = 4096
TP_SIZE = 2
PP_SIZE = 2
LAYERS_PER_STAGE = 16
RANKS = tuple(range(TP_SIZE * PP_SIZE))
MAX_EVIDENCE_PER_RANK = 320 * 2**20
STAGES = (
    "layer_input", "input_norm", "attention_output", "post_attention_norm",
    "moe_output", "layer_output", "attention_kernel",
)
STRICT_TOLERANCE = (1e-5, 1e-5)
ATTENTION_TOLERANCE = (0.001, 0.01)
RANDOM_SUFFIX = re.compile(r"-[a-fA-F0-9]{8}$")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, document):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _layout_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _request_table(requests):
    table = {row["request_id"]: tuple(row["prompt_token_ids"]) for row in requests}
    _require(len(table) == len(requests) > 0 and all(table.values()),
             "unique request IDs and nonempty prompts required")
    return table


def _external_id(internal_id, requests):
    if internal_id in requests:
        return internal_id
    match = RANDOM_SUFFIX.search(internal_id)
    _require(match is not None and internal_id[:match.start()] in requests,
             f"unbound vLLM request ID {internal_id!r}")
    return internal_id[:match.start()]


def _target_positions(requests, checkpoints, output_tokens, anchors):
    _require(tuple(checkpoints) == tuple(sorted(set(checkpoints)))
             and checkpoints and 1 <= checkpoints[0] <= checkpoints[-1] <= output_tokens,
             "invalid output ordinal checkpoints")
    targets = {(request, len(prompt) + ordinal - 2)
               for request, prompt in requests.items() for ordinal in checkpoints}
    restore = set()
    for anchor in anchors:
        request = anchor["request_id"]
        position = anchor["input_position"]
        _require(request in requests and type(position) is int
                 and len(requests[request]) <= position <= len(requests[request]) + output_tokens - 2,
                 f"restored history anchor must belong to an actual decode position: {anchor!r}")
        restore.add((request, position))
    _require(len(restore) == len(anchors)
             and len({request for request, _ in restore}) == len(restore),
             "one first restore anchor per request required")
    return targets, restore


def _identity(worker):
    import torch
    from vllm.distributed.parallel_state import (
        get_dp_group, get_ep_group, get_pp_group, get_tp_group,
    )

    rank = torch.distributed.get_rank()
    pp, tp, ep, dp = get_pp_group(), get_tp_group(), get_ep_group(), get_dp_group()
    _require(torch.distributed.get_world_size() == len(RANKS)
             and (pp.world_size, tp.world_size, ep.world_size, dp.world_size) == (2, 2, 2, 1)
             and rank == pp.rank_in_group * TP_SIZE + tp.rank_in_group,
             "TP2/PP2/EP2/DP1 rank layout differs")
    device = torch.cuda.current_device()
    uuid = str(torch.cuda.get_device_properties(device).uuid).lower().removeprefix("gpu-")
    _require(re.fullmatch(r"[a-f0-9-]{36}", uuid), "worker CUDA UUID unavailable")
    return dict(rank=rank, pp_rank=pp.rank_in_group, tp_rank=tp.rank_in_group,
                ep_rank=ep.rank_in_group, pid=os.getpid(), uuid="GPU-" + uuid,
                device_index=device)


def fence(worker):
    """Wait original PP send handles and the device, outside the forward loop."""
    import torch

    runner = worker.model_runner
    _require(runner.execute_model_state is None and runner.req_states.num_reqs == 0
             and all(slot is None for slot in runner.pp_handler.queue),
             "cannot fence unfinished requests or pending PP sampled-token state")
    handles = worker._pp_send_work
    for handle in handles:
        handle.wait()
    worker._pp_send_work = []
    torch.cuda.synchronize(worker.device)
    return dict(_identity(worker), pp_handles_waited=len(handles),
                pending_pp_handles=0, pending_pp_sampled_tokens=0,
                device_synchronized=True)


def validate_fences(fences, gpu_uuids, initial=None):
    """Require a real device-completion receipt from each expected live rank."""
    rows = sorted(fences, key=lambda row: row["rank"])
    _require(tuple(row["rank"] for row in rows) == RANKS
             and len(gpu_uuids) == len(RANKS) == len(set(gpu_uuids))
             and len({row["pid"] for row in rows}) == len(RANKS),
             "four distinct worker completion receipts required")
    for rank, (row, uuid) in enumerate(zip(rows, gpu_uuids, strict=True)):
        _require(row["uuid"].lower() == uuid.lower()
                 and (row["pp_rank"], row["tp_rank"], row["ep_rank"])
                 == (rank // TP_SIZE, rank % TP_SIZE, rank % TP_SIZE)
                 and row["device_synchronized"] is True and row["pending_pp_handles"] == 0
                 and row["pending_pp_sampled_tokens"] == 0,
                 f"rank {rank} worker binding or completion differs")
    if initial is not None:
        initial_by_rank = {row["rank"]: row for row in initial}
        _require(set(initial_by_rank) == set(RANKS) and len(initial) == len(RANKS)
                 and all(all(row[field] == initial_by_rank[row["rank"]][field]
                             for field in ("pid", "uuid", "device_index", "pp_rank", "tp_rank", "ep_rank"))
                         for row in rows),
                 "completion fence did not run in the original four workers")
    return rows


def _serial_pp_queue(handler, pp_rank):
    """Match the V2 sampled-token receive ring to the serial EngineCore.step."""
    _require(handler is not None and handler.is_last_rank is (pp_rank == PP_SIZE - 1),
             "V2 PP sampled-token handler is missing or on the wrong stage")
    queue = handler.queue
    expected = 0 if handler.is_last_rank else PP_SIZE
    _require(isinstance(queue, deque) and len(queue) == expected,
             "unexpected V2 PP sampled-token receive queue")
    warmup_slots = sum(slot is not None for slot in queue)
    if not handler.is_last_rank:
        # Warmup removes its requests but can leave completed, invalidated
        # receives. Let the native generation check discard those slots.
        for _ in range(expected):
            _require(handler.get_prev_sampled_outputs() is None,
                     "warmup left live V2 PP sampled-token state")
        # PPHandler ordinarily consumes output after PP_SIZE engine steps. The
        # serial scheduler can submit the same request on the very next step.
        handler.queue = deque([None])
    return queue, dict(before=expected, after=len(handler.queue), invalidated_warmup_slots=warmup_slots)


class _Capture:
    def __init__(self, worker, directory, requests, checkpoints, output_tokens, anchors):
        import torch

        self.runner = worker.model_runner
        self.model = self.runner.model
        self.identity = _identity(worker)
        self.rank = self.identity["rank"]
        self.layers = tuple(range(
            self.identity["pp_rank"] * LAYERS_PER_STAGE,
            (self.identity["pp_rank"] + 1) * LAYERS_PER_STAGE,
        ))
        self.requests = _request_table(requests)
        self.checkpoints, self.expected_anchors = _target_positions(
            self.requests, checkpoints, output_tokens, anchors or [],
        )
        self.discover_anchors = anchors is None
        self.live_anchors = set()
        self.directory = Path(directory) / f"rank{self.rank}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.layout_path = self.directory / "layout.jsonl"
        self.sample_path = self.directory / "samples.jsonl"
        self.layout_stream = None
        self.sample_stream = None
        self.bytes_written = 0
        self.step = 0
        self.completed_positions = set()
        self.internal_ids = {}
        self.batch = None
        self.frame = None
        self.handles = []
        self.patches = []
        self.original_pp_queue = None
        self.capture_actual_logits = False

        try:
            self.layout_stream = self.layout_path.open("x", encoding="utf-8")
            self.sample_stream = self.sample_path.open("x", encoding="utf-8")
            _require(worker.use_v2_model_runner and self.model.__class__.__name__ == "MixtralForCausalLM"
                     and self.runner.batch_sharder is None
                     and self.runner.vllm_config.scheduler_config.async_scheduling is False
                     and self.runner.num_speculative_steps == 0
                     and self.model.config.vocab_size == VOCAB_SIZE
                     and self.model.config.hidden_size == HIDDEN_SIZE
                     and self.model.config.num_hidden_layers == len(RANKS) * LAYERS_PER_STAGE // TP_SIZE
                     and self.model.model.start_layer == self.layers[0]
                     and self.model.model.end_layer == self.layers[-1] + 1
                     and all(p.dtype == torch.bfloat16 and p.device == worker.device
                             for p in self.model.parameters() if p.is_floating_point()),
                     "loaded model is outside the four-rank resident BF16 Mixtral contract")
            _require(self.runner.execute_model_state is None and self.runner.req_states.num_reqs == 0,
                     "must install capture before real admission, after warmup request cleanup")
            layout = self.runner.vllm_config.cache_config.kv_cache_layout
            _require(layout == "LBHNC", f"expected LBHNC KV cache layout, observed {layout!r}")
            self.layer_layouts = {}
            for layer_index in self.layers:
                layer = self.model.model.layers[layer_index]
                attention = layer.self_attn.attn
                cache = attention.kv_cache
                _require(attention.impl.__class__.__name__ == "TritonAttentionImpl"
                         and cache.dtype == torch.bfloat16 and cache.ndim == 4
                         and tuple(cache.shape[1:]) == (4, 16, 256),
                         f"unexpected BF16 Triton KV cache on layer {layer_index}")
                self.layer_layouts[str(layer_index)] = dict(
                    shape=list(cache.shape), stride=list(cache.stride()),
                    storage_offset=cache.storage_offset(), dtype=str(cache.dtype),
                )
                self._attach_layer_hooks(layer_index, layer, attention)
            self.handles.append(self.model.register_forward_pre_hook(self._begin_forward))
            self._patch(self.runner, "prepare_inputs", self._prepare_inputs)
            self._patch(self.runner, "sample_tokens", self._sample_tokens)
            if self.identity["pp_rank"] == PP_SIZE - 1:
                self._patch(self.runner, "sample", self._sample)
                self._patch(self.model, "compute_logits", self._compute_logits)
            torch.cuda.synchronize(worker.device)
            self.original_pp_queue, self.pp_cadence = _serial_pp_queue(
                self.runner.pp_handler, self.identity["pp_rank"],
            )
        except BaseException:
            self.close()
            raise

    def _patch(self, target, name, wrapper):
        # Restore the original descriptor lookup when the member was inherited.
        local = vars(target).get(name)
        had_local = name in vars(target)
        original = getattr(target, name)
        self.patches.append((target, name, had_local, local))
        setattr(target, name, wrapper(original))

    def _attach_layer_hooks(self, index, layer, attention):
        def pre_hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[1]
            self._capture_value(index, "layer_input", hidden)

        def output_hook(stage):
            def record(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                self._capture_value(index, stage, value)
            return record

        self.handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        for module, stage in (
            (layer.input_layernorm, "input_norm"),
            (layer.self_attn, "attention_output"),
            (attention, "attention_kernel"),
            (layer.post_attention_layernorm, "post_attention_norm"),
            (layer.block_sparse_moe, "moe_output"),
            (layer, "layer_output"),
        ):
            self.handles.append(module.register_forward_hook(output_hook(stage)))

    def _prepare_inputs(self, original):
        def wrapped(scheduler_output, batch_req_state, batch_desc):
            batch = original(scheduler_output, batch_req_state, batch_desc)
            _require(self.batch is None and self.frame is None,
                     "next actual InputBatch arrived before prior sampled step committed")
            self.batch = batch
            return batch
        return wrapped

    def _begin_forward(self, _module, _args):
        import torch

        batch = self.batch
        if batch is None:
            return
        _require(self.frame is None and batch.num_draft_tokens == 0
                 and batch.num_reqs == len(batch.req_ids)
                 and batch.num_reqs_after_padding == batch.num_reqs
                 and not batch.has_structured_output_reqs,
                 "unexpected padded requests, speculative or structured generation")
        from fluxmoe.host_kv_witness import pending_read_requests

        live_restores = pending_read_requests()
        total = batch.num_tokens
        starts = batch.query_start_loc[:batch.num_reqs + 1].detach().cpu().tolist()
        seq_lens = batch.seq_lens.detach().cpu().tolist()
        token_ids = batch.input_ids.detach().cpu().tolist()
        positions = batch.positions.detach().cpu().tolist()
        rows, selections, extra_logits = [], [], []
        if self.identity["pp_rank"] == PP_SIZE - 1:
            logits_rows = batch.logits_indices.detach().cpu().tolist()
        else:
            logits_rows = [end - 1 for end in starts[1:]]
        _require(starts == batch.query_start_loc_np[:batch.num_reqs + 1].tolist()
                 and starts[0] == 0 and starts[-1] == total > 0
                 and logits_rows == [end - 1 for end in starts[1:]]
                 and len(token_ids) == len(positions) == batch.num_tokens_after_padding,
                 "actual InputBatch query/logit/padding layout differs")
        for index, internal_id in enumerate(batch.req_ids):
            request = _external_id(internal_id, self.requests)
            _require(self.internal_ids.setdefault(request, internal_id) == internal_id,
                     f"request epoch changed within engine: {request}")
            begin, end = starts[index:index + 2]
            position = int(batch.num_computed_tokens_np[index])
            count = end - begin
            prefill_len = int(batch.prefill_len_np[index])
            seq_len = seq_lens[index]
            _require(count == int(batch.num_scheduled_tokens[index]) > 0
                     and seq_len == position + count
                     and positions[begin:end] == list(range(position, seq_len)),
                     f"actual scheduled positions differ for {request}")
            rows.append(dict(request_id=request, start=position, end=seq_len,
                             is_prefilling=bool(batch.is_prefilling_np[index]),
                             prefill_len=prefill_len))
            checkpoint = (request, seq_len - 1)
            checkpoint_label = ("checkpoint", *checkpoint)
            if (checkpoint in self.checkpoints and seq_len >= prefill_len
                    and checkpoint_label not in self.completed_positions):
                selections.append((end - 1, index, checkpoint_label))
            if internal_id in live_restores and all(k[0] != request for k in self.live_anchors):
                anchor = (request, live_restores[internal_id])
                _require(anchor[1] == position and position >= len(self.requests[request]),
                         "native restored read did not start at an actual decode-history query")
                _require(self.discover_anchors or anchor in self.expected_anchors,
                         f"reference observed a different first native restore: {anchor!r}")
                self.live_anchors.add(anchor)
                # The anchor can be the first row of a resumed multi-token
                # query. Read the actual sampler logits on single-token steps.
                selections.append((begin, index if begin == end - 1 else None,
                                   ("restore", *anchor)))
                if begin != end - 1:
                    extra_logits.append(begin)
        selections.sort(key=lambda selection: selection[0])
        _require(len({selection[2] for selection in selections}) == len(selections),
                 "duplicate sampled semantic positions in one step")
        payload_indices = None
        if selections:
            payload_indices = torch.tensor([row for row, _, _ in selections],
                                           dtype=torch.long, device=batch.input_ids.device)
        self.frame = dict(
            layout=dict(step=self.step, rows=rows, token_ids=token_ids, positions=positions,
                        num_tokens=total, padded_tokens=batch.num_tokens_after_padding),
            indices=payload_indices, labels=[key for _, _, key in selections],
            last_logits=[(slot, sample_index)
                         for slot, (_, sample_index, _) in enumerate(selections)
                         if sample_index is not None],
            extra_logits=extra_logits, arrays={}, logits={},
        )

    def _capture_value(self, layer, stage, value):
        import torch

        frame = self.frame
        if frame is None or not frame["labels"]:
            return
        key = f"L{layer:02d}_{stage}"
        _require(key not in frame["arrays"] and value.dtype == torch.bfloat16
                 and value.shape[0] == frame["layout"]["padded_tokens"],
                 f"duplicate or non-BF16 activation for {key}")
        frame["arrays"][key] = value.detach().index_select(0, frame["indices"]).flatten(1)

    def _sample(self, original):
        def wrapped(hidden_states, input_batch, grammar_output):
            frame = self.frame
            if frame and frame["extra_logits"]:
                import torch

                rows = torch.tensor(frame["extra_logits"], device=hidden_states.device, dtype=torch.long)
                logits = self.model.compute_logits(hidden_states.index_select(0, rows))
                selected = frame["indices"].detach().cpu().tolist()
                for index, row in enumerate(frame["extra_logits"]):
                    frame["logits"][selected.index(row)] = logits[index].detach().float().clone()
            self.capture_actual_logits = True
            try:
                return original(hidden_states, input_batch, grammar_output)
            finally:
                self.capture_actual_logits = False
        return wrapped

    def _compute_logits(self, original):
        def wrapped(hidden_states):
            logits = original(hidden_states)
            frame = self.frame
            if self.capture_actual_logits and frame and frame["last_logits"]:
                _require(logits is not None and logits.shape == (len(frame["layout"]["rows"]), VOCAB_SIZE),
                         "sampler did not consume complete vocabulary logits on each final TP rank")
                for slot, sample_index in frame["last_logits"]:
                    _require(slot not in frame["logits"], "duplicate logits capture")
                    frame["logits"][slot] = logits[sample_index].detach().float().clone()
            return logits
        return wrapped

    def _sample_tokens(self, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.batch is not None:
                _require(self.frame is not None, "actual scheduler step skipped captured model forward")
                self._commit_step()
            return result
        return wrapped

    def _commit_step(self):
        import numpy as np
        import torch

        frame = self.frame
        labels = frame["labels"]
        if labels:
            tensors = frame["arrays"]
            _require(set(tensors) == {f"L{layer:02d}_{stage}" for layer in self.layers for stage in STAGES},
                     "sampled step lacks all local activation and attention tensors")
            arrays = {key: value.to("cpu").contiguous().view(torch.uint16).numpy()
                      for key, value in tensors.items()}
            if self.identity["pp_rank"] == PP_SIZE - 1:
                _require(set(frame["logits"]) == set(range(len(labels))), "missing full vocabulary logits")
                arrays["logits"] = torch.stack([frame["logits"][i] for i in range(len(labels))]).cpu().numpy()
            path = self.directory / f"samples-{self.step:07d}.npz"
            pending = path.with_suffix(".tmp")
            raw_bytes = sum(value.nbytes for value in arrays.values())
            _require(self.bytes_written + raw_bytes <= MAX_EVIDENCE_PER_RANK,
                     "bounded raw numeric evidence exhausted")
            with pending.open("xb") as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            pending.replace(path)
            self.bytes_written += path.stat().st_size
            sample_row = dict(step=self.step, labels=labels, file=path.name, sha256=_sha(path))
            line = json.dumps(sample_row, separators=(",", ":"), allow_nan=False) + "\n"
            self.sample_stream.write(line)
            self.sample_stream.flush()
            os.fsync(self.sample_stream.fileno())
            self.bytes_written += len(line)
            self.completed_positions.update(labels)
        line = json.dumps(frame["layout"], separators=(",", ":"), allow_nan=False) + "\n"
        self.bytes_written += len(line)
        _require(self.bytes_written <= MAX_EVIDENCE_PER_RANK, "bounded InputBatch trace exhausted")
        self.layout_stream.write(line)
        self.layout_stream.flush()
        self.batch = self.frame = None
        self.step += 1

    def close(self):
        if self.original_pp_queue is not None:
            self.runner.pp_handler.queue = self.original_pp_queue
        for handle in self.handles:
            handle.remove()
        for target, name, had_local, local in reversed(self.patches):
            if had_local:
                setattr(target, name, local)
            else:
                delattr(target, name)
        for stream in (self.layout_stream, self.sample_stream):
            if stream is not None:
                stream.close()

    def finish(self):
        _require(self.batch is None and self.frame is None, "unfinished numeric step")
        expected = ({("checkpoint", *key) for key in self.checkpoints}
                    | {("restore", *key) for key in self.live_anchors})
        _require(self.completed_positions == expected
                 and (self.discover_anchors or self.live_anchors == self.expected_anchors),
                 "missing long output checkpoints or first native restore anchor")
        _require(set(self.internal_ids) == set(self.requests) and self.step > 0,
                 "real InputBatch omitted a declared request")
        self.layout_stream.flush()
        os.fsync(self.layout_stream.fileno())
        document = dict(
            **self.identity, status="completed", layers=list(self.layers), layer_layouts=self.layer_layouts,
            layout="LBHNC", backend="TRITON_ATTN", dtype="bfloat16", steps=self.step,
            pp_sample_queue=self.pp_cadence,
            layout_sha256=_sha(self.layout_path), sample_index_sha256=_sha(self.sample_path),
            restore_anchors=[dict(request_id=req, input_position=pos)
                             for req, pos in sorted(self.live_anchors)],
            internal_ids=self.internal_ids, bytes_written=self.bytes_written,
        )
        _write_json(self.directory / "capture.json", document)
        return dict(**self.identity, steps=self.step, samples=len(self.completed_positions),
                    restore_anchors=document["restore_anchors"],
                    manifest=str(self.directory / "capture.json"))


def install(worker, directory, requests, checkpoints, output_tokens, anchors=None):
    """Attach to actual model execution after loading, before first admission."""
    _require(not hasattr(worker, "_fluxmoe_host_kv_capture"), "capture already installed")
    capture = _Capture(worker, directory, requests, checkpoints, output_tokens, anchors)
    worker._fluxmoe_host_kv_capture = capture
    return dict(capture.identity, pp_sample_queue=capture.pp_cadence)


def finish(worker):
    """Finalize one rank's evidence and remove all diagnostic hooks."""
    capture = worker._fluxmoe_host_kv_capture
    try:
        return capture.finish()
    finally:
        capture.close()
        del worker._fluxmoe_host_kv_capture


def _validate_layout(row, request_table, output_tokens, histories, seen, checkpoint_positions):
    token_ids = row["token_ids"]
    positions = row["positions"]
    _require(0 < row["num_tokens"] <= row["padded_tokens"] == len(token_ids) == len(positions),
             f"invalid actual/padded token extent at step {row['step']}")
    _require(len({req["request_id"] for req in row["rows"]}) == len(row["rows"]) > 0,
             "duplicate or empty InputBatch requests")
    cursor = 0
    for req in row["rows"]:
        request_id = req["request_id"]
        prompt = request_table[request_id]
        start, end, prefill_len = req["start"], req["end"], req["prefill_len"]
        limit = len(prompt) + output_tokens - 1
        _require(0 <= start < end <= limit and len(prompt) <= prefill_len <= limit
                 and req["is_prefilling"] is (start < prefill_len),
                 f"query sequence extent differs for {request_id}")
        count = end - start
        _require(positions[cursor:cursor + count] == list(range(start, end)),
                 "actual absolute token positions differ from query sequence")
        actual_tokens = token_ids[cursor:cursor + count]
        if histories is not None:
            _require(actual_tokens == histories[request_id][start:end],
                     f"consumed token history differs for {request_id} at [{start}, {end})")
        elif start < len(prompt):
            _require(actual_tokens[:len(prompt) - start] == list(prompt[start:min(end, len(prompt))]),
                     f"consumed prompt token IDs differ for {request_id}")
        previous_end = seen.get(request_id, 0)
        _require(start <= previous_end, f"uncaptured position gap for {request_id}")
        seen[request_id] = max(previous_end, end)
        # A checkpoint is the last query row of a step that can emit a token.
        if end >= prefill_len:
            checkpoint_positions.add((request_id, end - 1))
        cursor += count
    _require(cursor == row["num_tokens"], "InputBatch query rows do not consume actual tokens")


def _capture_index(root, rank, request_table, expected_restore):
    directory = root / f"rank{rank}"
    path = directory / "capture.json"
    metadata = _read_json(path)
    layers = list(range(rank // TP_SIZE * LAYERS_PER_STAGE, (rank // TP_SIZE + 1) * LAYERS_PER_STAGE))
    cadence = metadata["pp_sample_queue"]
    depth = (PP_SIZE, 1) if rank // TP_SIZE == 0 else (0, 0)
    _require(metadata["status"] == "completed" and metadata["rank"] == rank
             and metadata["tp_rank"] == rank % TP_SIZE and metadata["ep_rank"] == rank % TP_SIZE
             and metadata["pp_rank"] == rank // TP_SIZE and metadata["layers"] == layers
             and set(metadata["layer_layouts"]) == {str(layer) for layer in layers}
             and metadata["layout"] == "LBHNC" and metadata["backend"] == "TRITON_ATTN"
             and metadata["dtype"] == "bfloat16"
             and (cadence["before"], cadence["after"]) == depth
             and 0 <= cadence["invalidated_warmup_slots"] <= depth[0]
             and set(metadata["internal_ids"]) == set(request_table)
             and len(set(metadata["internal_ids"].values())) == len(request_table)
             and all(_external_id(internal, request_table) == external
                     for external, internal in metadata["internal_ids"].items())
             and 0 < metadata["bytes_written"] <= MAX_EVIDENCE_PER_RANK,
             f"incomplete worker, model or storage layout capture for rank {rank}")
    anchors = metadata["restore_anchors"]
    _require(len(anchors) == len(expected_restore)
             and {(a["request_id"], a["input_position"]) for a in anchors} == expected_restore,
             f"rank {rank} capture differs from independently verified swap/read anchors")
    layout = directory / "layout.jsonl"
    _require(_sha(layout) == metadata["layout_sha256"], f"rank {rank} layout evidence changed")
    sample_by_step = {}
    sample_index = directory / "samples.jsonl"
    _require(_sha(sample_index) == metadata["sample_index_sha256"],
             f"rank {rank} sample inventory changed")
    bytes_written = layout.stat().st_size + sample_index.stat().st_size
    for sample in _layout_rows(sample_index):
        step = sample["step"]
        _require(type(step) is int and 0 <= step < metadata["steps"] and step not in sample_by_step,
                 "sample step is outside the recorded InputBatch trace")
        _require(sample["file"] == f"samples-{step:07d}.npz" and sample["labels"],
                 "invalid or empty numeric payload inventory")
        sample_path = directory / sample["file"]
        _require(sample_path.is_file() and not sample_path.is_symlink()
                 and 0 < sample_path.stat().st_size <= MAX_EVIDENCE_PER_RANK,
                 "invalid sample payload file")
        bytes_written += sample_path.stat().st_size
        sample_by_step[step] = (sample_path, sample)
    _require(bytes_written == metadata["bytes_written"], "numeric evidence byte inventory differs")
    return metadata, layout, sample_by_step


def _check_value(a, b, kind):
    import numpy as np

    _require(a.shape == b.shape, "same-layout numeric tensor shapes differ")
    if kind != "logits":
        _require(a.dtype == b.dtype == np.dtype("uint16"), "missing raw BF16 activation bits")
        a = (a.astype(np.uint32) << 16).view(np.float32)
        b = (b.astype(np.uint32) << 16).view(np.float32)
    else:
        _require(a.dtype == b.dtype == np.dtype("float32"), "missing full-vocab float32 logits")
    _require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite BF16 replay evidence")
    atol, rtol = ATTENTION_TOLERANCE if kind == "attention" else STRICT_TOLERANCE
    a, b = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a - b)
    return float(diff.max()), int(np.count_nonzero(diff > atol + rtol * np.abs(b)))


def compare(treatment_dir, reference_dir, requests, checkpoints, output_tokens, anchors, outputs=None):
    """Recompute the raw same-layout numerical gate without CUDA or vLLM.

    Outputs, when provided, are the already cross-checked full emitted token
    histories indexed by external request ID. Layout and actual sampler
    checkpoints are then bound to those histories.
    """
    import numpy as np

    roots = (Path(treatment_dir), Path(reference_dir))
    _require(roots[0].resolve() != roots[1].resolve(), "reference must be an independent engine run")
    request_table = _request_table(requests)
    target_checkpoints, target_anchors = _target_positions(
        request_table, checkpoints, output_tokens, anchors,
    )
    _require(bool(target_anchors), "same-layout reference requires a witnessed restored query")
    targets = ({("checkpoint", *key) for key in target_checkpoints}
               | {("restore", *key) for key in target_anchors})
    histories = None
    if outputs is not None:
        _require(set(outputs) == set(request_table)
                 and all(len(ids) == output_tokens
                         and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in ids)
                         for ids in outputs.values()),
                 "complete emitted token histories required")
        histories = {request: list(prompt) + outputs[request]
                     for request, prompt in request_table.items()}

    comparisons = Counter()
    maximum = Counter()
    failures = Counter()
    first_failure = {}
    logit_digests = {}
    rank_summaries = []
    common_layout_sha = None
    pids = set()
    devices = set()
    for rank in RANKS:
        (left, left_layout, left_samples), (right, right_layout, right_samples) = (
            _capture_index(root, rank, request_table, target_anchors) for root in roots
        )
        _require(left["uuid"] == right["uuid"] and left["uuid"] not in devices
                 and left["pid"] != right["pid"]
                 and left["pid"] not in pids and right["pid"] not in pids,
                 f"rank {rank} fresh model run or GPU assignment differs")
        devices.add(left["uuid"])
        pids.update((left["pid"], right["pid"]))
        _require(left["layer_layouts"] == right["layer_layouts"]
                 and left["steps"] == right["steps"]
                 and left["layout_sha256"] == right["layout_sha256"]
                 and set(left_samples) == set(right_samples),
                 f"rank {rank} actual InputBatch ordering, padding, cache layout or sample steps differ")
        common_layout_sha = common_layout_sha or left["layout_sha256"]
        _require(left["layout_sha256"] == common_layout_sha,
                 f"rank {rank} actual InputBatch does not match the other TP/PP ranks")
        covered, seen, consumed_samples = set(), {}, set()
        steps = 0
        for step, row in enumerate(_layout_rows(left_layout)):
            _require(row["step"] == step, "InputBatch steps must be contiguous")
            steps += 1
            actual_checkpoint_positions = set()
            _validate_layout(row, request_table, output_tokens, histories, seen, actual_checkpoint_positions)
            if step not in left_samples:
                continue
            sample_path, sample = left_samples[step]
            reference_path, reference = right_samples[step]
            _require(sample["labels"] == reference["labels"]
                     and _sha(sample_path) == sample["sha256"]
                     and _sha(reference_path) == reference["sha256"],
                     f"rank {rank} sample evidence bytes or labels differ at step {step}")
            labels = [tuple(label) for label in sample["labels"]]
            restored_query_positions = {(r["request_id"], r["start"]) for r in row["rows"]}
            for label in labels:
                kind, request_id, position = label
                available = actual_checkpoint_positions if kind == "checkpoint" else restored_query_positions
                _require(label in targets and label not in covered and (request_id, position) in available,
                         f"sample is extra, duplicated or mismatched to real InputBatch rows: {label}")
                covered.add(label)
            keys = {f"L{layer:02d}_{stage}" for layer in left["layers"] for stage in STAGES}
            if left["pp_rank"] == PP_SIZE - 1:
                keys.add("logits")
            with np.load(sample_path, allow_pickle=False) as a, np.load(reference_path, allow_pickle=False) as b:
                _require(len(a.files) == len(b.files) == len(keys) and set(a.files) == set(b.files) == keys,
                         "complete local activation and vocabulary tensor inventory required")
                for key in sorted(keys):
                    kind = ("logits" if key == "logits" else
                            "attention" if key.endswith("_attention_kernel") else "activation")
                    width = {"logits": VOCAB_SIZE, "attention": HIDDEN_SIZE // TP_SIZE,
                             "activation": HIDDEN_SIZE}[kind]
                    left_values, right_values = a[key], b[key]
                    _require(left_values.shape == right_values.shape == (len(labels), width),
                             f"incomplete sample geometry for rank {rank} step {step} tensor {key}")
                    max_error, mismatches = _check_value(left_values, right_values, kind)
                    comparisons[kind] += len(labels)
                    maximum[kind] = max(maximum[kind], max_error)
                    failures[kind] += mismatches
                    if mismatches:
                        first_failure.setdefault(kind, dict(rank=rank, step=step, tensor=key,
                                                            max_abs_error=max_error, mismatches=mismatches))
                    if kind == "logits":
                        for i, (label, request_id, position) in enumerate(labels):
                            row_digests = tuple(hashlib.sha256(arr[i].tobytes()).hexdigest()
                                                for arr in (left_values, right_values))
                            full_key = (label, request_id, position)
                            _require(logit_digests.setdefault(full_key, row_digests) == row_digests,
                                     "TP full-vocabulary all-gather differs")
                            if outputs is not None and label == "checkpoint":
                                target = outputs[request_id][position - len(request_table[request_id]) + 1]
                                _require(all(arr[i, target] == arr[i].max()
                                             for arr in (left_values, right_values)),
                                         f"captured greedy logits disagree with output for {full_key}")
            consumed_samples.add(step)
        _require(steps == left["steps"] and consumed_samples == set(left_samples) and covered == targets
                 and seen == {request: len(prompt) + output_tokens - 1
                              for request, prompt in request_table.items()},
                 f"rank {rank} did not capture all planned checkpoints and complete replay history")
        rank_summaries.append(dict(rank=rank, uuid=left["uuid"], treatment_pid=left["pid"],
                                   reference_pid=right["pid"], steps=steps, samples=len(covered)))
    _require(not any(failures.values()),
             f"strict same-layout replay mismatch: {dict(failures)}; first failing tensors: {first_failure}")
    return dict(status="FOUR_RANK_BF16_SAME_LAYOUT_PASSED", ranks=rank_summaries,
                layout_sha256=common_layout_sha, checkpoints=len(target_checkpoints),
                first_restored_query_anchors=len(target_anchors), comparison_rows=dict(comparisons),
                max_absolute_error=dict(maximum), tolerances=dict(
                    logits_and_activations=dict(atol=STRICT_TOLERANCE[0], rtol=STRICT_TOLERANCE[1]),
                    attention=dict(atol=ATTENTION_TOLERANCE[0], rtol=ATTENTION_TOLERANCE[1]),
                ), emitted_token_history_bound=outputs is not None,
                formal_performance_allowed=False, full_model_audit_passed=False)
