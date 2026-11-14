"""Observe native KV transfers without substituting their storage or scheduler."""
from __future__ import annotations

import re
import struct
from pathlib import Path

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group, get_world_group
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

from .host_kv_witness import (
    BLOCK_SIZE,
    IDENTITY_FIELDS,
    LAYER_BYTES,
    LAYERS_PER_STAGE,
    MAX_REQUESTS,
    PAYLOAD_BYTES,
    Journal,
    digest,
    require,
)


def _pair(gpu_spec, cpu_spec, gpu_block, host_block):
    """Bind the observed block to the actual native one-to-one copy plan."""
    require(isinstance(gpu_spec, GPULoadStoreSpec)
            and isinstance(cpu_spec, CPULoadStoreSpec), "wrong native transfer media")
    gpu_ids = gpu_spec.block_ids.tolist()
    cpu_ids = cpu_spec.block_ids.tolist()
    require(len(gpu_ids) == len(cpu_ids) and list(gpu_spec.group_sizes) == [len(gpu_ids)]
            and len(set(gpu_ids)) == len(gpu_ids) and len(set(cpu_ids)) == len(cpu_ids)
            and gpu_block in gpu_ids and host_block in cpu_ids
            and gpu_ids.index(gpu_block) == cpu_ids.index(host_block),
            "native copy plan does not pair the witnessed GPU and host pages")
    return len(gpu_ids)


def _request_tokens(request, logical):
    begin = logical * BLOCK_SIZE
    tokens = request.all_token_ids[begin:begin + BLOCK_SIZE]
    require(len(tokens) == BLOCK_SIZE, "incomplete logical decode block")
    return digest(struct.pack("<16i", *tokens))


def _patch_preemption():
    original = Scheduler._preempt_request
    if getattr(original, "_fluxmoe_kv_witness", False):
        return

    def preempt(scheduler, request, *args, **kwargs):
        connector = scheduler.connector
        if not isinstance(connector, WitnessOffloadingConnector):
            return original(scheduler, request, *args, **kwargs)
        before = connector.before_preemption(scheduler.kv_cache_manager, request)
        result = original(scheduler, request, *args, **kwargs)
        if before is not None:
            connector.after_preemption(scheduler.kv_cache_manager, request, *before)
        return result

    preempt._fluxmoe_kv_witness = True
    Scheduler._preempt_request = preempt


class WitnessOffloadingConnector(OffloadingConnector):
    """Capture a bounded, fully decoded 16-token page per request."""

    def __init__(self, vllm_config, role, kv_cache_config):
        parallel = vllm_config.parallel_config
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        cache = vllm_config.cache_config
        require((parallel.world_size, parallel.tensor_parallel_size,
                 parallel.pipeline_parallel_size) == (4, 2, 2)
                and not cache.enable_prefix_caching
                and not vllm_config.scheduler_config.async_scheduling
                and vllm_config.speculative_config is None
                and vllm_config.model_config.enforce_eager
                and vllm_config.use_v2_model_runner,
                "host-KV witness requires synchronous eager V2 TP2/PP2 without prefix caching/speculation")
        require(cache.block_size == BLOCK_SIZE and cache.kv_cache_layout == "LBHNC",
                "host-KV witness requires 16-token LBHNC pages")
        super().__init__(vllm_config, role, kv_cache_config)
        require(not self._canonical_layout, "host-KV witness requires direct native copies")
        root = Path(extra["witness_dir"])
        self.witness_reads = {}
        if role == KVConnectorRole.SCHEDULER:
            scheduler = self.connector_scheduler
            require(not scheduler.config.offload_prompt_only and
                    scheduler.config.blocks_per_chunk == 1 and
                    scheduler.config.tokens_per_hash == BLOCK_SIZE and
                    len(scheduler.config.kv_group_configs) == 1 and
                    scheduler.config.kv_group_configs[0].tokens_per_block == BLOCK_SIZE and
                    scheduler.config.kv_group_configs[0].sliding_window_size_in_chunks is None and
                    not scheduler.config.kv_group_configs[0].requires_cow_source,
                    "host-KV witness requires one native full-attention cache group")
            self.journal = Journal(root, "scheduler")
            self.journal.record("pool", host_cap=extra["cpu_bytes_to_use"])
            self.salts = set()
            self.request_ids = set()
            self.stores = {}
            self.loads = {}
            self.jobs = {}
            self.completions = set()
            self.releases = {}
            self.read_requests = set()
            _patch_preemption()
        else:
            self.rank = get_world_group().rank
            self.pp_rank = get_pp_group().rank_in_group
            self.tp_rank = get_tp_group().rank_in_group
            require(self.rank == 2 * self.pp_rank + self.tp_rank, "unexpected GPU rank mapping")
            self.journal = Journal(root, f"rank-{self.rank}")
            self.worker_jobs = {}
            self.worker_done = set()
            self.attention_pages = {}
            self.store_payloads = {}

    def on_new_request(self, request):
        salt = request.cache_salt
        rid = request.request_id
        require(isinstance(salt, str) and bool(salt) and salt not in self.salts
                and rid not in self.request_ids and len(self.request_ids) < MAX_REQUESTS,
                "host-KV witness needs unique request IDs and salts within one bounded batch")
        super().on_new_request(request)
        self.salts.add(salt)
        self.request_ids.add(rid)
        self.journal.record("request", request_id=rid, salt_sha256=digest(salt.encode()),
                            prompt_tokens=request.num_prompt_tokens)

    def before_preemption(self, manager, request):
        rid = request.request_id
        self.loads.pop(rid, None)
        self.releases.pop(rid, None)
        store = self.stores.get(rid)
        if store is None or store["job"] not in self.completions or rid in self.read_requests:
            return None
        (ids,) = manager.get_block_ids(rid)
        logical = store["logical_block"]
        if logical >= len(ids) or ids[logical] != store["gpu_block"]:
            return None
        block = manager.block_pool.blocks[ids[logical]]
        if block.ref_cnt != 1 or block.is_null:
            return None
        require(block.prev_free_block is None and block.next_free_block is None,
                "owned GPU block appeared in the free queue")
        return store, block, request.num_preemptions

    def after_preemption(self, manager, request, store, block, previous):
        (after,) = manager.get_block_ids(request.request_id)
        left, right = block.prev_free_block, block.next_free_block
        linked = (left is not None and right is not None and
                  left.next_free_block is block and right.prev_free_block is block)
        require(not after and block.ref_cnt == 0 and linked
                and request.num_preemptions == previous + 1,
                "old GPU allocation did not return to the free pool on preemption")
        self.releases[request.request_id] = self.journal.record(
            "gpu_released", request_id=request.request_id, gpu_block=block.block_id,
            logical_block=store["logical_block"], refs_before=1, refs_after=block.ref_cnt,
            table_after=list(after), preemptions=request.num_preemptions, free_queue_linked=linked,
            free_queue_neighbors=[left.block_id, right.block_id],
        )

    def _record_store(self, job, transfer):
        rid = transfer.req_id
        if rid in self.stores:
            return None
        scheduler = self.connector_scheduler
        state = scheduler._req_status[rid]
        request = state.req
        logical = (request.num_prompt_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        (group,) = state.group_states
        if logical >= len(group.offload_keys) or logical >= len(group.block_ids):
            return None
        key = group.offload_keys[logical]
        if key not in scheduler._jobs[job].keys:
            return None
        host = scheduler.manager._policy.get(key)
        require(host is not None, "native planned host block is absent")
        gpu_block = group.block_ids[logical]
        blocks = _pair(transfer.src_spec, transfer.dst_spec, gpu_block, host.block_id)
        row = self.journal.record(
            "store_planned", job=job, job_blocks=blocks, direction="store", request_id=rid,
            salt_sha256=digest(request.cache_salt.encode()), native_key=key.hex(),
            logical_block=logical, token_sha256=_request_tokens(request, logical),
            prompt_tokens=request.num_prompt_tokens, gpu_block=gpu_block, host_block=host.block_id,
        )
        self.stores[rid] = row
        self.jobs[job] = row
        return row

    def _record_load(self, job, transfer):
        rid = transfer.req_id
        release = self.releases.get(rid)
        if release is None or rid in self.read_requests:
            return None
        store = self.stores[rid]
        scheduler = self.connector_scheduler
        state = scheduler._req_status[rid]
        request = state.req
        logical = store["logical_block"]
        (group,) = state.group_states
        key = group.offload_keys[logical]
        host = scheduler.manager._policy.get(key)
        if (key.hex() != store["native_key"] or key not in scheduler._jobs[job].keys
                or host is None or host.block_id != store["host_block"]):
            return None
        (first,) = transfer.dst_spec.block_indices
        index = logical - first
        if not 0 <= index < len(transfer.dst_spec.block_ids):
            return None
        gpu_block = int(transfer.dst_spec.block_ids[index])
        blocks = _pair(transfer.dst_spec, transfer.src_spec, gpu_block, host.block_id)
        require(_request_tokens(request, logical) == store["token_sha256"]
                and request.num_preemptions == release["preemptions"]
                and digest(request.cache_salt.encode()) == store["salt_sha256"],
                "native load request generation or logical token history changed")
        row = self.journal.record(
            "load_planned", job=job, job_blocks=blocks, direction="load",
            store_job=store["job"], release_index=release["index"],
            gpu_block=gpu_block, preemptions=request.num_preemptions,
            **{key: store[key] for key in IDENTITY_FIELDS},
        )
        self.loads[rid] = row
        self.jobs[job] = row
        return row

    def _read_plan(self, output):
        reads = {}
        # V2 publishes resumed requests as full NewRequestData records.
        for request in output.scheduled_new_reqs:
            request_id = request.req_id
            loaded = self.loads.get(request_id)
            if (loaded is None or loaded["job"] not in self.completions
                    or request_id in self.read_requests):
                continue
            first_pos = request.num_computed_tokens
            new_tables = request.block_ids
            require(len(new_tables) == 1
                    and loaded["logical_block"] < len(new_tables[0])
                    and new_tables[0][loaded["logical_block"]] == loaded["gpu_block"]
                    and first_pos >= (loaded["logical_block"] + 1) * BLOCK_SIZE,
                    "first scheduled read does not bind the restored history page")
            row = self.journal.record(
                "scheduled_read", load_job=loaded["job"], store_job=loaded["store_job"],
                release_index=loaded["release_index"], preemptions=loaded["preemptions"],
                gpu_block=loaded["gpu_block"], query_position=first_pos,
                query_count=output.num_scheduled_tokens[request_id], block_table=new_tables[0],
                **{key: loaded[key] for key in IDENTITY_FIELDS},
            )
            reads[loaded["job"]] = row
            self.read_requests.add(request_id)
        return reads

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        jobs = {}
        for direction, record in ((meta.store_jobs, self._record_store),
                                  (meta.load_jobs, self._record_load)):
            for job, transfer in direction.items():
                row = record(job, transfer)
                if row is not None:
                    jobs[job] = row
        meta.fluxmoe_witness_jobs = jobs
        meta.fluxmoe_witness_reads = self._read_plan(scheduler_output)
        return meta

    def update_connector_output(self, connector_output):
        native = self.connector_scheduler
        meta = connector_output.kv_connector_worker_meta
        candidates = {} if meta is None else {
            job: (count, native._jobs[job].pending_count)
            for job, count in meta.completed_jobs.items() if job in self.jobs and job in native._jobs
        }
        super().update_connector_output(connector_output)
        for job, (reported, pending_before) in candidates.items():
            if job not in native._jobs and job not in self.completions:
                require(reported == pending_before > 0, "native completion counts disagree")
                self.completions.add(job)
                self.journal.record("transfer_done", job=job, direction=self.jobs[job]["direction"],
                                    ranks_completed=native.config.num_workers, reported=reported)

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        worker = self.connector_worker.worker
        require(isinstance(worker, CPUOffloadingWorker) and self.connector_worker._is_store_writer,
                "host-KV witness requires an actual local native store writer on each GPU")
        store = worker._store_handler
        load = worker._load_handler
        region = worker._mmap_region
        require(region is not None and region.is_pinned and region.rank == self.rank,
                "actual native CPU mmap was not CUDA-pinned for this rank")
        require(store._canonical_copy_plans is None and load._canonical_copy_plans is None
                and store.src_blocks_per_chunk == store.dst_blocks_per_chunk == 1
                and load.src_blocks_per_chunk == load.dst_blocks_per_chunk == 1,
                "native copy geometry changed")
        gpu_count = self.connector_worker.kv_cache_config.num_blocks
        (refs,) = store.layer_refs_per_group
        require(len(refs) == len(store.src_tensors) == len(store.dst_tensors) == LAYERS_PER_STAGE
                and {ref.tensor_idx for ref in refs} == set(range(LAYERS_PER_STAGE))
                and all(ref.page_size_bytes == LAYER_BYTES for ref in refs),
                "native transfer does not cover each local K/V layer")
        ptr_to_tensor = {tensor.data_ptr(): i for i, tensor in enumerate(store.src_tensors)}
        pages = {}
        layer_numbers = {}
        for name, cache in kv_caches.items():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            require(match is not None and cache.is_cuda and cache.dtype == torch.bfloat16
                    and cache.shape == (gpu_count, 4, BLOCK_SIZE, 256) and cache.is_contiguous(),
                    f"unsupported Triton BF16 cache layout for {name}")
            index = ptr_to_tensor.get(cache.data_ptr())
            require(index is not None, f"native worker does not transfer registered cache {name}")
            gpu, host = store.src_tensors[index], store.dst_tensors[index]
            require(gpu.shape == (gpu_count, LAYER_BYTES) and host.shape == (region.num_blocks, LAYER_BYTES)
                    and gpu.data_ptr() == load.dst_tensors[index].data_ptr()
                    and host.data_ptr() == load.src_tensors[index].data_ptr(),
                    f"store/load backing storage differs for {name}")
            layer = int(match.group(1))
            require(layer not in pages, "duplicate physical attention layer")
            pages[layer] = (gpu, host)
            layer_numbers[name] = layer
        start = self.pp_rank * LAYERS_PER_STAGE
        self.layers = list(range(start, start + LAYERS_PER_STAGE))
        require(sorted(pages) == self.layers
                and len({gpu.data_ptr() for gpu, _ in pages.values()}) == LAYERS_PER_STAGE
                and len({host.data_ptr() for _, host in pages.values()}) == LAYERS_PER_STAGE,
                "worker KV pages do not cover all local PP layers")
        self.pages = pages
        self.layer_numbers = layer_numbers
        self.worker = worker
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid).lower()
        require(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid) is not None,
                "invalid CUDA device UUID")
        self.journal.record(
            "pool", rank=self.rank, pp_rank=self.pp_rank, tp_rank=self.tp_rank,
            uuid="GPU-" + uuid, layers=self.layers, gpu_blocks=gpu_count,
            cpu_blocks=region.num_blocks, host_pool_bytes=region.total_size_bytes,
            pinned=region.is_pinned, page_bytes=PAYLOAD_BYTES,
        )
        self._install_worker_observer(worker)

    def _capture(self, job, kind, host=False):
        column, block = (1, job["host_block"]) if host else (0, job["gpu_block"])
        data = b"".join(self.pages[layer][column][block].cpu().numpy().tobytes()
                        for layer in self.layers)
        return self.journal.payload(job["job"], kind, data)

    @staticmethod
    def _transfer_fields(job):
        return {key: job[key] for key in (*IDENTITY_FIELDS, "gpu_block", "job", "job_blocks")}

    def _check_payload(self, planned, payload):
        source_job = planned["job"] if planned["direction"] == "store" else planned["store_job"]
        require(payload["sha256"] == self.store_payloads[source_job],
                f"native host-KV witness changed bits: {payload['path']}")

    def _install_worker_observer(self, worker):
        submit_store = worker.submit_store
        submit_load = worker.submit_load
        finished = worker.get_finished

        def store(job, src, dst):
            planned = self.worker_jobs.get(job)
            if planned is not None:
                require(planned["direction"] == "store" and
                        _pair(src, dst, planned["gpu_block"], planned["host_block"]) == planned["job_blocks"],
                        "native worker submitted a different D2H plan")
                payload = self._capture(planned, "gpu-original")
                self.store_payloads[job] = payload["sha256"]
                self.journal.record("store_source", **self._transfer_fields(planned),
                                    payload=payload)
            return submit_store(job, src, dst)

        def load(job, src, dst):
            planned = self.worker_jobs.get(job)
            if planned is not None:
                require(planned["direction"] == "load" and planned["store_job"] in self.worker_done and
                        _pair(dst, src, planned["gpu_block"], planned["host_block"]) == planned["job_blocks"],
                        "native worker submitted a different H2D plan")
                payload = self._capture(planned, "host-before-load", host=True)
                self.journal.record("load_source", **self._transfer_fields(planned),
                                    payload=payload)
                self._check_payload(planned, payload)
            return submit_load(job, src, dst)

        def get_finished():
            results = finished()
            for result in results:
                planned = self.worker_jobs.get(result.job_id)
                if planned is None:
                    continue
                require(result.success and result.transfer_size == PAYLOAD_BYTES * planned["job_blocks"],
                        "native completion does not prove the actual transferred bytes")
                is_store = planned["direction"] == "store"
                kind = "host-after-store" if is_store else "gpu-restored"
                payload = self._capture(planned, kind, host=is_store)
                self.journal.record(
                    planned["direction"] + "_done", **self._transfer_fields(planned),
                    transfer_bytes=result.transfer_size, payload=payload,
                )
                self._check_payload(planned, payload)
                self.worker_done.add(result.job_id)
            return results

        worker.submit_store = store
        worker.submit_load = load
        worker.get_finished = get_finished

    def _bind_jobs(self, metadata):
        self.worker_jobs.update(getattr(metadata, "fluxmoe_witness_jobs", {}))

    def handle_preemptions(self, metadata):
        self._bind_jobs(metadata)
        return super().handle_preemptions(metadata)

    def bind_connector_metadata(self, metadata):
        super().bind_connector_metadata(metadata)
        self._bind_jobs(metadata)
        self.witness_reads = getattr(metadata, "fluxmoe_witness_reads", {})
        require(all(job in self.worker_done for job in self.witness_reads),
                "attention scheduled before local H2D completion")
        self.attention_pages.clear()

    def clear_connector_metadata(self):
        require(all(set(self.attention_pages.get(job, ())) == set(self.layers)
                    for job in self.witness_reads), "resumed query did not run every local attention layer")
        self.witness_reads = {}
        self.attention_pages.clear()
        return super().clear_connector_metadata()

    def get_finished(self, finished_req_ids):
        # Diagnostic synchronization of native transfer events, identically in
        # the measured and replay phases, to stabilize their admission schedule.
        worker = self.worker
        pending = set(worker._store_handler._transfer_events) | set(worker._load_handler._transfer_events)
        worker.wait(pending)
        return super().get_finished(finished_req_ids)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        if not self.witness_reads:
            return
        require(type(attn_metadata).__name__ == "TritonAttentionMetadata"
                and attn_metadata.causal is True and not attn_metadata.use_cascade,
                "witness requires actual full causal Triton attention")
        _, attention, _, _ = get_attention_context(layer_name)
        require(type(attention.impl).__name__ == "TritonAttentionImpl"
                and attention.impl.sliding_window == (-1, -1)
                and attn_metadata.rswa_window is None and attn_metadata.mm_prefix_range is None,
                "restored page is outside the supported full-history attention contract")
        layer = self.layer_numbers[layer_name]
        require(kv_layer.data_ptr() == self.pages[layer][0].data_ptr(),
                "attention did not consume registered native cache")
        starts = attn_metadata.query_start_loc.cpu().tolist()
        lengths = attn_metadata.seq_lens.cpu().tolist()
        slots = attn_metadata.slot_mapping[:attn_metadata.num_actual_tokens].cpu().tolist()
        require(len(starts) == len(lengths) + 1, "invalid Triton row boundaries")
        for job, read in self.witness_reads.items():
            logical, block = read["logical_block"], read["gpu_block"]
            require(logical < attn_metadata.block_table.shape[1], "missing logical block column")
            column = attn_metadata.block_table[:len(lengths), logical].cpu().tolist()
            candidates = [
                i for i, (table_block, length, first, last) in enumerate(
                    zip(column, lengths, starts[:-1], starts[1:], strict=True))
                if table_block == block and last - first == read["query_count"]
                and length - (last - first) == read["query_position"]
            ]
            require(len(candidates) == 1 and not any(slot // BLOCK_SIZE == block for slot in slots),
                    "Triton attention did not uniquely read the restored page as prior history")
            row = candidates[0]
            seq_len = lengths[row]
            active_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            raw_row = {
                "layer": layer, "row": row, "query_start": starts[row],
                "query_stop": starts[row + 1], "seq_len": seq_len,
                "block_table": attn_metadata.block_table[row, :active_blocks].cpu().tolist(),
                "write_slots": slots[starts[row]:starts[row + 1]],
            }
            captures = self.attention_pages.setdefault(job, {})
            require(layer not in captures, "duplicate actual attention for restored block")
            captures[layer] = (self.pages[layer][0][block].cpu().numpy().tobytes(), raw_row)
            if sorted(captures) == self.layers:
                planned = self.worker_jobs[job]
                payload = self.journal.payload(job, "gpu-attention", b"".join(
                    captures[index][0] for index in self.layers))
                self.journal.record(
                    "attention_read", **self._transfer_fields(planned),
                    query_position=read["query_position"], query_count=read["query_count"],
                    rows=[captures[index][1] for index in self.layers], payload=payload,
                )
                self._check_payload(planned, payload)
