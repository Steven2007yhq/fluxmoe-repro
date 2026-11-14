import json
from collections import deque
from pathlib import Path
import tempfile
import unittest

from fluxmoe import host_kv_reference as replay

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is required to audit raw replay arrays")
class HostKVReferenceTest(unittest.TestCase):
    requests = [{"request_id": "q", "prompt_token_ids": [11, 12]}]
    checkpoints = (1, 2, 3)
    anchors = [{"request_id": "q", "input_position": 2}]
    outputs = {"q": [13, 14, 15]}

    @staticmethod
    def _json_lines(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _arm(self, root, phase, *, different_layout=False, activation_error=False,
             missing_restore=False):
        layout_rows = [
            dict(step=0, rows=[dict(request_id="q", start=0, end=2, prefill_len=2, is_prefilling=True)],
                 num_tokens=2, padded_tokens=2, positions=[0, 1], token_ids=[11, 12]),
            dict(step=1, rows=[dict(request_id="q", start=2, end=3, prefill_len=2, is_prefilling=False)],
                 num_tokens=1, padded_tokens=1, positions=[2], token_ids=[13]),
            dict(step=2, rows=[dict(request_id="q", start=3, end=4, prefill_len=2, is_prefilling=False)],
                 num_tokens=1, padded_tokens=1, positions=[3], token_ids=[14]),
        ]
        if different_layout:
            layout_rows[1].update(padded_tokens=2, positions=[2, 0], token_ids=[13, 0])
        for rank in replay.RANKS:
            rank_dir = root / f"rank{rank}"
            rank_dir.mkdir(parents=True)
            layout_path = rank_dir / "layout.jsonl"
            self._json_lines(layout_path, layout_rows)
            start = rank // 2 * replay.LAYERS_PER_STAGE
            layers = list(range(start, start + replay.LAYERS_PER_STAGE))
            inventory = []
            for step in range(3):
                labels = [("checkpoint", "q", step + 1)]
                if step == 1 and not missing_restore:
                    labels.append(("restore", "q", 2))
                arrays = {
                    f"L{layer:02d}_{stage}": np.zeros(
                        (len(labels), replay.HIDDEN_SIZE // 2 if stage == "attention_kernel" else replay.HIDDEN_SIZE),
                        dtype=np.uint16,
                    )
                    for layer in layers for stage in replay.STAGES
                }
                if activation_error and rank == 1 and step == 1:
                    arrays[f"L{start:02d}_layer_input"][1, 3] = 0x3F80
                if rank >= 2:
                    arrays["logits"] = np.zeros((len(labels), replay.VOCAB_SIZE), dtype=np.float32)
                    arrays["logits"][:, self.outputs["q"][step]] = 1
                filename = f"samples-{step:07d}.npz"
                path = rank_dir / filename
                np.savez(path, **arrays)
                inventory.append(dict(step=step, labels=labels, file=filename, sha256=replay._sha(path)))
            index_path = rank_dir / "samples.jsonl"
            self._json_lines(index_path, inventory)
            size = sum(path.stat().st_size for path in rank_dir.iterdir())
            replay._write_json(rank_dir / "capture.json", dict(
                status="completed", rank=rank, tp_rank=rank % 2, ep_rank=rank % 2, pp_rank=rank // 2,
                layers=layers, layer_layouts={str(layer): dict(layout="fixture") for layer in layers},
                pid=400 + rank + phase * 100, uuid=f"GPU-{rank}", layout="LBHNC", backend="TRITON_ATTN",
                dtype="bfloat16", internal_ids={"q": "q"}, bytes_written=size,
                pp_sample_queue=dict(before=2 if rank < 2 else 0,
                                     after=1 if rank < 2 else 0, invalidated_warmup_slots=0),
                restore_anchors=self.anchors, steps=3, layout_sha256=replay._sha(layout_path),
                sample_index_sha256=replay._sha(index_path),
            ))

    def _compare(self, treatment, reference, **kwargs):
        return replay.compare(treatment, reference, self.requests, self.checkpoints, 3, self.anchors,
                              outputs=kwargs.get("outputs", self.outputs))

    def test_recomputed_replay_and_negative_cases(self):
        cases = (
            ({}, None),
            (dict(different_layout=True), "actual InputBatch"),
            (dict(activation_error=True), "strict same-layout replay mismatch"),
            (dict(missing_restore=True), "evidence bytes or labels differ"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            treatment = root / "treatment"
            self._arm(treatment, 0)
            for index, (variant, error) in enumerate(cases):
                with self.subTest(variant=variant):
                    reference = root / f"reference-{index}"
                    self._arm(reference, 1, **variant)
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            self._compare(treatment, reference)
                    else:
                        result = self._compare(treatment, reference)
                        self.assertEqual(result["status"], "FOUR_RANK_BF16_SAME_LAYOUT_PASSED")
                        self.assertEqual(result["comparison_rows"], dict(logits=8, activation=1536, attention=256))
                        with self.assertRaisesRegex(ValueError, "consumed token history differs"):
                            self._compare(treatment, reference, outputs={"q": [14, 13, 15]})

    def test_fence_rejects_worker_replacement(self):
        uuids = [f"GPU-{i}" for i in replay.RANKS]
        initial = [
            dict(rank=i, uuid=uuid, pid=400 + i, pp_rank=i // 2, tp_rank=i % 2, ep_rank=i % 2,
                 device_index=i)
            for i, uuid in enumerate(uuids)
        ]
        fences = [dict(row, device_synchronized=True, pending_pp_handles=0,
                       pending_pp_sampled_tokens=0) for row in initial]
        self.assertEqual(replay.validate_fences(fences, uuids, initial), fences)
        fences[-1]["pid"] += 10
        with self.assertRaisesRegex(ValueError, "original four workers"):
            replay.validate_fences(fences, uuids, initial)

    def test_serial_receive_cadence_and_warmup_cleanup(self):
        class Handler:
            is_last_rank = False

            def __init__(self, warmup):
                self.queue = deque([warmup, None])

            def get_prev_sampled_outputs(self):
                slot = self.queue.popleft()
                self.queue.append(None)
                return None if slot in (None, "invalidated") else slot

        handler = Handler("invalidated")
        old, receipt = replay._serial_pp_queue(handler, 0)
        self.assertEqual(receipt, dict(before=2, after=1, invalidated_warmup_slots=1))
        self.assertEqual(list(old), [None, None])
        handler.queue[-1] = "current request token"
        self.assertEqual(handler.get_prev_sampled_outputs(), "current request token")
        with self.assertRaisesRegex(ValueError, "warmup left live"):
            replay._serial_pp_queue(Handler("live"), 0)
