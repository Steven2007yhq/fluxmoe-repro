"""CPU-only checks of the real tiny Mixtral safetensors/tokenizer fixture."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from fluxmoe.checkpoint import (
    DEFAULT_SEED, SAMPLE_PROMPTS, TINY_MIXTRAL_CONFIG,
    build_tiny_moe_checkpoint, expected_vllm_expert_manifest, verify_tiny_moe_checkpoint,
)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors", "tokenizers")),
                     "requires local torch, transformers, safetensors and tokenizers")
class TinyMixtralCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "model"
        cls.manifest = build_tiny_moe_checkpoint(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        cls.torch.set_num_threads(cls.old_threads)

    def test_config_dtype_and_real_hf_load(self):
        from transformers import MixtralForCausalLM
        from fluxmoe.checkpoint import _tensor_hash
        model = MixtralForCausalLM.from_pretrained(self.path, local_files_only=True, dtype=self.torch.bfloat16).eval()
        for key, value in TINY_MIXTRAL_CONFIG.items():
            self.assertEqual(getattr(model.config, key), value)
        self.assertEqual(model.config.hidden_size // model.config.num_attention_heads, 32)
        self.assertFalse(next(model.parameters()).is_cuda)
        for name, tensor in model.state_dict().items():
            self.assertEqual(tensor.dtype, self.torch.bfloat16)
            self.assertEqual(_tensor_hash(tensor), self.manifest["hf_native_tensors"][name]["sha256"])
        self.assertLess(self.manifest["parameter_bytes"], 10 * 1024**2)
        self.assertEqual(self.manifest["parameters"], sum(p.numel() for p in model.parameters()))

    def test_legacy_expert_mapping_and_manifest(self):
        from safetensors.torch import load_file
        tensors = load_file(self.path / "model.safetensors")
        for layer in range(4):
            for expert in range(4):
                prefix = f"model.layers.{layer}.block_sparse_moe.experts.{expert}"
                for proj in ("w1", "w2", "w3"):
                    self.assertEqual(tensors[f"{prefix}.{proj}.weight"].shape, (128, 128))
        self.assertFalse(any(".mlp." in name for name in tensors))
        self.assertEqual(set(tensors), set(self.manifest["tensors"]))
        self.assertTrue(self.manifest["hf_load_validation"]["all_parameters_bitwise_equal"])
        self.assertEqual(verify_tiny_moe_checkpoint(self.path), self.manifest)

    def test_tokenizer_text_and_token_roundtrip(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.assertEqual(len(tokenizer), 256)
        self.assertEqual(tokenizer.model_max_length, 256)
        for prompt in SAMPLE_PROMPTS:
            ids = tokenizer.encode(prompt)
            self.assertEqual(ids[0], tokenizer.bos_token_id)
            self.assertNotIn(tokenizer.unk_token_id, ids)
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), prompt)
            self.assertEqual(tokenizer.encode(tokenizer.decode(ids, skip_special_tokens=True)), ids)

    def test_disk_expert_oracle_matches_original_hf_fused_parameters(self):
        expected = expected_vllm_expert_manifest(self.path)
        self.assertEqual(len(expected), 4)
        # These hashes were computed before checkpoint serialization, using the
        # HF model's own fused state. The oracle instead re-reads and concatenates
        # individual disk tensors, independently of both HF and vLLM loaders.
        for layer, row in enumerate(expected):
            self.assertEqual(row["layer"], layer)
            self.assertEqual(row["name"], f"model.layers.{layer}.block_sparse_moe.experts")
            self.assertEqual(row["w13_shape"], [4, 256, 128])
            self.assertEqual(row["w2_shape"], [4, 128, 128])
            native = self.manifest["hf_native_tensors"]
            self.assertEqual(row["w13_sha256"], native[f"model.layers.{layer}.mlp.experts.gate_up_proj"]["sha256"])
            self.assertEqual(row["w2_sha256"], native[f"model.layers.{layer}.mlp.experts.down_proj"]["sha256"])

    def test_determinism_and_preserved_rng_state(self):
        before = self.torch.random.get_rng_state().clone()
        duplicate = build_tiny_moe_checkpoint(Path(self.temp.name) / "duplicate")
        self.assertTrue(self.torch.equal(before, self.torch.random.get_rng_state()))
        self.assertEqual(self.manifest["tensors"], duplicate["tensors"])
        self.assertEqual(self.manifest["files"], duplicate["files"])
        self.assertEqual(build_tiny_moe_checkpoint(self.path), self.manifest)

    def test_refuses_mismatched_or_damaged_existing_directory(self):
        with self.assertRaises(FileExistsError):
            build_tiny_moe_checkpoint(self.path, seed=DEFAULT_SEED + 1)
        unknown = Path(self.temp.name) / "unknown"
        unknown.mkdir()
        marker = unknown / "keep.txt"
        marker.write_text("preserve")
        with self.assertRaises(FileExistsError):
            build_tiny_moe_checkpoint(unknown)
        self.assertEqual(marker.read_text(), "preserve")
        manifest_path = self.path / "fixture_manifest.json"
        original = manifest_path.read_text()
        try:
            manifest = json.loads(original)
            manifest["files"]["model.safetensors"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify_tiny_moe_checkpoint(self.path)
        finally:
            manifest_path.write_text(original)

    def test_real_attention_cached_decode_matches_full_sequence(self):
        from transformers import MixtralForCausalLM
        torch = self.torch
        # FP32 arithmetic on the same BF16 checkpoint isolates cache semantics
        # from CPU BF16 matmul rounding. This is not the vLLM GPU comparison.
        model = MixtralForCausalLM.from_pretrained(
            self.path, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        ids = torch.tensor([self.manifest["sample_prompts"][1]["token_ids"]])
        with torch.inference_mode():
            full = model(ids, use_cache=False).logits[:, -1]
            prefill = model(ids[:, :-1], use_cache=True)
            self.assertEqual(prefill.past_key_values.get_seq_length(), ids.shape[1] - 1)
            decoded = model(ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True)
            self.assertEqual(decoded.past_key_values.get_seq_length(), ids.shape[1])
        torch.testing.assert_close(decoded.logits[:, -1], full, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
