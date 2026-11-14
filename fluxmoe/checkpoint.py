"""Offline, deterministic Mixtral checkpoint for inference correctness tests.

This is a randomly initialized architecture fixture, NOT a pretrained language
model.  It exercises real attention, RoPE, MoE routing, KV caches and the LM head.
The on-disk expert names use the original Hugging Face Mixtral convention that
vLLM consumes; Transformers 5 converts these to its fused expert parameters.
Heavy dependencies are imported only when building or validating a checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


FIXTURE_VERSION = 1
DEFAULT_SEED = 27415
TINY_MIXTRAL_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "max_position_embeddings": 256,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "tie_word_embeddings": False,
    "attention_dropout": 0.0,
    "router_jitter_noise": 0.0,
    "sliding_window": None,
}
SAMPLE_PROMPTS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "one two three four five",
    "the quick brown fox reads a small book",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor: Any) -> str:
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _legacy_mixtral_state(state: dict[str, Any]) -> dict[str, Any]:
    """Export HF 5 fused experts as HF 4/vLLM per-expert w1/w2/w3 tensors."""
    exported = {}
    for name, tensor in state.items():
        match = re.fullmatch(r"(model\.layers\.\d+)\.mlp\.experts\.(gate_up_proj|down_proj)", name)
        if match:
            prefix, kind = match.groups()
            for expert in range(tensor.shape[0]):
                base = f"{prefix}.block_sparse_moe.experts.{expert}"
                if kind == "gate_up_proj":
                    gate, up = tensor[expert].chunk(2, dim=0)
                    exported[f"{base}.w1.weight"] = gate.clone().contiguous()
                    exported[f"{base}.w3.weight"] = up.clone().contiguous()
                else:
                    exported[f"{base}.w2.weight"] = tensor[expert].clone().contiguous()
        else:
            exported[name.replace(".mlp.", ".block_sparse_moe.")] = tensor.clone().contiguous()
    return dict(sorted(exported.items()))


def _build_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    words = ("hello world the quick brown fox jumps over lazy dog one two three four five "
             "six seven eight nine ten reads a small book model cache memory gpu expert "
             "token test this is an example of local generation with attention and "
             "routing we verify weights are loaded correctly prompt answer yes no").split()
    vocab = {token: index for index, token in enumerate(["<pad>", "<bos>", "<eos>", "<unk>"] + words)}
    while len(vocab) < TINY_MIXTRAL_CONFIG["vocab_size"]:
        vocab[f"token_{len(vocab):03d}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
        model_max_length=TINY_MIXTRAL_CONFIG["max_position_embeddings"],
        clean_up_tokenization_spaces=False,
    )


def verify_tiny_moe_checkpoint(output: str | os.PathLike[str], seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Verify fixture identity and every recorded artifact hash without GPU use."""
    directory = Path(output)
    manifest_path = directory / "fixture_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"Refusing to replace an unrecognized checkpoint directory: {directory}")
    manifest = json.loads(manifest_path.read_text())
    expected = {"fixture_version": FIXTURE_VERSION, "seed": seed, "config": TINY_MIXTRAL_CONFIG}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise FileExistsError(f"Existing checkpoint has a different {field}: {directory}")
    for name, record in manifest["files"].items():
        file = directory / name
        if not file.is_file() or file.stat().st_size != record["bytes"] or _sha256(file) != record["sha256"]:
            raise ValueError(f"Checkpoint integrity mismatch: {file}")
    return manifest


def expected_vllm_expert_manifest(output: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Hash expert weights directly from the checkpoint, independently of vLLM.

    Matches ``loading.expert_weights`` from the engine integration for the
    supported single-GPU, unquantized NVIDIA Triton path. In vLLM 0.29 that path
    preserves ``w13=[w1;w3]`` and ``w2`` after kernel postprocessing. Other
    backends (notably FlashInfer), TP/EP sharding, or quantization need a different
    oracle and must not use these hashes. No vLLM loader or GPU is invoked here.
    """
    import torch
    from safetensors.torch import load_file

    output = Path(output)
    identity = json.loads((output / "fixture_manifest.json").read_text())
    manifest = verify_tiny_moe_checkpoint(output, seed=identity["seed"])
    if manifest["checkpoint_layout"] != "hf_mixtral_per_expert_w1_w2_w3":
        raise ValueError("Expected the per-expert Mixtral correctness fixture layout")
    config = manifest["config"]
    tensors = load_file(output / "model.safetensors", device="cpu")
    rows = []
    for layer in range(config["num_hidden_layers"]):
        name = f"model.layers.{layer}.block_sparse_moe.experts"
        w13, w2 = [], []
        for expert in range(config["num_local_experts"]):
            base = f"{name}.{expert}"
            gate, up, down = [tensors[f"{base}.{projection}.weight"] for projection in ("w1", "w3", "w2")]
            for tensor, shape in ((gate, (config["intermediate_size"], config["hidden_size"])),
                                  (up, (config["intermediate_size"], config["hidden_size"])),
                                  (down, (config["hidden_size"], config["intermediate_size"]))):
                if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                    raise ValueError(f"Unsupported dtype or shape in checkpoint expert {base}")
            w13.append(torch.cat((gate, up), dim=0))
            w2.append(down)
        fused13, fused2 = torch.stack(w13), torch.stack(w2)
        rows.append({"layer": layer, "name": name,
                     "w13_shape": list(fused13.shape), "w2_shape": list(fused2.shape),
                     "w13_sha256": _tensor_hash(fused13), "w2_sha256": _tensor_hash(fused2)})
    return rows


def build_tiny_moe_checkpoint(output: str | os.PathLike[str], seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Create a BF16 safetensors + tokenizer checkpoint, or verify an existing one.

    Only CPU operations and locally installed packages are used.  Repeated calls
    with the same seed reuse the verified files; a different or damaged fixture
    is rejected.  Newly generated files are staged before the final directory is
    created, so an interrupted build cannot masquerade as a finished checkpoint.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    output = Path(output).expanduser().absolute()
    if output.exists():
        return verify_tiny_moe_checkpoint(output, seed)

    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        config = MixtralConfig(**TINY_MIXTRAL_CONFIG)
        config.architectures = ["MixtralForCausalLM"]
        config.dtype = torch.bfloat16
        # fork_rng with an empty device list does not initialize a CUDA context.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            model = MixtralForCausalLM(config).to(dtype=torch.bfloat16).eval()
        state = model.state_dict()
        weights = _legacy_mixtral_state(state)
        save_file(weights, staging / "model.safetensors", metadata={"format": "pt"})
        config.save_pretrained(staging)
        model.generation_config.save_pretrained(staging)
        tokenizer = _build_tokenizer()
        tokenizer.save_pretrained(staging)

        # Validate the actual on-disk checkpoint through the public HF loader,
        # including its legacy-name conversion; no custom state loading here.
        loaded, loading_info = MixtralForCausalLM.from_pretrained(
            staging, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="eager", output_loading_info=True,
        )
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if loading_info.get(field):
                raise RuntimeError(f"HF checkpoint loading {field}: {loading_info[field]}")
        loaded_state = loaded.state_dict()
        if state.keys() != loaded_state.keys() or any(not torch.equal(value, loaded_state[name]) for name, value in state.items()):
            raise RuntimeError("On-disk HF checkpoint did not preserve every BF16 parameter")
        loaded_tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True)
        encoded_prompts = [loaded_tokenizer.encode(prompt) for prompt in SAMPLE_PROMPTS]
        if any(loaded_tokenizer.unk_token_id in ids for ids in encoded_prompts):
            raise RuntimeError("Fixture tokenizer does not cover its sample prompts")

        (staging / "README.md").write_text(
            "# Tiny Mixtral correctness fixture\n\n"
            "Randomly initialized locally; **not a pretrained language model**. "
            "Generation tests validate inference mechanics, not language quality.\n\n"
            "Four real Mixtral decoder layers, four experts per layer, top-2 routing, "
            "128 hidden/intermediate dimensions, four attention heads, two KV heads, "
            "256 vocabulary entries, 256-token context; BF16 weights.\n\n"
            "`model.safetensors` uses standard per-expert Mixtral `w1/w2/w3` keys "
            "for vLLM. Transformers 5 automatically fuses them on load. "
            "See `fixture_manifest.json` for hashes, exact dimensions and sample prompts.\n"
        )
        manifest = {
            "fixture_version": FIXTURE_VERSION,
            "purpose": "randomly_initialized_inference_correctness_fixture_not_pretrained",
            "architecture": "MixtralForCausalLM",
            "seed": seed,
            "config": TINY_MIXTRAL_CONFIG.copy(),
            "dtype": "bfloat16",
            "checkpoint_layout": "hf_mixtral_per_expert_w1_w2_w3",
            "shards": ["model.safetensors"],
            "parameters": sum(tensor.numel() for tensor in weights.values()),
            "parameter_bytes": sum(tensor.numel() * tensor.element_size() for tensor in weights.values()),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "tokenizers")},
            "hf_load_validation": {"all_parameters_bitwise_equal": True, "missing_keys": [], "unexpected_keys": []},
            "sample_prompts": [{"text": text, "token_ids": ids} for text, ids in zip(SAMPLE_PROMPTS, encoded_prompts)],
            "tensors": {name: {"shape": list(tensor.shape), "dtype": "bfloat16", "sha256": _tensor_hash(tensor), "shard": "model.safetensors"} for name, tensor in weights.items()},
            "hf_native_tensors": {name: {"shape": list(tensor.shape), "sha256": _tensor_hash(tensor)} for name, tensor in sorted(state.items())},
            "files": {file.name: {"bytes": file.stat().st_size, "sha256": _sha256(file)} for file in sorted(staging.iterdir()) if file.is_file()},
        }
        (staging / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # mkdir provides no-clobber behavior even if a competing builder finishes
        # while this one is validating. A mismatched existing target is untouched.
        output.mkdir(exist_ok=False)
        for file in sorted(staging.iterdir(), key=lambda file: file.name == "fixture_manifest.json"):
            file.rename(output / file.name)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="models/tiny_mixtral")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = build_tiny_moe_checkpoint(args.output, args.seed)
    print(json.dumps({"path": str(Path(args.output).absolute()), "parameters": result["parameters"],
                      "parameter_bytes": result["parameter_bytes"], "seed": result["seed"],
                      "purpose": result["purpose"]}, indent=2))
