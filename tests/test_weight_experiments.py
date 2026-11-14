"""Independent CPU codec oracles, checkpoint integrity and opt-in GPU timing."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from fluxmoe.codec import compress_bf16, decompress_bf16, _container_crc
from fluxmoe.weight_experiments import groups, included, load_manifest, safe_name, select_layers, sample_statistics, summarize_cases, main_model_groups, fetch, MODELS, merge, stream_layers, save, sha256, verified_summary

HAS_DEPS=all(importlib.util.find_spec(n) for n in ["numpy","torch","safetensors"])

@unittest.skipUnless(HAS_DEPS,"optional real-weight preparation dependencies")
class FastCodecTests(unittest.TestCase):
    def test_every_bf16_pattern_matches_independent_python_encoder_decoder(self):
        from fluxmoe.fast_codec import compress, prepare
        from fluxmoe.cuda_codec import _prepare
        raw=b"".join(struct.pack("<H",i) for i in range(65536))
        encoded=compress(raw)
        self.assertEqual(encoded,compress_bf16(raw))
        self.assertEqual(decompress_bf16(encoded),raw)
        for chunk in [1,257,65536]:
            actual=prepare(encoded,chunk)
            expected=_prepare(encoded,chunk)
            self.assertEqual(actual,expected)

    def test_random_empty_singleton_and_corruption(self):
        from fluxmoe.fast_codec import compress,prepare
        rng=random.Random(95)
        for raw in [b"",b"\0\0"*4097,bytes(rng.randrange(256) for _ in range(8192))]:
            encoded=compress(raw)
            self.assertEqual(encoded,compress_bf16(raw))
            self.assertEqual(prepare(encoded)["reference"],raw)
            damaged=bytearray(encoded);damaged[-1]^=1
            with self.assertRaises(ValueError):prepare(damaged)
        with self.assertRaises(ValueError):compress(b"x")
        with self.assertRaises(ValueError):prepare(b"x")

    def test_valid_container_crc_cannot_hide_wrong_raw_crc(self):
        from fluxmoe.fast_codec import compress,prepare
        encoded=bytearray(compress(b"\0\0"*64))
        encoded[288]^=1
        struct.pack_into("<I",encoded,28,_container_crc(encoded))
        with self.assertRaisesRegex(ValueError,"CRC"):prepare(encoded)

class ExperimentProtocolTests(unittest.TestCase):
    def test_expert_mapping_requires_complete_triplets(self):
        names={f"model.layers.20.mlp.experts.{e}.{p}_proj.weight":"part.safetensors"
               for e in range(4) for p in ["gate","up","down"]}
        result=groups(names)
        self.assertEqual(len(result[20]),4)
        del names["model.layers.20.mlp.experts.2.up_proj.weight"]
        with self.assertRaises(ValueError):groups(names)

    def test_auxiliary_mtp_experts_are_excluded_using_checkpoint_config(self):
        names={f"model.layers.{layer}.mlp.experts.0.{p}_proj.weight":"part.safetensors"
               for layer in [3,91,92] for p in ["gate","up","down"]}
        main,excluded=main_model_groups(names,{"num_hidden_layers":92})
        self.assertEqual(sorted(main),[3,91]);self.assertEqual(excluded,[92])

    def test_mixtral_placement_has_exact_875_percent_bf16_volume(self):
        # Fused gate/up is twice the volume of down. Both EP shards together
        # retain 21/24 matrix units with the paper's 12.5% host placement.
        retained=0
        for rank in range(2):
            for expert in range(rank*4,(rank+1)*4):
                for projection,size in [("gate_up",2),("down",1)]:
                    retained+=size*included("mixtral",rank,expert,rank*4,projection)
        self.assertEqual(retained,21)
        self.assertTrue(included("qwen",0,0,0,"gate_up"))
        self.assertEqual(select_layers("20,0,20",[0,20]),[0,20])
        with self.assertRaises(ValueError):select_layers("5",[0,20])
        for name in ["../x","/tmp/x","a/../../x","a\\b"]:
            with self.assertRaises(ValueError):safe_name(name)

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"optional NumPy")
    def test_statistics_use_uncompressed_decimal_gb_and_real_samples(self):
        stats=sample_statistics([2,4],8_000_000)
        self.assertEqual(stats["avg_latency_ms"],3)
        self.assertEqual(stats["throughput_gb_s"],3)
        self.assertAlmostEqual(stats["p99_latency_ms"],3.98)
        with self.assertRaises(ValueError):sample_statistics([0],100)

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"optional NumPy")
    def test_raw_summary_rejects_missing_duplicate_or_tampered_cases(self):
        meta={"model":"mixtral","selected_layers":[0]}
        cases=[{"status":"passed","layer":0,"logical_rank":r,"raw_bytes":1000,
                "container_bytes":700,"device_source_bytes":730,"samples_ms":[2,4],
                "statistics":sample_statistics([2,4],1000),"bitwise_equal_before_and_after":True}
               for r in range(2)]
        self.assertEqual(summarize_cases(cases,meta)["cases"],2)
        roundoff=copy.deepcopy(cases);roundoff[0]["statistics"]["avg_latency_ms"]+=1e-14
        self.assertEqual(summarize_cases(roundoff,meta)["cases"],2)
        for bad in [cases[:1],cases+[cases[0]]]:
            with self.assertRaises(ValueError):summarize_cases(bad,meta)
        damaged=copy.deepcopy(cases);damaged[0]["samples_ms"]=[1,1]
        with self.assertRaises(ValueError):summarize_cases(damaged,meta)

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"optional NumPy")
    def test_merge_checks_raw_hashes_protocol_and_disjoint_layer_grids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first=saved_benchmark(root/"first",0)
            second=saved_benchmark(root/"second",1)
            result=merge([first,second],root/"combined")
            self.assertEqual(result["cases"],4)
            self.assertEqual(verified_summary(root/"combined/summary.json")[0],result_without_hashes(result))
            self.assertEqual(result["metadata"]["coverage"],"partial_layers_pilot")
            with self.assertRaisesRegex(ValueError,"duplicate"):
                merge([first,first],root/"duplicate")
            other=saved_benchmark(root/"other-device",1,gpu="different GPU")
            with self.assertRaisesRegex(ValueError,"identical model"):
                merge([first,other],root/"wrong-device")
            other=saved_benchmark(root/"other-protocol",1,warmup=3)
            with self.assertRaisesRegex(ValueError,"sample protocol"):
                merge([first,other],root/"wrong-protocol")
            raw=first.parent/"cases.jsonl"
            raw.write_text(raw.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"hashes"):
                merge([first,second],root/"corrupted")

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"optional NumPy")
    def test_layer_stream_enforces_caps_and_removes_only_owned_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";data.mkdir()
            user_file=data/"existing-user-checkpoint";user_file.write_text("preserve me")
            plan=stream_fixture_plan(root)
            staged=[]
            def fake_fetch(plan_path,directory,max_download_gb):
                directory.mkdir();staged.append(directory)
                save(directory/"download_manifest.json",json.loads(Path(plan_path).read_text()))
            def fake_benchmark(manifest,output,warmup,samples,chunk):
                d=json.loads(Path(manifest).read_text())
                saved_benchmark(output,d["selected_layers"][0],warmup=warmup)
            with patch("fluxmoe.weight_experiments.fetch",side_effect=fake_fetch) as download, \
                 patch("fluxmoe.weight_experiments.benchmark",side_effect=fake_benchmark):
                for per_layer,total in [(1e-9,1),(1,1e-9)]:
                    with self.assertRaisesRegex(ValueError,"cap exceeded"):
                        stream_layers(plan,data,root/"over-cap",per_layer,total,2,2,256)
                download.assert_not_called()
                stream_layers(plan,data,root/"results",1,1,2,2,256)
            self.assertEqual(len(staged),2)
            self.assertTrue(all(not p.parent.exists() for p in staged))
            self.assertEqual(user_file.read_text(),"preserve me")
            self.assertEqual(list(data.iterdir()),[user_file])
            self.assertEqual(verified_summary(root/"results/combined/summary.json")[0]["cases"],4)
            self.assertTrue((root/"results/layer-0/download_manifest.json").is_file())

    def test_layer_stream_failure_retains_manifest_and_failed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";data.mkdir();plan=stream_fixture_plan(root)
            def fake_fetch(plan_path,directory,max_download_gb):
                directory.mkdir()
                save(directory/"download_manifest.json",json.loads(Path(plan_path).read_text()))
            def fail_benchmark(manifest,output,*args):
                save(output/"summary.json",{"status":"failed","error":"intentional test failure"})
                raise RuntimeError("intentional test failure")
            with patch("fluxmoe.weight_experiments.fetch",side_effect=fake_fetch), \
                 patch("fluxmoe.weight_experiments.benchmark",side_effect=fail_benchmark):
                with self.assertRaisesRegex(RuntimeError,"intentional"):
                    stream_layers(plan,data,root/"results",1,1,2,2,256)
            self.assertEqual(list(data.iterdir()),[])
            self.assertTrue((root/"results/layer-0/download_manifest.json").is_file())
            self.assertEqual(json.loads((root/"results/layer-0/benchmark/summary.json").read_text())["status"],"failed")

def result_without_hashes(result):
    return {k:v for k,v in result.items() if k not in {"raw_cases_sha256","metadata_sha256"}}

def saved_benchmark(directory,layer,gpu="unit-test-device",warmup=2):
    """Persist synthetic timing records only for summary/merge integrity tests."""
    directory=Path(directory);directory.mkdir()
    meta={"model":"mixtral","selected_layers":[layer],"available_layers":[0,1],
          "gpu":gpu,"weight_origin":"unit_test_fixture","chunk_size":256}
    cases=[{"status":"passed","layer":layer,"logical_rank":r,"raw_bytes":1000,
            "container_bytes":700,"device_source_bytes":730,"samples_ms":[2,4],
            "statistics":sample_statistics([2,4],1000),"bitwise_equal_before_and_after":True,
            "warmup":warmup} for r in range(2)]
    save(directory/"metadata.json",meta)
    (directory/"cases.jsonl").write_text("".join(json.dumps(c)+"\n" for c in cases))
    result=summarize_cases(cases,meta)
    result.update(raw_cases_sha256=sha256(directory/"cases.jsonl"),metadata_sha256=sha256(directory/"metadata.json"))
    save(directory/"summary.json",result)
    return directory/"summary.json"

def stream_fixture_plan(root):
    mapping={f"model.layers.{layer}.mlp.experts.0.{p}_proj.weight":f"part-{layer}.safetensors"
             for layer in [0,1] for p in ["gate","up","down"]}
    plan={"model":"mixtral","selected_layers":[0,1],"available_layers":[0,1],"weight_map":mapping,
          "shards":[{"name":f"part-{l}.safetensors","bytes":10,"expected_sha256":"0"*64} for l in [0,1]]}
    path=root/"plan.json";save(path,plan);return path

def fixture(directory):
    """Explicitly label copied random weights so they cannot be real results."""
    from safetensors import safe_open
    source=Path(__file__).resolve().parents[1]/"models/tiny_mixtral/model.safetensors"
    target=directory/"model.safetensors";shutil.copy2(source,target)
    with safe_open(target,framework="pt") as f:
        weight_map={k:target.name for k in f.keys() if ".experts." in k}
    (directory/"model.safetensors.index.json").write_text(json.dumps({"weight_map":weight_map}))
    files={p.name:{"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size}
           for p in directory.iterdir() if p.is_file()}
    d={"status":"downloaded_verified","model":"mixtral","repo_id":"unit_test_fixture",
       "revision":"unit_test_fixture_not_pretrained","origin":"unit_test_fixture",
       "selected_layers":[0],"available_layers":[0,1,2,3],"files":files,
       "weight_map":{k:v for k,v in weight_map.items() if ".layers.0." in k}}
    path=directory/"download_manifest.json";path.write_text(json.dumps(d));return path

@unittest.skipUnless(HAS_DEPS,"optional checkpoint dependencies")
class WeightIntegrityTests(unittest.TestCase):
    def test_download_cap_and_shard_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";source.mkdir();m=fixture(source)
            d=json.loads(m.read_text());(source/"config.json").write_text('{}')
            tensor=source/"model.safetensors"
            plan=dict(d,repo_id=MODELS['mixtral']['repo'],revision='a'*40,
                      required_download_bytes=tensor.stat().st_size,
                      shards=[{'name':tensor.name,'bytes':tensor.stat().st_size,'expected_sha256':hashlib.sha256(tensor.read_bytes()).hexdigest()}])
            plan_path=root/"plan.json";plan_path.write_text(json.dumps(plan))
            def fake_download(repo,name,revision,local_dir):
                path=Path(local_dir)/name;shutil.copy2(source/name,path);return str(path)
            with patch('huggingface_hub.hf_hub_download',side_effect=fake_download) as download:
                with self.assertRaises(ValueError):fetch(plan_path,root/"cap",1e-8)
                download.assert_not_called()
                fetch(plan_path,root/"good",1)
                self.assertEqual(load_manifest(root/"good/download_manifest.json")[0]['status'],'downloaded_verified')
                broken=copy.deepcopy(plan);broken['shards'][0]['expected_sha256']='0'*64
                plan_path.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError,'SHA256'):fetch(plan_path,root/"bad",1)
                self.assertEqual(json.loads((root/"bad/download_manifest.json").read_text())['status'],'downloading')

    def test_selected_tensor_must_match_hashed_index(self):
        from fluxmoe.weight_experiments import analyze
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);source=d/"source";source.mkdir();manifest=fixture(source)
            analyze(manifest,"0",d/"analysis")
            r=json.loads((d/"analysis/figure11.json").read_text())
            self.assertEqual(r["weight_origin"],"unit_test_fixture")
            layer=r["layers"][0]
            self.assertEqual(sum(layer["exponent_counts"]),sum(layer["mantissa_counts"]))
            self.assertEqual(len(layer["tensors"]),12)
            bad=json.loads(manifest.read_text())
            name=next(iter(bad["weight_map"]));bad["weight_map"][name]="unverified.safetensors"
            manifest.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load_manifest(manifest)

@unittest.skipUnless(HAS_DEPS and os.environ.get("FLUXMOE_TEST_CUDA")=="1","opt-in native preparation + real CUDA timing fixture")
class WeightCudaTests(unittest.TestCase):
    def test_native_prepared_sources_two_stream_timing_and_exact_output(self):
        from fluxmoe.weight_experiments import benchmark
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);source=d/"source";source.mkdir();manifest=fixture(source)
            benchmark(manifest,d/"benchmark",warmup=2,samples=5,chunk_size=256)
            result=json.loads((d/"benchmark/summary.json").read_text())
            self.assertEqual(result["status"],"passed")
            self.assertEqual(result["metadata"]["weight_origin"],"unit_test_fixture")
            self.assertEqual(result["metadata"]["coverage"],"partial_layers_pilot")
            cases=[json.loads(s) for s in (d/"benchmark/cases.jsonl").read_text().splitlines()]
            self.assertEqual(len(cases),2)
            self.assertTrue(all(c["bitwise_equal_before_and_after"] for c in cases))
            self.assertTrue(all(len(c["samples_ms"])==5 for c in cases))

if __name__=="__main__":unittest.main()
