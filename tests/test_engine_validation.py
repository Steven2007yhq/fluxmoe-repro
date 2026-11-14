"""Pure-data portability and historical-artifact provenance checks."""
import copy
from pathlib import Path
import tempfile
import unittest

from fluxmoe.engine_validation import compare_execution_environments, _write_report


def environment_fixture():
    return {
        "execution_environment_schema_version": 1,
        "hostname": "test-server", "gpu": "NVIDIA A40", "gpu_compute_capability": [8, 6],
        "torch_cuda": "13.0",
        "versions": {"python": "3.12.12", "numpy": "2.3.5", "triton": "3.7.0",
                     "vllm": "0.29.0", "torch": "2.13.0+cu130", "transformers": "5.5.0"},
        "process_environment": {"CUDA_VISIBLE_DEVICES": "2", "CUDA_DEVICE_ORDER": None,
                                "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    }


class ExecutionEnvironmentTests(unittest.TestCase):
    def test_a40_python312_pair_is_valid_without_local_machine_assumptions(self):
        result = compare_execution_environments(environment_fixture(), environment_fixture())
        self.assertEqual(result["status"], "passed", result)

    def test_explicitly_unset_cuda_visibility_is_valid_evidence(self):
        reference = environment_fixture()
        reference["process_environment"]["CUDA_VISIBLE_DEVICES"] = None
        result = compare_execution_environments(reference, copy.deepcopy(reference))
        self.assertEqual(result["status"], "passed", result)

    def test_cross_platform_or_device_pair_fails(self):
        for field, value in (("gpu", "NVIDIA GeForce RTX 4060 Laptop GPU"),
                             ("gpu_compute_capability", [8, 9]), ("hostname", "other-server"),
                             ("torch_cuda", "12.6")):
            with self.subTest(field=field):
                reference, paged = environment_fixture(), environment_fixture()
                paged[field] = value
                result = compare_execution_environments(reference, paged)
                self.assertEqual(result["status"], "failed")
                self.assertTrue(any(field in error for error in result["errors"]))
        reference, paged = environment_fixture(), environment_fixture()
        paged["process_environment"]["CUDA_VISIBLE_DEVICES"] = "3"
        self.assertEqual(compare_execution_environments(reference, paged)["status"], "failed")

    def test_python_numpy_triton_and_framework_mismatch_fail(self):
        for package in environment_fixture()["versions"]:
            with self.subTest(package=package):
                reference, paged = environment_fixture(), environment_fixture()
                paged["versions"][package] = "different-version"
                result = compare_execution_environments(reference, paged)
                self.assertEqual(result["status"], "failed")
                self.assertTrue(any(package in error for error in result["errors"]))

    def test_legacy_missing_provenance_is_incomplete_not_numerical_failure(self):
        legacy = environment_fixture()
        del legacy["execution_environment_schema_version"]
        del legacy["hostname"], legacy["gpu_compute_capability"]
        for package in ("python", "numpy", "triton"):
            del legacy["versions"][package]
        del legacy["process_environment"]["CUDA_VISIBLE_DEVICES"]
        original = copy.deepcopy(legacy)
        result = compare_execution_environments(legacy, copy.deepcopy(legacy))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["missing_evidence"])
        self.assertEqual(legacy, original)
        # Mixing a legacy artifact with a newer artifact also preserves the
        # distinction between missing fields and conflicting known fields.
        result = compare_execution_environments(legacy, environment_fixture())
        self.assertEqual(result["status"], "incomplete")

    def test_known_conflict_still_fails_when_other_evidence_is_missing(self):
        reference, paged = environment_fixture(), environment_fixture()
        del reference["hostname"], paged["hostname"]
        paged["versions"]["torch"] = "different-version"
        self.assertEqual(compare_execution_environments(reference, paged)["status"], "failed")

    def test_report_uses_actual_gpu_versions_and_explains_legacy_status(self):
        report = {
            "status": "incomplete", "compared_requests": 11, "compared_generated_tokens": 176,
            "compared_logprobs": 176, "compared_prompt_logprobs": 200,
            "max_logprob_abs_error": 0, "max_prompt_logprob_abs_error": 0,
            "uninstrumented_checks": {}, "pager_counters": {}, "errors": [],
            "incomplete_reasons": ["resident: missing execution environment hostname"],
        }
        with tempfile.TemporaryDirectory() as directory:
            _write_report(Path(directory), environment_fixture(), environment_fixture(), report)
            text = (Path(directory) / "report.md").read_text()
        self.assertIn("NVIDIA A40", text)
        self.assertIn("3.12.12 / 2.3.5 / 3.7.0", text)
        self.assertNotIn("RTX 4060", text)
        self.assertIn("旧实验的原始数值结论不因此改判失败", text)
        self.assertIn("新的输出目录", text)


if __name__ == "__main__":
    unittest.main()
