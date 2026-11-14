import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from fluxmoe import host_kv_execution as execution


class HostKVExecutionTests(unittest.TestCase):
    def setUp(self):
        contexts = contextlib.ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.temporary = Path(contexts.enter_context(tempfile.TemporaryDirectory()))
        self.config = self.temporary / "config.json"
        self.work = self.temporary / "work"
        self.archive = self.temporary / "evidence.tar.gz"
        self.gpus = [f"GPU-{i:08x}-0000-0000-0000-000000000000" for i in range(4)]
        self.config.write_text(json.dumps(dict(gpu_uuids=self.gpus, owned_rss_cap_bytes=160 * execution.GIB,
                                               work_cap_bytes=16 * execution.GIB,
                                               host_kv_bytes=18 * execution.GIB)))
        contexts.enter_context(patch.object(execution, "_lock_directory", return_value=self.temporary / "lease"))
        self.gpu_query = contexts.enter_context(patch.object(execution, "_query_gpu"))

    def run_fixture(self, code, timeout=5):
        command = [sys.executable, "-c", code, str(self.work)]
        with patch.object(execution, "_qualification_command", return_value=command):
            return execution.run(self.config, self.work, timeout)

    def test_success_export_and_verified_recovery(self):
        code = (
            "import hashlib, json, os, pathlib, sys\n"
            "p = pathlib.Path(sys.argv[1])\n"
            "env = os.environ['CUDA_VISIBLE_DEVICES'].encode()\n"
            "(p / 'env.txt').write_bytes(env)\n"
            "config_hash = hashlib.sha256((p / 'config.json').read_bytes()).hexdigest()\n"
            "(p / 'result.json').write_text(json.dumps(dict("
            f"status={execution.PASS!r}, formal_performance_allowed=False, "
            "config_sha256=config_hash, evidence={'env.txt': hashlib.sha256(env).hexdigest()})))\n"
        )
        self.assertEqual(self.run_fixture(code), 0)
        self.assertEqual([call.args[1] for call in self.gpu_query.call_args_list], ["before", "after"])
        self.assertEqual(execution.status(self.work)["observed_status"], "success")
        receipt = execution.export(self.work, self.archive)
        output = self.temporary / "recovered"
        result = execution.recover(self.archive, Path(receipt["receipt"]), output, receipt["receipt_sha256"])
        self.assertTrue(result["transport_only"])
        self.assertEqual((output / "evidence" / "env.txt").read_text(), ",".join(self.gpus))
        self.assertEqual((output / "evidence" / "config.json").read_bytes(), self.config.read_bytes())
        with self.assertRaises(FileExistsError):
            self.run_fixture(code)
        self.assertEqual(execution.status(self.work)["observed_status"], "success")

    def test_zero_exit_without_pass_marker_stays_failed_and_exportable(self):
        self.assertEqual(self.run_fixture("print('no qualification result')"), 1)
        self.assertEqual(execution.status(self.work)["observed_status"], "failed")
        receipt = execution.export(self.work, self.archive)
        self.assertEqual(receipt["attempt_status"], "failed")
        with self.archive.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "archive size or SHA-256"):
            execution.recover(self.archive, Path(receipt["receipt"]), self.temporary / "recovered")

    def test_busy_uid_lease_records_admission_failure_without_launching(self):
        with execution._take_gpu_lock(execution._boot_id()):
            self.assertEqual(self.run_fixture("raise AssertionError('launched despite held lease')"), 1)
        state = execution.status(self.work)
        self.assertEqual(state["observed_status"], "failed")
        self.assertIn("holds the GPU lock", state["reason"])
        self.assertFalse((self.work / "control" / "worker.log").exists())
        self.gpu_query.assert_not_called()

    def test_timeout_closes_descendant_group_and_preserves_output(self):
        code = (
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(p.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        self.assertEqual(self.run_fixture(code, timeout=1), 124)
        state = execution.status(self.work)
        self.assertEqual(state["observed_status"], "timeout")
        self.assertEqual(state["live_worker_group"], [])
        self.assertGreaterEqual(len(state["worker_group_before_cleanup"]), 2)
        self.assertTrue((self.work / "control" / "worker.log").read_text().strip().isdigit())
        self.assertEqual(execution.export(self.work, self.archive)["attempt_status"], "timeout")

    def test_active_worker_group_blocks_export(self):
        boot = execution._boot_id()
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              start_new_session=True) as process:
            try:
                control = self.work / "control"
                control.mkdir(parents=True)
                execution._publish_json(control / "execution.json",
                                        dict(schema=execution.SCHEMA, status="running", boot_id=boot,
                                             supervisor=None, worker=execution._process(process.pid)))
                with self.assertRaisesRegex(RuntimeError, "still active"):
                    execution.export(self.work, self.archive)
            finally:
                process.terminate()
        self.assertEqual(execution.status(self.work)["observed_status"], "interrupted")
        self.assertEqual(execution.export(self.work, self.archive)["attempt_status"], "interrupted")

    def test_verified_transport_rejects_archive_traversal(self):
        with tarfile.open(self.archive, "x:gz") as tar:
            member = tarfile.TarInfo("evidence/../outside.txt")
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
        raw = self.archive.read_bytes()
        receipt = self.temporary / "receipt.json"
        receipt.write_text(json.dumps(dict(schema=execution.SCHEMA, status="exported", transport_only=True,
                                          attempt_status="failed", formal_performance_allowed=False,
                                          archive_bytes=len(raw), archive_sha256=hashlib.sha256(raw).hexdigest())))
        output = self.temporary / "recovered"
        with self.assertRaisesRegex(ValueError, "unexpected archive member"):
            execution.recover(self.archive, receipt, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
