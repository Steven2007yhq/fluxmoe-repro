"""Run and recover a single four-GPU host KV qualification attempt."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid


SCHEMA = "fluxmoe-host-kv-evidence-v1"
PASS = "HOST_KV_LONG_QUALIFICATION_PASSED"
GIB = 2**30
MAX_RAW_BYTES = 16 * GIB
MAX_EXPORT_BYTES = MAX_RAW_BYTES + 64 * 2**20
MAX_FILES = 100_000
MAX_JSON_BYTES = 32 * 2**20
PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
SHA256 = re.compile(r"[0-9a-f]{64}")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
TERMINAL_STATES = {"success", "failed", "timeout", "interrupted"}
WORKER_BOOTSTRAP = """
import os
import sys

gate = int(sys.argv[1])
ready = os.read(gate, 1)
os.close(gate)
if ready != b"1":
    raise SystemExit("supervisor did not commit the worker identity")
os.execv(sys.argv[2], sys.argv[2:])
"""


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _json_file(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds evidence size limit: {path}")
    return json.loads(raw)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _durable_bytes(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def _publish_new_file(staging, destination):
    # Reserve the name first, including on the workspace's exFAT filesystem.
    with destination.open("xb"):
        pass
    try:
        os.replace(staging, destination)
    except OSError:
        destination.unlink()
        raise


def _boot_id():
    return BOOT_ID.read_text().strip()


def _process(pid):
    try:
        # The command name can contain spaces and parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return dict(pid=pid, state=fields[0], pgid=int(fields[2]), sid=int(fields[3]),
                start_ticks=int(fields[19]), rss_bytes=max(0, int(fields[21])) * PAGE_BYTES)


def _live_owner(owner, boot):
    if not owner or boot != _boot_id():
        return False
    observed = _process(owner["pid"])
    return bool(observed and observed["state"] not in {"Z", "X"}
                and observed["start_ticks"] == owner["start_ticks"])


def _group_members(leader, boot):
    if not leader or boot != _boot_id():
        return []
    current = _process(leader["pid"])
    if current and current["start_ticks"] != leader["start_ticks"]:
        return []
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            item = _process(int(path.name))
            if (item and item["state"] not in {"Z", "X"} and
                    item["pgid"] == leader["pgid"] and item["sid"] == leader["sid"] and
                    item["start_ticks"] >= leader["start_ticks"]):
                members.append(item)
    return sorted(members, key=lambda item: item["pid"])


def _lock_directory():
    return Path("/tmp") / f"fluxmoe-host-kv-{os.getuid()}"


def _take_gpu_lock(boot):
    directory = _lock_directory()
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"GPU lock directory is not private to this UID: {directory}")
    fd = os.open(directory / "gpu.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    lock = os.fdopen(fd, "r+")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another host KV qualification holds the GPU lock for this UID") from error
        previous = directory / "worker.json"
        if previous.exists():
            recorded = _json_file(previous)
            if recorded["boot_id"] == boot and _group_members(recorded["worker"], boot):
                raise RuntimeError("the previous qualification still has a live worker group")
        return lock
    except BaseException:
        lock.close()
        raise


def _record_gpu_lock(boot, worker, work):
    _publish_json(_lock_directory() / "worker.json",
                  dict(boot_id=boot, worker=worker, work=str(work)))


def _query_gpu(control, label, uuids, limits):
    queries = {
        "identity": ["nvidia-smi", "--id=" + ",".join(uuids),
                     "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,memory.free,driver_version",
                     "--format=csv,noheader,nounits"],
        "compute": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                    "--format=csv,noheader,nounits"],
        "topology": ["nvidia-smi", "topo", "-m"],
    }
    memory = Path("/proc/meminfo").read_text()
    available = next(int(line.split()[1]) * 1024 for line in memory.splitlines()
                     if line.startswith("MemAvailable:"))
    report = dict(gpu_uuids=uuids, unix_time=time.time(), host_memory=memory,
                  host_available_bytes=available, shared_memory_free_bytes=shutil.disk_usage("/dev/shm").free,
                  work_filesystem_free_bytes=shutil.disk_usage(control).free, limits=limits, queries={})
    for name, command in queries.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            value = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            value = dict(returncode=None, error=str(error))
        report["queries"][name] = dict(argv=command, **value)
    if all(report["queries"][name]["returncode"] == 0 for name in ("identity", "compute")):
        identity = [list(map(str.strip, row)) for row in
                    csv.reader(io.StringIO(report["queries"]["identity"]["stdout"])) if row]
        compute = [list(map(str.strip, row)) for row in
                   csv.reader(io.StringIO(report["queries"]["compute"]["stdout"])) if row]
        report["gpu_rows"] = identity
        report["selected_compute_rows"] = [row for row in compute if row and row[0] in uuids]
        report["identity_matched"] = len(identity) == 4 and all(len(row) == 8 for row in identity)
        report["identity_matched"] = report["identity_matched"] and {row[1] for row in identity} == set(uuids)
        report["selected_devices_idle"] = (not report["selected_compute_rows"] and
                                          all(len(row) == 3 and row[1].isdigit() for row in compute))
        report["gpu_memory_admissible"] = report["identity_matched"] and all(
            row[5].isdigit() and row[6].isdigit() and int(row[5]) <= 512 and int(row[6]) >= 44_000
            for row in identity
        )
    _publish_json(control / f"gpu-{label}.json", report)
    if not report.get("identity_matched"):
        raise RuntimeError(f"could not verify the four selected GPUs; see gpu-{label}.json")
    if not report["selected_devices_idle"]:
        raise RuntimeError(f"selected GPU compute processes remain; see gpu-{label}.json")
    if label == "before":
        if not report["gpu_memory_admissible"]:
            raise RuntimeError("each selected GPU requires at most 512 MiB used and at least 44,000 MiB free")
        if available < max(limits["owned_rss_cap_bytes"], 192 * GIB):
            raise RuntimeError("MemAvailable is below 192 GiB or the aggregate RSS cap")
        if report["shared_memory_free_bytes"] < limits["host_kv_bytes"]:
            raise RuntimeError("/dev/shm cannot accommodate the configured native host KV pool")
        if report["work_filesystem_free_bytes"] < max(limits["work_cap_bytes"], 32 * GIB):
            raise RuntimeError("work filesystem requires 32 GiB free for evidence and archive")


def _work_size(work):
    total = 0
    for directory, _subdirs, names in os.walk(work, followlinks=False):
        for name in names:
            try:
                info = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _source_snapshot(control):
    source_root = Path(__file__).resolve().parent
    snapshot = control / "source" / "fluxmoe"
    digests = {}
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        _durable_bytes(destination, data)
        digests[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    distributions = {}
    for name in ("torch", "vllm", "numpy", "safetensors", "transformers"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    _publish_json(control / "environment.json",
                  dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       node=platform.node(), cwd=os.getcwd(), distributions=distributions,
                       source_sha256=digests))


def _wait_for_worker(process, deadline, get_interrupt, work, state, limits):
    next_sample = 0
    worker = state["worker"]
    path = work / "control" / "resources.jsonl"
    while process.poll() is None:
        if get_interrupt():
            return "interrupted", f"supervisor received signal {get_interrupt()}"
        now = time.monotonic()
        if now >= deadline:
            return "timeout", "worker deadline elapsed"
        if now >= next_sample:
            members = _group_members(worker, state["boot_id"])
            supervisor = _process(os.getpid())
            rss = supervisor["rss_bytes"] + sum(member["rss_bytes"] for member in members)
            work_bytes = _work_size(work)
            state["sampled_peak_rss_bytes"] = max(rss, state.get("sampled_peak_rss_bytes", 0))
            state["sampled_peak_work_bytes"] = max(work_bytes, state.get("sampled_peak_work_bytes", 0))
            row = dict(monotonic=now, unix_time=time.time(), processes=members, supervisor=supervisor,
                       aggregate_rss_bytes=rss, work_bytes=work_bytes)
            with path.open("ab") as stream:
                stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            if rss > limits["owned_rss_cap_bytes"]:
                return "failed", "sampled aggregate RSS exceeds the configured cap"
            if work_bytes > limits["work_cap_bytes"]:
                return "failed", "sampled work directory size exceeds the configured cap"
            next_sample = now + 2
        time.sleep(0.2)
    if get_interrupt():
        return "interrupted", f"supervisor received signal {get_interrupt()}"
    if process.returncode:
        return "failed", f"qualification process returned {process.returncode}"
    return "success", None


def _close_group(process, worker, boot):
    if worker is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return [], []
    initial = _group_members(worker, boot)
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not _group_members(worker, boot):
            break
        try:
            os.killpg(worker["pgid"], sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            process.poll()
            if not _group_members(worker, boot):
                break
            time.sleep(0.2)
    process.poll()
    return initial, _group_members(worker, boot)


def _qualification_command(config, work):
    return [sys.executable, "-B", "-m", "fluxmoe.host_kv_qualification",
            "--config", str(config), "--work", str(work)]


def _read_pass_result(work, config_sha):
    result = work / "result.json"
    if not stat.S_ISREG(result.lstat().st_mode):
        raise ValueError("qualification result is not a regular file")
    record = _json_file(result)
    if (not isinstance(record, dict) or record.get("status") != PASS or
            record.get("formal_performance_allowed") is not False or record.get("config_sha256") != config_sha):
        raise ValueError("qualification result did not pass with the frozen configuration")
    evidence = record.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("qualification result lacks evidence bindings")
    for name, expected in evidence.items():
        member = _safe_member("evidence/" + name)
        path = work.joinpath(*member.parts[1:])
        if path == result or path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"invalid qualification evidence file: {name}")
        with path.open("rb") as stream:
            if _digest_stream(stream)[1] != expected:
                raise ValueError(f"qualification evidence hash differs: {name}")
    with result.open("rb") as stream:
        return _digest_stream(stream)[1]


def run(config, work, timeout):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    work = work.resolve()
    work.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(mode=0o700)
    control = work / "control"
    control.mkdir()
    state_file = control / "execution.json"
    boot = _boot_id()
    state = dict(schema=SCHEMA, status="running", phase="prepare", boot_id=boot, attempt_id=uuid.uuid4().hex,
                 supervisor=_process(os.getpid()), work=str(work), started_unix=time.time(),
                 timeout_seconds=timeout, formal_performance_allowed=False)
    _publish_json(state_file, state)
    caught = None

    def handle(signum, _frame):
        nonlocal caught
        if caught is None:
            caught = signum

    previous_handlers = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    lock = process = worker = None
    uuids = limits = None
    preflight_started = False
    outcome, reason = "failed", "qualification did not start"
    try:
        frozen = work / "config.json"
        with config.open("rb") as stream:
            config_data = stream.read(MAX_JSON_BYTES + 1)
        _durable_bytes(frozen, config_data)
        state["config_sha256"] = hashlib.sha256(config_data).hexdigest()
        if len(config_data) > MAX_JSON_BYTES:
            raise ValueError("qualification config exceeds the JSON size limit")
        values = json.loads(config_data)
        uuids = values.get("gpu_uuids") if isinstance(values, dict) else None
        if (not isinstance(uuids, list) or len(uuids) != 4 or
                not all(isinstance(uuid, str) and GPU_UUID.fullmatch(uuid) for uuid in uuids) or
                len({uuid.lower() for uuid in uuids}) != 4):
            raise ValueError("config requires four distinct full GPU UUIDs in rank order")
        limits = {name: values.get(name) for name in ("owned_rss_cap_bytes", "work_cap_bytes", "host_kv_bytes")}
        if (any(type(value) is not int or value <= 0 for value in limits.values()) or
                limits["work_cap_bytes"] > MAX_RAW_BYTES or limits["host_kv_bytes"] >= limits["owned_rss_cap_bytes"]):
            raise ValueError("config requires positive RSS, work and host KV caps within the evidence limits")
        command = _qualification_command(frozen, work)
        source_cwd = control / "source"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(uuids),
                   CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
        state.update(phase="preflight", gpu_uuids=uuids, argv=command, limits=limits, cwd=str(source_cwd),
                     worker_environment={name: env[name] for name in
                                         ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTHONUNBUFFERED")})
        _publish_json(state_file, state)
        _source_snapshot(control)
        lock = _take_gpu_lock(boot)
        preflight_started = True
        _query_gpu(control, "before", uuids, limits)
        if caught:
            raise InterruptedError(f"supervisor received signal {caught}")
        with (control / "worker.log").open("xb", buffering=0) as log:
            started = time.monotonic()
            read_gate, write_gate = os.pipe()
            try:
                bootstrap = [sys.executable, "-B", "-c", WORKER_BOOTSTRAP, str(read_gate)]
                process = subprocess.Popen(bootstrap + command, cwd=source_cwd, env=env,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=(lock.fileno(), read_gate))
                worker = _process(process.pid)
                if not worker or worker["pid"] != worker["pgid"] or worker["pid"] != worker["sid"]:
                    raise RuntimeError("worker did not become the expected session and process group leader")
                _record_gpu_lock(boot, worker, work)
                state.update(phase="qualification", worker=worker)
                _publish_json(state_file, state)
                if not caught:
                    os.write(write_gate, b"1")
            finally:
                os.close(write_gate)
                os.close(read_gate)
            outcome, reason = _wait_for_worker(process, started + timeout, lambda: caught, work, state, limits)
    except (OSError, ValueError, RuntimeError) as error:
        outcome, reason = ("interrupted" if caught else "failed"), str(error)
    finally:
        try:
            state["phase"] = "reap"
            _publish_json(state_file, state)
            if process is not None:
                initial, remaining = _close_group(process, worker, boot)
                state.update(worker_returncode=process.returncode,
                             worker_group_before_cleanup=initial, worker_group_remaining=remaining)
                if outcome == "success" and initial:
                    outcome, reason = "failed", "worker exited with live processes in its group"
                if remaining:
                    state["cleanup_error"] = "owned worker group still has live processes"
                    if outcome == "success":
                        outcome, reason = "failed", state["cleanup_error"]
                with (control / "worker.log").open("rb") as log:
                    os.fsync(log.fileno())
            if preflight_started:
                try:
                    _query_gpu(control, "after", uuids, limits)
                except (OSError, RuntimeError) as error:
                    state["gpu_postflight_error"] = str(error)
                    if outcome == "success":
                        outcome, reason = "failed", str(error)
            if caught and outcome == "success":
                outcome, reason = "interrupted", f"supervisor received signal {caught}"
            if outcome == "success":
                try:
                    if _work_size(work) > limits["work_cap_bytes"]:
                        raise ValueError("final work directory size exceeds the configured cap")
                    state["result_sha256"] = _read_pass_result(work, state["config_sha256"])
                except (OSError, ValueError) as error:
                    outcome, reason = "failed", str(error)
            state.update(status=outcome, phase="finished", reason=reason, finished_unix=time.time())
            _publish_json(state_file, state)
        finally:
            if lock is not None:
                lock.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    print(json.dumps(state))
    return dict(success=0, failed=1, timeout=124, interrupted=128 + (caught or signal.SIGINT))[outcome]


def status(work):
    state = _json_file(work / "control" / "execution.json")
    if state.get("schema") != SCHEMA:
        raise ValueError("unrecognized qualification attempt")
    owner_alive = state["status"] == "running" and _live_owner(state.get("supervisor"), state["boot_id"])
    members = _group_members(state.get("worker"), state["boot_id"])
    observed = state["status"]
    if observed == "running" and not owner_alive and not members:
        observed = "interrupted"
    return dict(**state, observed_status=observed, supervisor_alive=owner_alive, live_worker_group=members)


def _digest_stream(stream, output=None):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        if output is not None:
            output.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


class _HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size):
        raw = self.stream.read(size)
        self.digest.update(raw)
        self.size += len(raw)
        return raw


def _file_signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _safe_member(name):
    path = PurePosixPath(name)
    if (path.as_posix() != name or path.is_absolute() or ".." in path.parts or
            "\\" in name or len(path.parts) < 2 or path.parts[0] != "evidence"):
        raise ValueError(f"unexpected archive member: {name}")
    return path


def _work_files(work):
    paths = []
    total = 0
    for directory, subdirs, files in os.walk(work, followlinks=False):
        base = Path(directory)
        for name in subdirs:
            if not stat.S_ISDIR((base / name).lstat().st_mode):
                raise ValueError(f"non-directory in evidence tree: {base / name}")
        for name in files:
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"non-regular file in evidence tree: {path}")
            paths.append((path, info))
            total += info.st_size
            if len(paths) > MAX_FILES or total > MAX_EXPORT_BYTES:
                raise ValueError("work exceeds evidence size limits")
    return sorted(paths)


def _write_evidence(tar, work, paths):
    inventory = {}
    for path, original in paths:
        name = "evidence/" + path.relative_to(work).as_posix()
        _safe_member(name)
        member = tarfile.TarInfo(name)
        member.size = original.st_size
        member.mode = 0o600
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if _file_signature(os.fstat(fd)) != _file_signature(original):
                raise ValueError(f"evidence file changed: {path}")
            reader = _HashingReader(stream)
            tar.addfile(member, reader)
            if reader.size != member.size or _file_signature(os.fstat(fd)) != _file_signature(original):
                raise ValueError(f"evidence file changed while archiving: {path}")
        inventory[name] = dict(bytes=reader.size, sha256=reader.digest.hexdigest())
    return inventory


def _require_finished(work):
    state = status(work)
    if (state["observed_status"] not in TERMINAL_STATES or state["supervisor_alive"] or
            state["live_worker_group"]):
        raise RuntimeError("the qualification attempt or its worker group is still active")
    return state


def export(work, archive):
    work, archive = work.resolve(), archive.resolve()
    if archive == work or work in archive.parents:
        raise ValueError("archive destination must be outside the attempt directory")
    receipt = archive.with_name(archive.name + ".receipt.json")
    if archive.exists() or receipt.exists():
        raise FileExistsError("archive and receipt require new paths")
    state = _require_finished(work)
    paths = _work_files(work)
    with tempfile.TemporaryDirectory(dir=archive.parent) as temporary:
        staged_archive = Path(temporary) / "archive"
        with tarfile.open(staged_archive, "x:gz", compresslevel=1) as tar:
            inventory = _write_evidence(tar, work, paths)
            manifest = _json_bytes(dict(schema=SCHEMA, attempt_status=state["observed_status"],
                                        formal_performance_allowed=False, files=inventory))
            if len(manifest) > MAX_JSON_BYTES:
                raise ValueError("evidence manifest exceeds size limit")
            member = tarfile.TarInfo("manifest.json")
            member.size = len(manifest)
            member.mode = 0o600
            tar.addfile(member, io.BytesIO(manifest))
        latest = _work_files(work)
        if ([(path, _file_signature(info)) for path, info in latest] !=
                [(path, _file_signature(info)) for path, info in paths] or _require_finished(work) != state):
            raise RuntimeError("work changed during evidence export")
        with staged_archive.open("rb") as stream:
            archive_bytes, archive_sha = _digest_stream(stream)
            os.fsync(stream.fileno())
        value = dict(schema=SCHEMA, status="exported", transport_only=True,
                     attempt_status=state["observed_status"], formal_performance_allowed=False,
                     archive_bytes=archive_bytes, archive_sha256=archive_sha,
                     manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                     file_count=len(inventory), raw_bytes=sum(row["bytes"] for row in inventory.values()))
        staged_receipt = Path(temporary) / "receipt"
        _durable_bytes(staged_receipt, _json_bytes(value))
        _publish_new_file(staged_archive, archive)
        try:
            _publish_new_file(staged_receipt, receipt)
        except OSError:
            archive.unlink()
            raise
        _sync_directory(archive.parent)
    return dict(**value, archive=str(archive), receipt=str(receipt),
                receipt_sha256=hashlib.sha256(_json_bytes(value)).hexdigest())


def _recover_members(tar, staging):
    inventory = {}
    total = 0
    manifest = None
    for member in tar:
        if (not member.isfile() or member.issparse() or member.size < 0 or
                member.size > MAX_EXPORT_BYTES or manifest is not None):
            raise ValueError(f"invalid evidence member: {member.name}")
        with tar.extractfile(member) as data:
            if member.name == "manifest.json":
                if member.size > MAX_JSON_BYTES:
                    raise ValueError("archive manifest exceeds size limit")
                manifest = data.read()
                continue
            relative = _safe_member(member.name)
            if member.name in inventory or len(inventory) >= MAX_FILES or total + member.size > MAX_EXPORT_BYTES:
                raise ValueError("duplicate archive member or evidence exceeds limits")
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                size, digest = _digest_stream(data, output)
                output.flush()
                os.fsync(output.fileno())
            if size != member.size:
                raise ValueError(f"truncated member: {member.name}")
            inventory[member.name] = dict(bytes=size, sha256=digest)
            total += size
    if manifest is None or not inventory:
        raise ValueError("archive is missing a manifest or evidence")
    return manifest, inventory, total


def recover(archive, receipt, output, expected_receipt_sha=None):
    with receipt.open("rb") as stream:
        receipt_raw = stream.read(MAX_JSON_BYTES + 1)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if len(receipt_raw) > MAX_JSON_BYTES or (expected_receipt_sha and receipt_sha != expected_receipt_sha):
        raise ValueError("evidence receipt size or SHA-256 differs")
    value = json.loads(receipt_raw)
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("status") != "exported" or
            value.get("transport_only") is not True or value.get("formal_performance_allowed") is not False or
            value.get("attempt_status") not in TERMINAL_STATES):
        raise ValueError("invalid evidence receipt")
    if output.exists() or output.is_symlink():
        raise FileExistsError("recovery destination must be new")
    with archive.open("rb") as stream:
        original = _file_signature(os.fstat(stream.fileno()))
        size, digest = _digest_stream(stream)
        if size != value["archive_bytes"] or digest != value["archive_sha256"]:
            raise ValueError("archive size or SHA-256 differs")
        stream.seek(0)
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            staging = Path(temporary) / "verified"
            staging.mkdir()
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                manifest, inventory, raw_bytes = _recover_members(tar, staging)
            if (_file_signature(os.fstat(stream.fileno())) != original or
                    hashlib.sha256(manifest).hexdigest() != value["manifest_sha256"] or
                    len(inventory) != value["file_count"] or raw_bytes != value["raw_bytes"]):
                raise ValueError("archive identity, manifest hash, or evidence totals differ")
            parsed = json.loads(manifest)
            if parsed != dict(schema=SCHEMA, attempt_status=value["attempt_status"],
                              formal_performance_allowed=False, files=inventory):
                raise ValueError("manifest and recovered evidence inventory differ")
            _durable_bytes(staging / "manifest.json", manifest)
            _durable_bytes(staging / "receipt.json", receipt_raw)
            result = dict(status="TRANSPORT_VERIFIED", attempt_status=value["attempt_status"],
                          transport_only=True, formal_performance_allowed=False, file_count=len(inventory),
                          raw_bytes=raw_bytes, receipt_sha256=receipt_sha, output=str(output))
            _durable_bytes(staging / "verification.json", _json_bytes(result))
            output.mkdir(mode=0o700)
            try:
                os.replace(staging, output)
            except OSError:
                output.rmdir()
                raise
            _sync_directory(output.parent)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--work", type=Path, required=True)
    launch.add_argument("--timeout", type=int, default=7200)
    inspect = commands.add_parser("status")
    inspect.add_argument("--work", type=Path, required=True)
    package = commands.add_parser("export")
    package.add_argument("--work", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    unpack = commands.add_parser("recover")
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--receipt", type=Path, required=True)
    unpack.add_argument("--receipt-sha256")
    unpack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "run":
            return run(args.config, args.work, args.timeout)
        if args.action == "status":
            result = status(args.work)
        elif args.action == "export":
            result = export(args.work, args.archive)
        else:
            if args.receipt_sha256 and not SHA256.fullmatch(args.receipt_sha256):
                raise ValueError("receipt SHA-256 must be 64 lowercase hexadecimal characters")
            result = recover(args.archive, args.receipt, args.output, args.receipt_sha256)
    except (OSError, KeyError, ValueError, RuntimeError, tarfile.TarError) as error:
        print(f"{args.action}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
