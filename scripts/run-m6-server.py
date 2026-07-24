from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.experiment import build_runtime_manifest


RUN_SEEDS = (17, 29, 43)
METHODS = ("random", "heuristic", "heuristic-search", "learned", "learned-search")
FINAL_SEED_START = 2_000_000
FINAL_SEED_COUNT = 1_024
TARGET_UPDATES = 5_000


@dataclass(frozen=True, slots=True)
class Resources:
    parallel_train_runs: int
    parallel_evaluations: int
    omp_threads: int
    torch_threads: int
    torch_interop_threads: int
    stress_workers: int
    stress_torch_threads: int
    stress_rollout_steps: int


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    assets: Path
    pipeline: Path
    gates: Path
    training: Path
    evaluations: Path
    config: Path


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    command: tuple[str, ...]
    stdout: Path
    stderr: Path


class PipelinePaused(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portable Ubuntu M6 formal training and evaluation pipeline."
    )
    parser.add_argument(
        "command",
        choices=("plan", "prepare", "train", "evaluate", "audit", "all", "status", "pause"),
        nargs="?",
        default="plan",
    )
    parser.add_argument("--profile", choices=("conservative", "balanced", "max"), default="balanced")
    parser.add_argument("--parallel-train-runs", type=int)
    parser.add_argument("--parallel-evaluations", type=int)
    parser.add_argument("--omp-threads", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--torch-interop-threads", type=int)
    parser.add_argument("--stress-workers", type=int)
    parser.add_argument("--stress-torch-threads", type=int)
    parser.add_argument("--stress-rollout-steps", type=int)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--assets-dir", type=Path, default=PROJECT_ROOT / "server_assets" / "m6")
    parser.add_argument(
        "--pipeline-root", type=Path, default=PROJECT_ROOT / "experiments" / "m6r_server_pipeline"
    )
    parser.add_argument(
        "--gate-root", type=Path, default=PROJECT_ROOT / "experiments" / "m6r_server_gates"
    )
    parser.add_argument(
        "--training-root", type=Path, default=PROJECT_ROOT / "experiments" / "m6r_server_training"
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m6r_server_evaluations",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "m6_recurrent_ppo.json")
    return parser.parse_args()


def resource_defaults(profile: str, cpu_count: int) -> Resources:
    if profile == "conservative":
        return Resources(1, 2, 1, 2, 1, min(16, cpu_count), 2, 64)
    if profile == "max":
        return Resources(3, 5, 1, 2, 1, min(32, cpu_count), 2, 64)
    return Resources(2 if cpu_count >= 24 else 1, 3, 1, 2, 1, min(24, cpu_count), 2, 64)


def resolve_resources(args: argparse.Namespace) -> Resources:
    defaults = resource_defaults(args.profile, os.cpu_count() or 1)
    values = {
        field: getattr(args, field) if getattr(args, field) is not None else getattr(defaults, field)
        for field in asdict(defaults)
    }
    resources = Resources(**values)
    if min(asdict(resources).values()) <= 0:
        raise ValueError("all resource controls must be positive")
    if resources.parallel_train_runs > len(RUN_SEEDS):
        raise ValueError("parallel-train-runs cannot exceed three formal seeds")
    return resources


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def last_jsonl(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            last = None
            for line in stream:
                if line.strip():
                    last = line
        return dict(json.loads(last)) if last is not None else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def is_m6_pipeline_process(process_id: int) -> bool:
    if process_id <= 0:
        return False
    if os.name == "nt":
        return process_id == os.getpid()
    try:
        command_line = Path(f"/proc/{process_id}/cmdline").read_bytes()
    except OSError:
        return False
    return b"run-m6-server.py" in command_line


class M6ServerPipeline:
    def __init__(
        self,
        paths: PipelinePaths,
        resources: Resources,
        python: Path,
        cuda_visible_devices: str,
        allow_busy_gpu: bool,
    ) -> None:
        self.paths = paths
        self.resources = resources
        self.python = python.resolve()
        self.cuda_visible_devices = cuda_visible_devices
        self.allow_busy_gpu = allow_busy_gpu
        self.children: dict[int, subprocess.Popen[str]] = {}
        self.stop_requested = False
        self.status_path = self.paths.pipeline / "status.json"
        self.events_path = self.paths.pipeline / "events.jsonl"
        self.pid_path = self.paths.pipeline / "pipeline.pid"

    def environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
                "OMP_NUM_THREADS": str(self.resources.omp_threads),
                "MKL_NUM_THREADS": str(self.resources.omp_threads),
                "OPENBLAS_NUM_THREADS": str(self.resources.omp_threads),
                "NUMEXPR_NUM_THREADS": str(self.resources.omp_threads),
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(PROJECT_ROOT / "src"),
            }
        )
        return environment

    def write_status(self, stage: str, data: dict[str, Any]) -> None:
        self.paths.pipeline.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "data": data,
        }
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.status_path)

    def event(self, name: str, data: dict[str, Any]) -> None:
        self.paths.pipeline.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": name,
            "data": data,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def _start(self, job: Job) -> tuple[subprocess.Popen[str], Any, Any]:
        job.stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout = job.stdout.open("a", encoding="utf-8")
        stderr = job.stderr.open("a", encoding="utf-8")
        self.event("job_started", {"name": job.name, "command": list(job.command)})
        process = subprocess.Popen(
            job.command,
            cwd=PROJECT_ROOT,
            env=self.environment(),
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=os.name != "nt",
        )
        self.children[process.pid] = process
        return process, stdout, stderr

    def _stop_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        else:
            process.terminate()

    def terminate_children(self) -> None:
        for process in tuple(self.children.values()):
            self._stop_process(process)
        deadline = time.monotonic() + 20.0
        for process in tuple(self.children.values()):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
        self.children.clear()

    def handle_signal(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.event("pause_requested", {"signal": signum})
        self.terminate_children()
        self.write_status("pipeline_paused", {"state": "paused", "signal": signum})

    def run_job(self, job: Job, *, status: bool = True) -> None:
        if self.stop_requested:
            raise PipelinePaused("pipeline pause requested")
        if status:
            self.write_status(job.name, {"state": "running"})
        process, stdout, stderr = self._start(job)
        try:
            return_code = process.wait()
        finally:
            self.children.pop(process.pid, None)
            stdout.close()
            stderr.close()
        if self.stop_requested:
            raise PipelinePaused("pipeline pause requested")
        if return_code != 0:
            self.event("job_failed", {"name": job.name, "returncode": return_code})
            raise RuntimeError(f"{job.name} failed with exit code {return_code}")
        self.event("job_completed", {"name": job.name, "returncode": return_code})

    def run_parallel(self, jobs: Sequence[Job], maximum: int, stage: str) -> None:
        if self.stop_requested:
            raise PipelinePaused("pipeline pause requested")
        pending = list(jobs)
        running: dict[int, tuple[Job, subprocess.Popen[str], Any, Any]] = {}
        completed = 0
        try:
            while pending or running:
                while pending and len(running) < maximum:
                    job = pending.pop(0)
                    process, stdout, stderr = self._start(job)
                    running[process.pid] = (job, process, stdout, stderr)
                self.write_status(
                    stage,
                    {
                        "state": "running",
                        "completed": completed,
                        "running": len(running),
                        "pending": len(pending),
                        "total": len(jobs),
                        "maximum_parallel": maximum,
                    },
                )
                time.sleep(1.0)
                for process_id, entry in tuple(running.items()):
                    job, process, stdout, stderr = entry
                    return_code = process.poll()
                    if return_code is None:
                        continue
                    self.children.pop(process_id, None)
                    running.pop(process_id)
                    stdout.close()
                    stderr.close()
                    if self.stop_requested:
                        raise PipelinePaused("pipeline pause requested")
                    if return_code != 0:
                        raise RuntimeError(f"{job.name} failed with exit code {return_code}")
                    completed += 1
                    self.event("job_completed", {"name": job.name, "returncode": return_code})
        except BaseException:
            self.terminate_children()
            for _, _, stdout, stderr in running.values():
                stdout.close()
                stderr.close()
            raise

    def check_gpu(self) -> dict[str, Any]:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
        selected_gpu = self.cuda_visible_devices.split(",", 1)[0].strip()
        gpu_rows = [row for row in completed.stdout.splitlines() if row.strip()]
        selected_row = next(
            (row for row in gpu_rows if row.split(",", 1)[0].strip() == selected_gpu),
            gpu_rows[0] if gpu_rows else None,
        )
        if selected_row is None:
            raise RuntimeError("nvidia-smi returned no GPU rows")
        columns = [column.strip() for column in selected_row.split(",")]
        memory_used_mib = int(columns[3])
        if memory_used_mib >= 500 and not self.allow_busy_gpu:
            raise RuntimeError(
                f"GPU {selected_gpu} is already using {memory_used_mib} MiB; "
                "use an idle GPU or pass --allow-busy-gpu deliberately"
            )
        probe = subprocess.run(
            [
                str(self.python),
                "-c",
                "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
            ],
            cwd=PROJECT_ROOT,
            env=self.environment(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode != 0:
            raise RuntimeError(f"PyTorch CUDA probe failed: {probe.stderr.strip()}")
        return {
            "nvidia_smi": completed.stdout.strip(),
            "torch_device": probe.stdout.strip(),
            "selected_gpu": selected_gpu,
            "memory_used_mib": memory_used_mib,
        }

    def source_freeze_current(self) -> bool:
        payload = read_json(self.paths.gates / "source-freeze.json")
        if payload is None:
            return False
        frozen_manifest = dict(payload.get("runtime_manifest") or {})
        frozen_config = dict(payload.get("config") or {})
        return (
            payload.get("status") == "frozen"
            and frozen_manifest.get("source_sha256")
            == build_runtime_manifest(PROJECT_ROOT)["source_sha256"]
            and frozen_config.get("sha256") == sha256_file(self.paths.config)
            and frozen_config.get("path") == str(self.paths.config.resolve())
        )

    def prepare(self) -> None:
        required_assets = (
            self.paths.assets / "teacher-v4",
            self.paths.assets / "seed-17" / "checkpoint.pt",
            self.paths.assets / "communication-differential.json",
        )
        for path in required_assets:
            if not path.exists():
                raise FileNotFoundError(path)
        self.paths.gates.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.paths.assets / "communication-differential.json",
            self.paths.gates / "communication-differential.json",
        )
        self.event("server_preflight", {"resources": asdict(self.resources)})
        commands = (
            Job(
                "python_tests",
                (str(self.python), "scripts/run-m6-python-gate.py", "--output", str(self.paths.gates / "python-tests.json")),
                self.paths.pipeline / "python-tests.stdout.log",
                self.paths.pipeline / "python-tests.stderr.log",
            ),
            Job(
                "prefix_recovery",
                (
                    str(self.python),
                    "scripts/verify-prefix-recovery.py",
                    "--checks",
                    "1000",
                    "--episodes",
                    "32",
                    "--output",
                    str(self.paths.gates / "prefix-recovery.json"),
                ),
                self.paths.pipeline / "prefix-recovery.stdout.log",
                self.paths.pipeline / "prefix-recovery.stderr.log",
            ),
            Job(
                "stress_10000",
                (
                    str(self.python),
                    "scripts/stress-m6-episodes.py",
                    "--episodes",
                    "10000",
                    "--workers",
                    str(self.resources.stress_workers),
                    "--rollout-steps",
                    str(self.resources.stress_rollout_steps),
                    "--torch-threads",
                    str(self.resources.stress_torch_threads),
                    "--device",
                    "cuda",
                    "--output",
                    str(self.paths.gates / "stress-10000.json"),
                ),
                self.paths.pipeline / "stress-10000.stdout.log",
                self.paths.pipeline / "stress-10000.stderr.log",
            ),
            Job(
                "teacher_corpus",
                (
                    str(self.python),
                    "scripts/manifest-m6-teacher-corpus.py",
                    "--corpus",
                    str(self.paths.assets / "teacher-v4"),
                    "--output",
                    str(self.paths.gates / "teacher-corpus.json"),
                ),
                self.paths.pipeline / "teacher-corpus.stdout.log",
                self.paths.pipeline / "teacher-corpus.stderr.log",
            ),
        )
        for job in commands:
            self.run_job(job)
        freeze_command = [str(self.python), "scripts/freeze-m6-source.py"]
        for gate_name in (
            "python-tests",
            "prefix-recovery",
            "stress-10000",
            "communication-differential",
            "teacher-corpus",
        ):
            freeze_command.extend(
                ["--gate", f"{gate_name}={self.paths.gates / f'{gate_name}.json'}"]
            )
        freeze_command.extend(
            ["--config", str(self.paths.config), "--output", str(self.paths.gates / "source-freeze.json")]
        )
        self.run_job(
            Job(
                "freeze_source",
                tuple(freeze_command),
                self.paths.pipeline / "freeze-source.stdout.log",
                self.paths.pipeline / "freeze-source.stderr.log",
            )
        )
        migration_smoke_root = PROJECT_ROOT / "experiments" / "m6r_server_migration_smoke"
        self.run_job(
            Job(
                "migration_resume_smoke",
                (
                    str(self.python),
                    "scripts/train-m6.py",
                    "--config",
                    str(self.paths.config),
                    "--run-seed",
                    "17",
                    "--output",
                    str(migration_smoke_root),
                    "--source-freeze",
                    str(self.paths.gates / "source-freeze.json"),
                    "--resume",
                    str(self.paths.assets / "seed-17" / "checkpoint.pt"),
                    "--stop-after-update",
                    "101",
                    "--torch-threads",
                    str(self.resources.torch_threads),
                    "--torch-interop-threads",
                    str(self.resources.torch_interop_threads),
                ),
                self.paths.pipeline / "migration-resume-smoke.stdout.log",
                self.paths.pipeline / "migration-resume-smoke.stderr.log",
            )
        )
        smoke_metric = last_jsonl(migration_smoke_root / "seed-17" / "metrics.jsonl")
        if smoke_metric is None or int(smoke_metric.get("update", -1)) != 101:
            raise RuntimeError("Windows-to-Linux checkpoint migration smoke test failed")

    def training_complete(self, run_seed: int) -> bool:
        run_directory = self.paths.training / f"seed-{run_seed}"
        metric = last_jsonl(run_directory / "metrics.jsonl")
        return (
            metric is not None
            and int(metric.get("update", -1)) == TARGET_UPDATES
            and metric.get("stage") == "full_run"
            and (run_directory / "best-evaluation-checkpoint.pt").is_file()
        )

    def train(self) -> None:
        if not self.source_freeze_current():
            raise RuntimeError("server source freeze is missing or stale; run prepare first")
        jobs: list[Job] = []
        for run_seed in RUN_SEEDS:
            if self.training_complete(run_seed):
                self.event("training_skipped_complete", {"run_seed": run_seed})
                continue
            run_directory = self.paths.training / f"seed-{run_seed}"
            run_directory.mkdir(parents=True, exist_ok=True)
            command = [
                str(self.python),
                "scripts/train-m6.py",
                "--config",
                str(self.paths.config),
                "--run-seed",
                str(run_seed),
                "--output",
                str(self.paths.training),
                "--source-freeze",
                str(self.paths.gates / "source-freeze.json"),
                "--torch-threads",
                str(self.resources.torch_threads),
                "--torch-interop-threads",
                str(self.resources.torch_interop_threads),
            ]
            local_checkpoint = run_directory / "checkpoint.pt"
            transfer_checkpoint = self.paths.assets / "seed-17" / "checkpoint.pt"
            resume = local_checkpoint if local_checkpoint.is_file() else transfer_checkpoint if run_seed == 17 else None
            if resume is not None:
                command.extend(["--resume", str(resume)])
            jobs.append(
                Job(
                    f"train_seed_{run_seed}",
                    tuple(command),
                    run_directory / "formal-train.stdout.log",
                    run_directory / "formal-train.stderr.log",
                )
            )
        if jobs:
            self.run_parallel(jobs, self.resources.parallel_train_runs, "formal_training")
        incomplete = [run_seed for run_seed in RUN_SEEDS if not self.training_complete(run_seed)]
        if incomplete:
            raise RuntimeError(f"formal training incomplete for seeds: {incomplete}")

    def freeze_checkpoints(self) -> None:
        command = [str(self.python), "scripts/freeze-m6.py"]
        for run_seed in RUN_SEEDS:
            checkpoint = self.paths.training / f"seed-{run_seed}" / "best-evaluation-checkpoint.pt"
            command.extend(["--checkpoint", f"{run_seed}={checkpoint}"])
        command.extend(["--output", str(self.paths.gates / "checkpoint-freeze.json")])
        self.run_job(
            Job(
                "freeze_checkpoints",
                tuple(command),
                self.paths.pipeline / "freeze-checkpoints.stdout.log",
                self.paths.pipeline / "freeze-checkpoints.stderr.log",
            )
        )

    def evaluation_complete(self, run_seed: int, method: str, path: Path) -> bool:
        payload = read_json(path)
        summary = dict((payload or {}).get("summary") or {})
        expected_run_seed = run_seed if method.startswith("learned") else None
        return (
            payload is not None
            and bool(payload.get("final"))
            and payload.get("method") == method
            and int(payload.get("policy_seed", -1)) == run_seed
            and payload.get("run_seed") == expected_run_seed
            and payload.get("seed_range") == [FINAL_SEED_START, FINAL_SEED_START + FINAL_SEED_COUNT - 1]
            and int(summary.get("errors", -1)) == 0
            and len(list(summary.get("episodes") or [])) == FINAL_SEED_COUNT
        )

    def evaluate(self) -> None:
        checkpoint_freeze = self.paths.gates / "checkpoint-freeze.json"
        if not checkpoint_freeze.is_file():
            self.freeze_checkpoints()
        jobs: list[Job] = []
        evaluation_paths: list[Path] = []
        for run_seed in RUN_SEEDS:
            run_directory = self.paths.evaluations / f"run-{run_seed}"
            run_directory.mkdir(parents=True, exist_ok=True)
            checkpoint = self.paths.training / f"seed-{run_seed}" / "best-evaluation-checkpoint.pt"
            for method in METHODS:
                output = run_directory / f"{method}.json"
                evaluation_paths.append(output)
                if self.evaluation_complete(run_seed, method, output):
                    self.event("evaluation_skipped_complete", {"run_seed": run_seed, "method": method})
                    continue
                command = [
                    str(self.python),
                    "scripts/evaluate-m6.py",
                    "--method",
                    method,
                    "--seed-start",
                    str(FINAL_SEED_START),
                    "--seed-count",
                    str(FINAL_SEED_COUNT),
                    "--policy-seed",
                    str(run_seed),
                    "--search-budget",
                    "64",
                    "--max-steps",
                    "5000",
                    "--bootstrap-samples",
                    "10000",
                    "--final",
                    "--freeze-manifest",
                    str(checkpoint_freeze),
                    "--output",
                    str(output),
                ]
                if method.startswith("learned"):
                    command.extend(["--checkpoint", str(checkpoint)])
                jobs.append(
                    Job(
                        f"evaluate_{run_seed}_{method}",
                        tuple(command),
                        run_directory / f"{method}.stdout.log",
                        run_directory / f"{method}.stderr.log",
                    )
                )
        if jobs:
            self.run_parallel(jobs, self.resources.parallel_evaluations, "final_evaluations")
        summary_command = [str(self.python), "scripts/summarize-m6-evaluations.py"]
        for path in evaluation_paths:
            summary_command.extend(["--evaluation", str(path)])
        summary_command.extend(
            [
                "--reference-method",
                "heuristic",
                "--bootstrap-samples",
                "10000",
                "--output",
                str(self.paths.evaluations / "summary.json"),
            ]
        )
        self.run_job(
            Job(
                "summarize_final_evaluations",
                tuple(summary_command),
                self.paths.evaluations / "summary.stdout.log",
                self.paths.evaluations / "summary.stderr.log",
            )
        )
        learned_wins = 0
        for path in evaluation_paths:
            payload = read_json(path) or {}
            if str(payload.get("method", "")).startswith("learned"):
                summary = dict(payload.get("summary") or {})
                learned_wins += sum(
                    bool(episode.get("won")) for episode in list(summary.get("episodes") or [])
                )
        if learned_wins < 1:
            raise RuntimeError("formal evaluation produced no complete learned A0 win")
        self.write_status(
            "pipeline_complete",
            {
                "state": "complete",
                "learned_wins": learned_wins,
                "final_evaluations": len(evaluation_paths),
                "summary": str(self.paths.evaluations / "summary.json"),
            },
        )

    def audit(self) -> None:
        command = (
            str(self.python),
            "scripts/audit-m6.py",
            "--pipeline-root",
            str(self.paths.pipeline),
            "--gate-root",
            str(self.paths.gates),
            "--training-root",
            str(self.paths.training),
            "--evaluation-root",
            str(self.paths.evaluations),
        )
        self.run_job(
            Job(
                "completion_audit",
                command,
                self.paths.pipeline / "completion-audit.stdout.log",
                self.paths.pipeline / "completion-audit.stderr.log",
            ),
            status=False,
        )

    def plan(self) -> None:
        payload = {
            "project_root": str(PROJECT_ROOT),
            "python": str(self.python),
            "cuda_visible_devices": self.cuda_visible_devices,
            "resources": asdict(self.resources),
            "paths": {name: str(path) for name, path in asdict(self.paths).items()},
            "formal_config_unchanged": True,
            "run_seeds": list(RUN_SEEDS),
            "final_evaluations": len(RUN_SEEDS) * len(METHODS),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    def status(self) -> None:
        payload = read_json(self.status_path) or {"stage": "not_started", "data": {}}
        training = {}
        for run_seed in RUN_SEEDS:
            metric = last_jsonl(self.paths.training / f"seed-{run_seed}" / "metrics.jsonl")
            training[str(run_seed)] = (
                {"update": metric.get("update"), "stage": metric.get("stage")}
                if metric is not None
                else None
            )
        payload["training"] = training
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    def pause(self) -> None:
        try:
            process_id = int(self.pid_path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, ValueError):
            raise RuntimeError("no active server pipeline PID was found")
        if process_id == os.getpid():
            raise RuntimeError("pause must be invoked from a separate process")
        if not is_m6_pipeline_process(process_id):
            raise RuntimeError(f"PID {process_id} is not an active M6 server pipeline")
        os.kill(process_id, signal.SIGTERM)
        print(json.dumps({"pause_requested": True, "process_id": process_id}, sort_keys=True))

    def run(self, command: str) -> None:
        if command == "plan":
            self.plan()
            return
        if command == "status":
            self.status()
            return
        if command == "pause":
            self.pause()
            return
        self.paths.pipeline.mkdir(parents=True, exist_ok=True)
        if self.pid_path.is_file():
            try:
                existing_process_id = int(self.pid_path.read_text(encoding="ascii").strip())
                os.kill(existing_process_id, 0)
            except (ValueError, ProcessLookupError):
                pass
            except PermissionError as error:
                raise RuntimeError(
                    f"cannot inspect existing M6 pipeline PID {existing_process_id}"
                ) from error
            else:
                if is_m6_pipeline_process(existing_process_id):
                    raise RuntimeError(
                        f"M6 server pipeline is already running as PID {existing_process_id}"
                    )
        self.pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        self.event("pipeline_started", {"command": command, "resources": asdict(self.resources)})
        try:
            if command in {"prepare", "train", "evaluate", "all"}:
                self.event("gpu_preflight", self.check_gpu())
            if command in {"prepare", "all"} and (command == "prepare" or not self.source_freeze_current()):
                self.prepare()
            if command in {"train", "all"}:
                self.train()
            if command in {"evaluate", "all"}:
                self.freeze_checkpoints()
                self.evaluate()
            if command in {"audit", "all"}:
                self.audit()
            self.event("pipeline_finished", {"command": command})
        except PipelinePaused:
            self.event("pipeline_paused", {"command": command})
        except BaseException as error:
            self.write_status("pipeline_failed", {"state": "failed", "message": str(error)})
            self.event("pipeline_failed", {"command": command, "message": str(error)})
            raise
        finally:
            self.terminate_children()
            try:
                if int(self.pid_path.read_text(encoding="ascii").strip()) == os.getpid():
                    self.pid_path.unlink()
            except (FileNotFoundError, ValueError):
                pass


def main() -> int:
    args = parse_args()
    resources = resolve_resources(args)
    paths = PipelinePaths(
        assets=args.assets_dir.resolve(),
        pipeline=args.pipeline_root.resolve(),
        gates=args.gate_root.resolve(),
        training=args.training_root.resolve(),
        evaluations=args.evaluation_root.resolve(),
        config=args.config.resolve(),
    )
    pipeline = M6ServerPipeline(
        paths,
        resources,
        args.python,
        args.cuda_visible_devices,
        args.allow_busy_gpu,
    )
    pipeline.run(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
