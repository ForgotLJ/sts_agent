from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from sts_env import LightspeedBackend, StsEnv
from sts_env.trace import EpisodeTrace
from sts_env.training import (
    HeuristicPolicy,
    load_m6_checkpoint,
    load_m7_checkpoint,
    load_m7b_checkpoint,
    load_m7c_checkpoint,
)
from sts_env.training.m7c_dagger import (
    build_m7c_corpus_manifest,
    m7c_dagger_labels,
    record_m7c_dagger_trace,
    sha256_file,
    summarize_m7c_on_policy_labels,
)
from sts_env.training.m7c_protocol import (
    m7c_seed_registry,
    require_registered_seed_range,
)


_WORKER_TRAINER: Any | None = None
_WORKER_BEHAVIOR_POLICY: dict[str, Any] | None = None
_WORKER_TEACHER_IDENTITY: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect persistent M7-C student-on-policy DAgger traces."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-range-name", required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--teacher-mix-probability", type=float, required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint_trainer(path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    protocol = str(payload.get("protocol") or "m6")
    if protocol == "m7b":
        loaded = load_m7b_checkpoint(path, device=device)
    elif protocol == "m7c-dagger":
        loaded = load_m7c_checkpoint(path, device=device)
    elif protocol == "m7":
        loaded = load_m7_checkpoint(path, device=device)
    else:
        loaded = load_m6_checkpoint(path, device=device)
        protocol = "m6"
    return loaded.trainer, {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_protocol": protocol,
        "run_seed": int(loaded.config.run_seed),
    }


def teacher_identity() -> str:
    path = PROJECT_ROOT / "src" / "sts_env" / "training" / "policies.py"
    return f"heuristic-policy:{sha256_file(path)}"


def initialize_worker(
    checkpoint: str,
    device: str,
    behavior_policy: dict[str, Any],
    resolved_teacher_identity: str,
) -> None:
    global _WORKER_BEHAVIOR_POLICY, _WORKER_TEACHER_IDENTITY, _WORKER_TRAINER
    trainer, loaded_behavior_policy = load_checkpoint_trainer(Path(checkpoint), device)
    if loaded_behavior_policy != behavior_policy:
        raise ValueError("M7-C checkpoint provenance differs between collector workers")
    _WORKER_TRAINER = trainer
    _WORKER_BEHAVIOR_POLICY = dict(behavior_policy)
    _WORKER_TEACHER_IDENTITY = resolved_teacher_identity


def mixing_seed(run_seed: int, round_index: int, environment_seed: int) -> int:
    if min(run_seed, round_index, environment_seed) < 0:
        raise ValueError("M7-C mixing seed inputs must be non-negative")
    payload = f"{run_seed}:{round_index}:{environment_seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def collect_one(
    seed: int,
    destination: Path,
    *,
    run_seed: int,
    round_index: int,
    teacher_mix_probability: float,
    max_steps: int,
) -> dict[str, Any]:
    if _WORKER_TRAINER is None or _WORKER_BEHAVIOR_POLICY is None or _WORKER_TEACHER_IDENTITY is None:
        raise RuntimeError("M7-C collector worker is not initialized")
    trace = record_m7c_dagger_trace(
        StsEnv(LightspeedBackend()),
        _WORKER_TRAINER,
        HeuristicPolicy(),
        seed=seed,
        max_steps=max_steps,
        teacher_mix_probability=teacher_mix_probability,
        mixing_seed=mixing_seed(run_seed, round_index, seed),
        behavior_policy=_WORKER_BEHAVIOR_POLICY,
        teacher_identity=_WORKER_TEACHER_IDENTITY,
        round_index=round_index,
    )
    temporary = destination.with_name(destination.name + ".tmp")
    trace.write_jsonl(temporary)
    temporary.replace(destination)
    metadata = dict(trace.metadata or {})
    return {
        "seed": seed,
        "steps": len(trace.steps),
        "won": bool(metadata["won"]),
        "final_floor": int(metadata["final_floor"]),
        "student_noncombat_steps": int(metadata["student_noncombat_steps"]),
        "mixed_noncombat_steps": int(metadata["mixed_noncombat_steps"]),
    }


def validate_existing_trace(
    path: Path,
    *,
    seed: int,
    round_index: int,
    teacher_mix_probability: float,
    behavior_policy: dict[str, Any],
    resolved_teacher_identity: str,
) -> None:
    trace = EpisodeTrace.read_jsonl(path)
    metadata = dict(trace.metadata or {})
    if (
        trace.seed != seed
        or int(metadata.get("round_index", -1)) != round_index
        or float(metadata.get("teacher_mix_probability", -1.0)) != teacher_mix_probability
        or dict(metadata.get("behavior_policy") or {}) != behavior_policy
        or str(metadata.get("teacher_identity") or "") != resolved_teacher_identity
    ):
        raise ValueError(f"existing M7-C trace does not match collection request: {path}")
    m7c_dagger_labels(trace)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if (
        args.round_index < 0
        or args.run_seed < 0
        or args.max_steps <= 0
        or args.workers <= 0
        or args.progress_interval <= 0
        or not 0.0 <= args.teacher_mix_probability <= 1.0
    ):
        raise ValueError("M7-C collection arguments are invalid")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M7-C CUDA collection requested unavailable CUDA")
    if args.device.startswith("cuda") and args.workers != 1:
        raise ValueError("M7-C CUDA collection requires exactly one worker")
    registry = m7c_seed_registry()
    if args.seed_range_name not in registry:
        raise ValueError(f"unknown M7-C seed range: {args.seed_range_name}")
    registered_range = registry[args.seed_range_name]
    seed_range = require_registered_seed_range(
        args.seed_range_name,
        start=registered_range.start,
        count=registered_range.count,
        registry=registry,
    )
    if seed_range.locked:
        raise ValueError("M7-C collection cannot write a locked seed range")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _, behavior_policy = load_checkpoint_trainer(checkpoint, "cpu")
    resolved_teacher_identity = teacher_identity()
    traces_directory = args.output / "traces"
    traces_directory.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[int, Path]] = []
    reused = 0
    for seed in seed_range.values:
        path = traces_directory / f"seed-{seed:08d}.jsonl"
        if path.is_file():
            validate_existing_trace(
                path,
                seed=seed,
                round_index=args.round_index,
                teacher_mix_probability=args.teacher_mix_probability,
                behavior_policy=behavior_policy,
                resolved_teacher_identity=resolved_teacher_identity,
            )
            reused += 1
        else:
            pending.append((seed, path))
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    initializer_arguments = (
        str(checkpoint),
        args.device,
        behavior_policy,
        resolved_teacher_identity,
    )
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)),
            initializer=initialize_worker,
            initargs=initializer_arguments,
        ) as pool:
            futures = {
                pool.submit(
                    collect_one,
                    seed,
                    destination,
                    run_seed=args.run_seed,
                    round_index=args.round_index,
                    teacher_mix_probability=args.teacher_mix_probability,
                    max_steps=args.max_steps,
                ): seed
                for seed, destination in pending
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                try:
                    completed.append(future.result())
                except Exception as error:
                    errors.append(
                        {
                            "seed": seed,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                if completed_count % args.progress_interval == 0 or completed_count == len(pending):
                    print(
                        json.dumps(
                            {
                                "completed": completed_count,
                                "pending": len(pending) - completed_count,
                                "errors": len(errors),
                                "wall_seconds": time.perf_counter() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    summary = {
        "protocol": "m7c-dagger-collection",
        "complete": not errors,
        "seed_range_name": seed_range.name,
        "seed_range": [seed_range.start, seed_range.end],
        "round_index": args.round_index,
        "teacher_mix_probability": args.teacher_mix_probability,
        "checkpoint": behavior_policy,
        "teacher_identity": resolved_teacher_identity,
        "requested": seed_range.count,
        "reused": reused,
        "collected": len(completed),
        "errors": len(errors),
        "error_messages": errors,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(args.output / "collection-summary.json", summary)
    if errors:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1
    manifest = build_m7c_corpus_manifest(
        args.output,
        seed_start=seed_range.start,
        seed_count=seed_range.count,
        round_index=args.round_index,
    )
    atomic_write_json(args.output / "manifest.json", manifest)
    diagnostic = summarize_m7c_on_policy_labels(
        tuple(
            EpisodeTrace.read_jsonl(args.output / str(entry["path"]))
            for entry in manifest["files"]
        )
    )
    diagnostic.update(
        {
            "seed_range_name": seed_range.name,
            "seed_range": manifest["seed_range"],
            "round_index": args.round_index,
            "corpus_sha256": manifest["aggregate_sha256"],
            "checkpoint": behavior_policy,
        }
    )
    atomic_write_json(args.output / "on-policy-diagnostic.json", diagnostic)
    print(
        json.dumps(
            {
                **summary,
                "corpus_sha256": manifest["aggregate_sha256"],
                "student_behavior": diagnostic["student_behavior"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
