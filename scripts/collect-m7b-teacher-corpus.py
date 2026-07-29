from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import LightspeedBackend, StsEnv
from sts_env.training import HeuristicPolicy
from sts_env.training.m7b_distillation import (
    build_m7b_corpus_manifest,
    record_m7b_teacher_trace,
)
from sts_env.trace import EpisodeTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a persistent M7-B heuristic teacher corpus."
    )
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def collect_one(seed: int, destination: Path, max_steps: int) -> dict[str, Any]:
    trace = record_m7b_teacher_trace(
        StsEnv(LightspeedBackend()),
        HeuristicPolicy(),
        seed=seed,
        max_steps=max_steps,
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
        "phase_supervision_counts": dict(metadata["phase_supervision_counts"]),
    }


def validate_existing_trace(path: Path, seed: int, max_steps: int) -> None:
    trace = EpisodeTrace.read_jsonl(path)
    metadata = dict(trace.metadata or {})
    if (
        trace.seed != seed
        or metadata.get("protocol") != "m7b-teacher"
        or int(metadata.get("collection_max_steps", -1)) != max_steps
    ):
        raise ValueError(f"invalid existing M7-B trace: {path}")


def main() -> int:
    args = parse_args()
    if (
        args.seed_start < 0
        or args.seed_count <= 0
        or args.max_steps <= 0
        or args.workers <= 0
        or args.progress_interval <= 0
    ):
        raise ValueError("M7-B collection arguments are invalid")
    traces_directory = args.output / "traces"
    traces_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pending = []
    reused = 0
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        path = traces_directory / f"seed-{seed:08d}.jsonl"
        if path.is_file():
            validate_existing_trace(path, seed, args.max_steps)
            reused += 1
        else:
            pending.append((seed, path))
    completed = []
    errors = []
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            futures = {
                pool.submit(collect_one, seed, path, args.max_steps): seed
                for seed, path in pending
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
                if completed_count % args.progress_interval == 0:
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
        "protocol": "m7b-teacher-collection",
        "complete": not errors,
        "seed_range": [args.seed_start, args.seed_start + args.seed_count - 1],
        "max_steps": args.max_steps,
        "requested": args.seed_count,
        "reused": reused,
        "collected": len(completed),
        "errors": len(errors),
        "error_messages": errors,
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output / "collection-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1
    manifest = build_m7b_corpus_manifest(
        args.output,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        collection_max_steps=args.max_steps,
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **summary,
                "aggregate_sha256": manifest["aggregate_sha256"],
                "phase_supervision_counts": manifest["phase_supervision_counts"],
                "wins": manifest["wins"],
                "mean_final_floor": manifest["mean_final_floor"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
