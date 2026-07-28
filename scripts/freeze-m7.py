from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import (
    M7_FINAL_SEED_END,
    M7_FINAL_SEED_START,
    load_m7_checkpoint,
    validate_m7_fixed_budget_progress,
)
from sts_env.training.experiment import build_runtime_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze M7 checkpoints before final evaluation.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="RUN_SEED=PATH",
    )
    parser.add_argument(
        "--completion-checkpoint",
        action="append",
        required=True,
        metavar="RUN_SEED=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> tuple[int, Path]:
    run_seed_text, separator, path_text = value.partition("=")
    if not separator or not run_seed_text.isdigit() or not path_text:
        raise ValueError("M7 checkpoint must use RUN_SEED=PATH")
    return int(run_seed_text), Path(path_text).resolve()


def validate_completion_checkpoint(
    completed: Any,
    *,
    run_seed: int,
    source_sha256: str,
) -> None:
    if completed.config.run_seed != run_seed:
        raise ValueError("M7 completion checkpoint belongs to another run")
    if completed.manifest.get("evaluation_only"):
        raise ValueError("M7 completion checkpoint must be resumable")
    if completed.scheduler.current.name != "full_run":
        raise ValueError("M7 completion checkpoint is not at full_run")
    validate_m7_fixed_budget_progress(
        completed.config,
        completed.progress,
        completed.update_index,
    )
    if completed.manifest.get("source_sha256") != source_sha256:
        raise ValueError("M7 completion and evaluation checkpoint sources differ")


def main() -> int:
    args = parse_args()
    completion_paths = dict(parse_checkpoint(value) for value in args.completion_checkpoint)
    if len(completion_paths) != len(args.completion_checkpoint):
        raise ValueError("duplicate M7 completion checkpoint run seed")
    if set(completion_paths) != {17, 29, 43}:
        raise ValueError("M7 freeze requires three completion checkpoints")
    entries: dict[str, Any] = {}
    seen: set[int] = set()
    current_source_sha256 = build_runtime_manifest(PROJECT_ROOT)["source_sha256"]
    for value in args.checkpoint:
        run_seed, path = parse_checkpoint(value)
        if run_seed in seen:
            raise ValueError(f"duplicate M7 run seed {run_seed}")
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded = load_m7_checkpoint(path, device="cpu")
        if loaded.config.run_seed != run_seed:
            raise ValueError("M7 checkpoint belongs to a different run seed")
        if loaded.scheduler.current.name != "full_run":
            raise ValueError("M7 checkpoint is not at full_run")
        if not loaded.manifest.get("evaluation_only"):
            raise ValueError("M7 freeze requires evaluation-only checkpoints")
        if loaded.manifest.get("parameter_source") != "ema":
            raise ValueError("M7 freeze requires EMA parameters")
        source_sha256 = str(loaded.manifest.get("source_sha256") or "")
        if source_sha256 != current_source_sha256:
            raise ValueError("M7 checkpoint source differs from the current source tree")
        selection = dict(loaded.manifest.get("selection") or {})
        if not selection or selection.get("stage") != "full_run":
            raise ValueError("M7 checkpoint lacks a full-run selection record")
        completion_path = completion_paths[run_seed]
        if not completion_path.is_file():
            raise FileNotFoundError(completion_path)
        completed = load_m7_checkpoint(completion_path, device="cpu")
        validate_completion_checkpoint(
            completed,
            run_seed=run_seed,
            source_sha256=source_sha256,
        )
        seen.add(run_seed)
        entries[str(run_seed)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "update": loaded.update_index,
            "full_run_updates": loaded.progress.full_run_updates_completed,
            "run_seed": run_seed,
            "source_sha256": source_sha256,
            "stage": loaded.scheduler.current.name,
            "evaluation_only": True,
            "parameter_source": "ema",
            "selection_key": selection.get("selection_key"),
            "selection_combat_policy": loaded.config.selection_combat_policy,
            "completion_checkpoint": {
                "path": str(completion_path),
                "sha256": sha256_file(completion_path),
                "update": completed.update_index,
                "full_run_updates": completed.progress.full_run_updates_completed,
            },
        }
    if seen != {17, 29, 43}:
        raise ValueError("M7 freeze requires run seeds 17, 29, and 43")
    payload = {
        "protocol": "m7",
        "schema_version": 1,
        "final_seed_range": [M7_FINAL_SEED_START, M7_FINAL_SEED_END],
        "source_sha256": current_source_sha256,
        "checkpoints": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
