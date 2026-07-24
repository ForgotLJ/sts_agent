from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import M6_FINAL_SEED_END, M6_FINAL_SEED_START, load_m6_checkpoint
from sts_env.training.experiment import build_runtime_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze M6 checkpoints before formal evaluation.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="RUN_SEED=PATH",
        help="checkpoint mapping, repeated once for each formal run seed",
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
        raise ValueError("checkpoint must use RUN_SEED=PATH syntax")
    return int(run_seed_text), Path(path_text).resolve()


def main() -> int:
    args = parse_args()
    entries: dict[str, Any] = {}
    seen: set[int] = set()
    current_source_sha256 = build_runtime_manifest(PROJECT_ROOT)["source_sha256"]
    for value in args.checkpoint:
        run_seed, path = parse_checkpoint(value)
        if run_seed in seen:
            raise ValueError(f"duplicate run seed {run_seed}")
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded = load_m6_checkpoint(path, device="cpu")
        if loaded.config.run_seed != run_seed:
            raise ValueError(f"checkpoint {path} belongs to run seed {loaded.config.run_seed}")
        if loaded.scheduler.current.name != "full_run":
            raise ValueError(
                f"run seed {run_seed} is at {loaded.scheduler.current.name}, not full_run"
            )
        if not loaded.manifest.get("evaluation_only"):
            raise ValueError("M6 freeze requires evaluation-only checkpoints")
        if loaded.manifest.get("parameter_source") != "ema":
            raise ValueError("M6 freeze requires EMA evaluation checkpoints")
        source_sha256 = str(loaded.manifest.get("source_sha256") or "")
        if source_sha256 != current_source_sha256:
            raise ValueError("checkpoint source hash differs from the current source tree")
        seen.add(run_seed)
        entries[str(run_seed)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "update": loaded.update_index,
            "run_seed": run_seed,
            "source_sha256": source_sha256,
            "stage": loaded.scheduler.current.name,
            "evaluation_only": True,
            "parameter_source": "ema",
        }
    if seen != {17, 29, 43}:
        raise ValueError("M6 freeze requires run seeds 17, 29, and 43")
    payload = {
        "schema_version": 1,
        "protocol": "docs/M6_IMPLEMENTATION_AND_EVALUATION_PLAN.md",
        "final_seed_range": [M6_FINAL_SEED_START, M6_FINAL_SEED_END],
        "source_sha256": current_source_sha256,
        "checkpoints": entries,
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
