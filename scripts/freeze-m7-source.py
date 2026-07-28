from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import M7_FINAL_SEED_END, M7_FINAL_SEED_START, M7TrainingConfig
from sts_env.training.experiment import build_runtime_manifest
from sts_env.training.teacher_corpus import verify_teacher_corpus_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze M7 source before training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "m7_recurrent_ppo.json",
    )
    parser.add_argument(
        "--gate",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gate(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise ValueError("M7 gate must use NAME=PATH syntax")
    return name, Path(path_text)


def gate_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("complete") is not True or int(payload.get("errors", -1)) != 0:
        raise ValueError(f"M7 gate is incomplete or failed: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "payload": payload,
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != "m7":
        raise ValueError("M7 source freeze requires protocol=m7")
    experiment_payload = dict(config["experiment"])
    run_seeds = sorted(int(seed) for seed in experiment_payload.pop("run_seeds"))
    if run_seeds != [17, 29, 43]:
        raise ValueError("formal M7 source freeze requires run seeds 17, 29, and 43")
    experiment = M7TrainingConfig(run_seed=run_seeds[0], **experiment_payload)
    gates = dict(parse_gate(value) for value in args.gate)
    if len(gates) != len(args.gate):
        raise ValueError("M7 source freeze contains duplicate gate names")
    required = {
        "python-tests",
        "prefix-recovery",
        "stress-10000",
        "communication-differential",
        "teacher-corpus",
    }
    if not required.issubset(gates):
        raise ValueError(f"M7 source freeze lacks gates: {sorted(required - gates.keys())}")
    gate_entries = {name: gate_entry(path) for name, path in sorted(gates.items())}
    teacher_payload = gate_entries["teacher-corpus"]["payload"]
    verify_teacher_corpus_manifest(teacher_payload)
    runtime_manifest = build_runtime_manifest(PROJECT_ROOT)
    if runtime_manifest.get("git_dirty") is True and not args.allow_dirty:
        raise ValueError("formal M7 source freeze requires a clean git worktree")
    ranges = experiment.seed_ranges()
    payload = {
        "protocol": "m7",
        "schema_version": 1,
        "status": "frozen",
        "runtime_manifest": runtime_manifest,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "payload": config,
        },
        "run_seeds": run_seeds,
        "seed_ranges": {
            name: [seed_range.start, seed_range.stop - 1]
            for name, seed_range in ranges.items()
        },
        "final_seed_range": [M7_FINAL_SEED_START, M7_FINAL_SEED_END],
        "teacher_corpus": teacher_payload,
        "gates": gate_entries,
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
