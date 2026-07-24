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
    M6_FINAL_SEED_END,
    M6_FINAL_SEED_START,
)
from sts_env.training.experiment import build_runtime_manifest
from sts_env.training.teacher_corpus import verify_teacher_corpus_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze M6 source before formal training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "m6_recurrent_ppo.json",
    )
    parser.add_argument(
        "--gate",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="named validated gate artifact; repeat for every required gate",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = None
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("complete") is False or int(payload.get("errors", 0)) != 0:
            raise ValueError(f"M6 gate is incomplete or failed: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "payload": payload,
    }


def parse_gate(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise ValueError("M6 gate must use NAME=PATH syntax")
    return name, Path(path_text)


def validate_required_gate_payload(name: str, payload: dict[str, Any]) -> None:
    if payload.get("complete") is not True or int(payload.get("errors", -1)) != 0:
        raise ValueError(f"required M6 gate lacks an explicit success status: {name}")
    if name == "python-tests" and int(payload.get("tests", 0)) < 100:
        raise ValueError("python-tests gate does not cover the full M6 suite")
    if name == "prefix-recovery" and int(payload.get("checks", 0)) < 1000:
        raise ValueError("prefix-recovery gate requires at least 1,000 checks")
    if name == "stress-10000":
        target = int(payload.get("target_episodes", 0))
        episodes = int(payload.get("episodes", 0))
        if target < 10_000 or episodes < target:
            raise ValueError("stress-10000 gate requires at least 10,000 completed episodes")
    if name == "communication-differential":
        if int(payload.get("records", 0)) <= 0 or int(payload.get("differences", -1)) != 0:
            raise ValueError("communication-differential gate requires nonempty zero-difference coverage")
    if name == "teacher-corpus":
        counts = dict(payload.get("stage_trace_counts") or {})
        if counts != {"act1_clear": 1024, "act2_clear": 14}:
            raise ValueError("teacher-corpus gate has unexpected stage trace counts")
        if int(payload.get("validated_trace_count", 0)) != 1038:
            raise ValueError("teacher-corpus gate did not validate all 1,038 traces")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_seeds = sorted(int(seed) for seed in config["experiment"]["run_seeds"])
    if run_seeds != [17, 29, 43]:
        raise ValueError("formal M6 source freeze requires run seeds 17, 29, and 43")
    gates = dict(parse_gate(value) for value in args.gate)
    if len(gates) != len(args.gate):
        raise ValueError("M6 source freeze contains duplicate gate names")
    required_gates = {
        "python-tests",
        "prefix-recovery",
        "stress-10000",
        "communication-differential",
        "teacher-corpus",
    }
    if not required_gates.issubset(gates):
        missing = sorted(required_gates - gates.keys())
        raise ValueError(f"M6 source freeze lacks required gates: {missing}")
    gate_entries = {
        name: gate_entry(path)
        for name, path in sorted(gates.items())
    }
    for name in sorted(required_gates):
        gate_payload = gate_entries[name]["payload"]
        if gate_payload is None:
            raise ValueError(f"required M6 gate must be a JSON artifact: {name}")
        validate_required_gate_payload(name, gate_payload)
    teacher_payload = gate_entries["teacher-corpus"]["payload"]
    if teacher_payload is None:
        raise ValueError("teacher-corpus gate must be a JSON artifact")
    verify_teacher_corpus_manifest(teacher_payload)
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "protocol": "docs/M6_IMPLEMENTATION_AND_EVALUATION_PLAN.md",
        "runtime_manifest": build_runtime_manifest(PROJECT_ROOT),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "payload": config,
        },
        "run_seeds": run_seeds,
        "training_seed_range": [0, 999999],
        "validation_seed_range": [1100000, 1102047],
        "final_seed_range": [M6_FINAL_SEED_START, M6_FINAL_SEED_END],
        "gates": {
            name: entry
            for name, entry in sorted(gate_entries.items())
        },
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
