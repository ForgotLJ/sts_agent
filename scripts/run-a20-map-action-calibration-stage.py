#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.map_action_audit import audit_map_policy_evaluations
from sts_env.training.map_action_stage import (
    load_json,
    map_evaluation_gate,
    map_override_coverage_gate,
    select_profile_margin_by_coverage,
    sha256_file,
)
from sts_env.training.map_action_value import MAP_ACTION_LABEL_MODE


CARD_CHECKPOINT_SHA256 = "8c7f053c64b9bd57ccba6ae64ecba8586a29d37dfaf1842f00d083b07b113a3c"
V5_MAP_CHECKPOINT_SHA256 = "a7f80ba27b66a631cb4ff51c46085c2046259c6e13afc30bbe8ea11c2e603dda"
CARD_MARGIN = 0.016514360904693604
BOOTSTRAP_SAMPLES = 10_000
TARGET_OVERRIDE_RATE = 0.075
PROFILE_MIN_OVERRIDE_RATE = 0.05
PROFILE_MAX_OVERRIDE_RATE = 0.10
EVALUATION_MIN_OVERRIDE_RATE = 0.03
EVALUATION_MAX_OVERRIDE_RATE = 0.15
SMOKE_MIN_OVERRIDES = 2
FORMAL_MIN_OVERRIDES = 16
TRAINED_ACTS = (1,)
TRAINED_FLOOR_RANGE = (0, 0)
STAGE_PROTOCOL = "a20-map-action-calibration-stage-v6"

PROFILE_RANGE = ("map_act1_value_profile_v6", 2_360_000, 512)
SMOKE_RANGE = ("map_act1_value_smoke_v6", 2_361_000, 64)
FORMAL_RANGE = ("map_act1_value_formal_v6", 2_362_000, 512)
REPLICATION_RANGE = ("map_act1_value_replication_v6", 2_363_000, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen V6 calibrated-margin evaluation stage for the V5 map model."
    )
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--map-checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples != BOOTSTRAP_SAMPLES:
        raise ValueError("V6 bootstrap samples are frozen at 10000")
    if args.dry_run:
        print(json.dumps(_dry_run_payload(args), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.output.exists():
        raise FileExistsError(f"V6 output already exists: {args.output}")
    source_stage = args.source_stage.resolve()
    map_checkpoint = args.map_checkpoint.resolve()
    card_checkpoint = args.card_checkpoint.resolve()
    _validate_inputs(source_stage, map_checkpoint, card_checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "protocol": STAGE_PROTOCOL,
        "schema_version": 1,
        "status": "running",
        "source": _source_identity(),
        "source_stage": {
            "path": str(source_stage),
            "stage_json_sha256": sha256_file(source_stage / "stage.json"),
        },
        "map_checkpoint": {"path": str(map_checkpoint), "sha256": sha256_file(map_checkpoint)},
        "card_checkpoint": {"path": str(card_checkpoint), "sha256": sha256_file(card_checkpoint)},
        "frozen_parameters": _frozen_parameters(args),
        "steps": [],
    }
    state_path = args.output / "stage.json"

    def persist(status: str, *, exit_code: int = 0, **details: Any) -> int:
        state["status"] = status
        state.update(details)
        _write_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
        return exit_code

    def step(name: str, command: list[str] | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": name, "status": "running"}
        if command is not None:
            entry["command"] = command
        state["steps"].append(entry)
        _write_json(state_path, state)
        return entry

    def finish(entry: dict[str, Any], returncode: int | None = None) -> None:
        entry["status"] = "complete" if returncode in (None, 0) else "failed"
        if returncode is not None:
            entry["returncode"] = returncode
        _write_json(state_path, state)

    profile = args.output / "profile.json"
    profile_command = _evaluation_command(
        map_checkpoint,
        card_checkpoint,
        profile,
        *PROFILE_RANGE,
        margin=0.0,
        device=args.evaluation_device,
        bootstrap_samples=args.bootstrap_samples,
        record_only=True,
    )
    profile_step = step("profile", profile_command)
    profile_returncode = _run(profile_command)
    finish(profile_step, profile_returncode)
    if profile_returncode != 0:
        return persist("failed_profile_command", exit_code=1, returncode=profile_returncode)
    try:
        margin = select_profile_margin_by_coverage(
            load_json(profile),
            expected_map_checkpoint_sha256=V5_MAP_CHECKPOINT_SHA256,
            expected_card_checkpoint_sha256=CARD_CHECKPOINT_SHA256,
            target_override_rate=TARGET_OVERRIDE_RATE,
            minimum_override_rate=PROFILE_MIN_OVERRIDE_RATE,
            maximum_override_rate=PROFILE_MAX_OVERRIDE_RATE,
            expected_range_name=PROFILE_RANGE[0],
            expected_trained_acts=frozenset(TRAINED_ACTS),
            expected_trained_floor_range=TRAINED_FLOOR_RANGE,
            expected_label_mode=MAP_ACTION_LABEL_MODE,
        )
    except ValueError as error:
        return persist("stopped_profile_calibration", error=str(error))
    margin["profile"] = {"path": str(profile), "sha256": sha256_file(profile)}
    _write_json(args.output / "margin.json", margin)

    smoke = args.output / "smoke.json"
    smoke_command = _evaluation_command(
        map_checkpoint,
        card_checkpoint,
        smoke,
        *SMOKE_RANGE,
        margin=float(margin["override_margin"]),
        device=args.evaluation_device,
        bootstrap_samples=args.bootstrap_samples,
    )
    smoke_step = step("smoke", smoke_command)
    smoke_returncode = _run(smoke_command)
    finish(smoke_step, smoke_returncode)
    if smoke_returncode != 0:
        return persist("failed_smoke_command", exit_code=1, returncode=smoke_returncode)
    smoke_coverage_step = step("smoke_coverage_gate")
    smoke_coverage = map_override_coverage_gate(
        load_json(smoke),
        minimum_override_rate=EVALUATION_MIN_OVERRIDE_RATE,
        maximum_override_rate=EVALUATION_MAX_OVERRIDE_RATE,
        minimum_overrides=SMOKE_MIN_OVERRIDES,
    )
    _write_json(args.output / "smoke-coverage-gate.json", smoke_coverage)
    finish(smoke_coverage_step)
    if not smoke_coverage["passed"]:
        return persist("stopped_smoke_coverage_gate", smoke_coverage_gate=smoke_coverage)
    smoke_gate_step = step("smoke_gate")
    smoke_gate = _effect_gate(smoke, SMOKE_RANGE[0], map_checkpoint, card_checkpoint, False)
    _write_json(args.output / "smoke-gate.json", smoke_gate)
    finish(smoke_gate_step)
    if not smoke_gate["passed"]:
        return persist("stopped_smoke_gate", smoke_gate=smoke_gate)

    formal = args.output / "formal.json"
    formal_command = _evaluation_command(
        map_checkpoint,
        card_checkpoint,
        formal,
        *FORMAL_RANGE,
        margin=float(margin["override_margin"]),
        device=args.evaluation_device,
        bootstrap_samples=args.bootstrap_samples,
    )
    formal_step = step("formal", formal_command)
    formal_returncode = _run(formal_command)
    finish(formal_step, formal_returncode)
    if formal_returncode != 0:
        return persist("failed_formal_command", exit_code=1, returncode=formal_returncode)
    formal_coverage_step = step("formal_coverage_gate")
    formal_coverage = map_override_coverage_gate(
        load_json(formal),
        minimum_override_rate=EVALUATION_MIN_OVERRIDE_RATE,
        maximum_override_rate=EVALUATION_MAX_OVERRIDE_RATE,
        minimum_overrides=FORMAL_MIN_OVERRIDES,
    )
    _write_json(args.output / "formal-coverage-gate.json", formal_coverage)
    finish(formal_coverage_step)
    if not formal_coverage["passed"]:
        return persist("stopped_formal_coverage_gate", formal_coverage_gate=formal_coverage)
    formal_gate_step = step("formal_gate")
    formal_gate = _effect_gate(formal, FORMAL_RANGE[0], map_checkpoint, card_checkpoint, True)
    _write_json(args.output / "formal-gate.json", formal_gate)
    finish(formal_gate_step)
    if not formal_gate["passed"]:
        return persist("stopped_formal_gate", formal_gate=formal_gate)

    replication = args.output / "replication.json"
    replication_command = _evaluation_command(
        map_checkpoint,
        card_checkpoint,
        replication,
        *REPLICATION_RANGE,
        margin=float(margin["override_margin"]),
        device=args.evaluation_device,
        bootstrap_samples=args.bootstrap_samples,
    )
    replication_step = step("replication", replication_command)
    replication_returncode = _run(replication_command)
    finish(replication_step, replication_returncode)
    if replication_returncode != 0:
        return persist("failed_replication_command", exit_code=1, returncode=replication_returncode)
    replication_coverage_step = step("replication_coverage_gate")
    replication_coverage = map_override_coverage_gate(
        load_json(replication),
        minimum_override_rate=EVALUATION_MIN_OVERRIDE_RATE,
        maximum_override_rate=EVALUATION_MAX_OVERRIDE_RATE,
        minimum_overrides=FORMAL_MIN_OVERRIDES,
    )
    _write_json(args.output / "replication-coverage-gate.json", replication_coverage)
    finish(replication_coverage_step)
    if not replication_coverage["passed"]:
        return persist("stopped_replication_coverage_gate", replication_coverage_gate=replication_coverage)
    replication_gate_step = step("replication_gate")
    replication_gate = _effect_gate(
        replication,
        REPLICATION_RANGE[0],
        map_checkpoint,
        card_checkpoint,
        True,
    )
    _write_json(args.output / "replication-gate.json", replication_gate)
    finish(replication_gate_step)
    if not replication_gate["passed"]:
        return persist("stopped_replication_gate", replication_gate=replication_gate)

    audit_step = step("audit")
    audit = audit_map_policy_evaluations(
        formal,
        replication,
        expected_map_checkpoint_sha256=V5_MAP_CHECKPOINT_SHA256,
        expected_card_checkpoint_sha256=CARD_CHECKPOINT_SHA256,
        bootstrap_samples=args.bootstrap_samples,
        expected_formal_range_name=FORMAL_RANGE[0],
        expected_replication_range_name=REPLICATION_RANGE[0],
        expected_trained_acts=frozenset(TRAINED_ACTS),
        expected_trained_floor_range=TRAINED_FLOOR_RANGE,
        expected_label_mode=MAP_ACTION_LABEL_MODE,
    )
    _write_json(args.output / "audit.json", audit)
    finish(audit_step)
    return persist("complete" if audit["verdict"] == "replicated_improved" else "stopped_audit", audit=audit)


def _validate_inputs(source_stage: Path, map_checkpoint: Path, card_checkpoint: Path) -> None:
    required = (source_stage / "stage.json", source_stage / "frozen-evaluation.json", map_checkpoint, card_checkpoint)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("V6 source stage, map checkpoint, or card checkpoint is missing")
    if sha256_file(map_checkpoint) != V5_MAP_CHECKPOINT_SHA256:
        raise ValueError("map checkpoint SHA-256 differs from the frozen V5 checkpoint")
    if sha256_file(card_checkpoint) != CARD_CHECKPOINT_SHA256:
        raise ValueError("card checkpoint SHA-256 differs from the frozen clone-value baseline")
    source_state = load_json(source_stage / "stage.json")
    if source_state.get("protocol") != "a20-map-action-act1-stage-v5":
        raise ValueError("source stage protocol differs from frozen V5")
    evaluation = load_json(source_stage / "frozen-evaluation.json")
    source_checkpoint = dict(evaluation.get("checkpoint") or {})
    if source_checkpoint.get("sha256") != V5_MAP_CHECKPOINT_SHA256:
        raise ValueError("source stage checkpoint differs from frozen V5")
    if evaluation.get("label_mode") != MAP_ACTION_LABEL_MODE:
        raise ValueError("source map checkpoint label mode differs")
    if not bool(dict(evaluation.get("config") or {}).get("include_behavior_action")):
        raise ValueError("source map checkpoint lacks behavior-action features")


def _effect_gate(
    evaluation_path: Path,
    range_name: str,
    map_checkpoint: Path,
    card_checkpoint: Path,
    require_effect: bool,
) -> dict[str, Any]:
    return map_evaluation_gate(
        load_json(evaluation_path),
        expected_range_name=range_name,
        expected_map_checkpoint_sha256=sha256_file(map_checkpoint),
        expected_card_checkpoint_sha256=sha256_file(card_checkpoint),
        require_effect=require_effect,
        expected_trained_acts=frozenset(TRAINED_ACTS),
        expected_trained_floor_range=TRAINED_FLOOR_RANGE,
        expected_label_mode=MAP_ACTION_LABEL_MODE,
    )


def _evaluation_command(
    map_checkpoint: Path,
    card_checkpoint: Path,
    output: Path,
    range_name: str,
    seed_start: int,
    seed_count: int,
    *,
    margin: float,
    device: str,
    bootstrap_samples: int,
    record_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate-a20-map-value-policy.py"),
        "--checkpoint",
        str(map_checkpoint),
        "--card-checkpoint",
        str(card_checkpoint),
        "--output",
        str(output),
        "--seed-start",
        str(seed_start),
        "--seed-count",
        str(seed_count),
        "--seed-range-name",
        range_name,
        "--override-margin",
        str(margin),
        "--card-override-margin",
        str(CARD_MARGIN),
        "--device",
        device,
        "--bootstrap-samples",
        str(bootstrap_samples),
    ]
    if record_only:
        command.append("--record-only")
    return command


def _frozen_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_map_checkpoint_sha256": V5_MAP_CHECKPOINT_SHA256,
        "card_checkpoint_sha256": CARD_CHECKPOINT_SHA256,
        "card_override_margin": CARD_MARGIN,
        "label_mode": MAP_ACTION_LABEL_MODE,
        "include_behavior_action": True,
        "trained_acts": list(TRAINED_ACTS),
        "trained_floor_range": list(TRAINED_FLOOR_RANGE),
        "profile": _range_payload(PROFILE_RANGE),
        "smoke": _range_payload(SMOKE_RANGE),
        "formal": _range_payload(FORMAL_RANGE),
        "replication": _range_payload(REPLICATION_RANGE),
        "target_override_rate": TARGET_OVERRIDE_RATE,
        "profile_override_rate_interval": [PROFILE_MIN_OVERRIDE_RATE, PROFILE_MAX_OVERRIDE_RATE],
        "evaluation_override_rate_interval": [
            EVALUATION_MIN_OVERRIDE_RATE,
            EVALUATION_MAX_OVERRIDE_RATE,
        ],
        "smoke_min_overrides": SMOKE_MIN_OVERRIDES,
        "formal_min_overrides": FORMAL_MIN_OVERRIDES,
        "evaluation_device": args.evaluation_device,
        "bootstrap_samples": args.bootstrap_samples,
    }


def _range_payload(seed_range: tuple[str, int, int]) -> dict[str, int | str]:
    return {"seed_range_name": seed_range[0], "seed_start": seed_range[1], "seed_count": seed_range[2]}


def _dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": f"{STAGE_PROTOCOL}-dry-run",
        "schema_version": 1,
        "source": _source_identity(),
        "frozen_parameters": _frozen_parameters(args),
        "steps": [
            {"name": "profile", "gate": "record-only target override-rate calibration"},
            {"name": "smoke", "gate": "safety and intervention coverage"},
            {"name": "formal", "gate": "coverage plus positive effect"},
            {"name": "replication", "gate": "independent coverage plus positive effect"},
            {"name": "audit", "gate": "replicated improvement"},
        ],
    }


def _source_identity() -> dict[str, str | bool | None]:
    def git_output(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git_output("status", "--porcelain")
    return {
        "project_root": str(PROJECT_ROOT),
        "commit": git_output("rev-parse", "HEAD"),
        "describe": git_output("describe", "--always", "--tags", "--dirty"),
        "worktree_clean": status == "" if status is not None else None,
    }


def _run(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
