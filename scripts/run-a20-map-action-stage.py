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
    select_profile_margin,
    sha256_file,
)
from sts_env.training.map_counterfactual_diagnostics import diagnose_map_counterfactual_corpus
from sts_env.training.map_counterfactual import validate_map_counterfactual_corpus


CARD_MARGIN = 0.016514360904693604
CARD_CHECKPOINT_SHA256 = "8c7f053c64b9bd57ccba6ae64ecba8586a29d37dfaf1842f00d083b07b113a3c"
BOOTSTRAP_SAMPLES = 10_000
STAGE_PROTOCOL = "a20-map-action-act1-stage-v2"
TRAINED_ACTS = (1,)
COLLECTION_SEED_START = 2_322_000
COLLECTION_SEED_COUNT = 4_096
PROFILE_SEED_START = 2_328_000
SMOKE_SEED_START = 2_329_000
FORMAL_SEED_START = 2_330_000
REPLICATION_SEED_START = 2_331_000
TRAINING_SEED = 17

_ACTIVE_STAGE: tuple[Path, dict[str, Any]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered A20 Ironclad Act 1 map-action stage with automatic gates."
    )
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-device", default="cpu")
    parser.add_argument("--training-device", default="cuda")
    parser.add_argument("--evaluation-device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--collection-per-act", type=int, default=300)
    parser.add_argument("--collection-particles", type=int, default=2)
    parser.add_argument("--collection-rollout-max-steps", type=int, default=5000)
    parser.add_argument("--training-epochs", type=int, default=60)
    parser.add_argument("--training-groups-per-batch", type=int, default=32)
    parser.add_argument("--training-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen stage plan without accessing checkpoints, pilot data, or the simulator.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.collection_per_act != 300 or args.collection_particles != 2:
        raise ValueError("map-stage corpus quota and particles are frozen at 300 and 2")
    if args.collection_rollout_max_steps != 5000:
        raise ValueError("map-stage rollout horizon is frozen at 5000")
    if args.training_epochs != 60 or args.training_groups_per_batch != 32:
        raise ValueError("map-stage training epochs and batch groups are frozen")
    if args.training_learning_rate != 3e-4:
        raise ValueError("map-stage learning rate is frozen")
    if args.bootstrap_samples != BOOTSTRAP_SAMPLES:
        raise ValueError("map-stage bootstrap samples are frozen at 10000")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": f"{STAGE_PROTOCOL}-dry-run",
                    "schema_version": 2,
                    "source": _source_identity(),
                    "frozen_parameters": _frozen_parameters(args),
                    "steps": _stage_plan(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output.exists():
        raise FileExistsError(f"stage output already exists: {args.output}")
    card_checkpoint = args.card_checkpoint.resolve()
    if not card_checkpoint.is_file():
        raise FileNotFoundError(card_checkpoint)
    card_hash = sha256_file(card_checkpoint)
    if card_hash != CARD_CHECKPOINT_SHA256:
        raise ValueError("card checkpoint SHA-256 differs from the frozen clone-value baseline")
    args.output.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "protocol": STAGE_PROTOCOL,
        "schema_version": 2,
        "status": "running",
        "source": _source_identity(),
        "frozen_parameters": _frozen_parameters(args),
        "plan": _stage_plan(),
        "card_checkpoint": {"path": str(card_checkpoint), "sha256": card_hash},
        "steps": [],
    }
    state_path = args.output / "stage.json"
    global _ACTIVE_STAGE
    _ACTIVE_STAGE = (state_path, state)
    _write_json(state_path, state)

    def persist(status: str, *, exit_code: int = 0, **details: Any) -> int:
        state["status"] = status
        state.update(details)
        _write_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
        return exit_code

    def step(name: str, command: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name, "status": "running"}
        if command is not None:
            payload["command"] = command
        state["steps"].append(payload)
        _write_json(state_path, state)
        return payload

    def finish_step(payload: dict[str, Any], *, returncode: int | None = None) -> None:
        payload["status"] = "complete" if returncode in (None, 0) else "failed"
        if returncode is not None:
            payload["returncode"] = returncode
        _write_json(state_path, state)

    pilot_step = step("pilot_diagnostic")
    pilot_validation = validate_map_counterfactual_corpus(args.pilot)
    pilot_diagnostic = diagnose_map_counterfactual_corpus(args.pilot)
    (args.output / "pilot-validation.json").write_text(
        json.dumps(pilot_validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "pilot-diagnostics.json").write_text(
        json.dumps(pilot_diagnostic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finish_step(pilot_step)
    if not pilot_validation["valid"] or not pilot_validation["complete"]:
        return persist("stopped_pilot_validation", pilot_validation=pilot_validation)
    try:
        pilot_manifest = load_json(args.pilot / "manifest.json")
        pilot_checkpoint_hash = str(pilot_manifest["checkpoint"]["sha256"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        return persist("stopped_pilot_manifest", error=str(error))
    if pilot_checkpoint_hash != card_hash:
        return persist(
            "stopped_pilot_checkpoint_mismatch",
            pilot_checkpoint_sha256=pilot_checkpoint_hash,
        )
    if not pilot_diagnostic["scale_gate"]["eligible"]:
        return persist("stopped_pilot_gate", pilot_diagnostic=pilot_diagnostic)

    corpus = args.output / "corpus"
    collection_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "collect-map-counterfactual-corpus.py"),
        "--checkpoint",
        str(card_checkpoint),
        "--output",
        str(corpus),
        "--seed-start",
        str(COLLECTION_SEED_START),
        "--seed-count",
        str(COLLECTION_SEED_COUNT),
        "--seed-range-name",
        "map_act1_collection_v2",
        "--acts",
        "1",
        "--per-act",
        "300",
        "--particles-per-action",
        "2",
        "--max-decisions-per-seed",
        "1",
        "--rollout-max-steps",
        "5000",
        "--override-margin",
        str(CARD_MARGIN),
        "--device",
        args.rollout_device,
    ]
    collection_step = step("collect_corpus", collection_command)
    collection_returncode = _run(collection_command)
    finish_step(collection_step, returncode=collection_returncode)
    if collection_returncode != 0:
        return persist(
            "failed_collection_command",
            exit_code=1,
            returncode=collection_returncode,
        )
    corpus_validation = validate_map_counterfactual_corpus(corpus)
    (args.output / "corpus-validation.json").write_text(
        json.dumps(corpus_validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not corpus_validation["valid"] or not corpus_validation["complete"]:
        return persist("stopped_corpus_validation", corpus_validation=corpus_validation)
    corpus_diagnostic_step = step("corpus_diagnostic")
    corpus_diagnostic = diagnose_map_counterfactual_corpus(
        corpus,
        min_records=300,
        min_contrasting_fraction=0.20,
    )
    (args.output / "corpus-diagnostics.json").write_text(
        json.dumps(corpus_diagnostic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finish_step(corpus_diagnostic_step)
    if not corpus_diagnostic["scale_gate"]["eligible"]:
        return persist("stopped_corpus_diagnostic", corpus_diagnostic=corpus_diagnostic)

    checkpoint = args.output / "a20-map-action-value-IRONCLAD.pt"
    frozen_evaluation = args.output / "frozen-evaluation.json"
    training_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train-map-action-value.py"),
        "--input",
        str(corpus),
        "--output",
        str(checkpoint),
        "--frozen-evaluation",
        str(frozen_evaluation),
        "--epochs",
        "60",
        "--groups-per-batch",
        "32",
        "--learning-rate",
        "0.0003",
        "--seed",
        str(TRAINING_SEED),
        "--device",
        args.training_device,
    ]
    training_step = step("train_map_model", training_command)
    training_returncode = _run(training_command)
    finish_step(training_step, returncode=training_returncode)
    if training_returncode != 0:
        return persist(
            "failed_training_command",
            exit_code=1,
            returncode=training_returncode,
        )
    if not checkpoint.is_file() or not frozen_evaluation.is_file():
        return persist("stopped_training_artifact_missing")
    try:
        offline_evaluation = load_json(frozen_evaluation)
        trained_acts = tuple(int(value) for value in offline_evaluation["trained_acts"])
        act_counts = {
            str(act): int(count)
            for act, count in dict(offline_evaluation["corpus"]["act_counts"]).items()
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        return persist("stopped_training_scope_validation", error=str(error))
    if trained_acts != TRAINED_ACTS or act_counts != {"1": args.collection_per_act}:
        return persist(
            "stopped_training_scope_validation",
            trained_acts=list(trained_acts),
            act_counts=act_counts,
        )
    map_hash = sha256_file(checkpoint)

    profile = args.output / "profile.json"
    profile_command = _evaluation_command(
        checkpoint,
        card_checkpoint,
        profile,
        seed_start=PROFILE_SEED_START,
        seed_count=128,
        range_name="map_act1_value_profile_v2",
        map_margin=0.0,
        device=args.evaluation_device,
        bootstrap_samples=args.bootstrap_samples,
        record_only=True,
    )
    profile_step = step("profile_map_policy", profile_command)
    profile_returncode = _run(profile_command)
    finish_step(profile_step, returncode=profile_returncode)
    if profile_returncode != 0:
        return persist("failed_profile_command", exit_code=1, returncode=profile_returncode)
    try:
        margin = select_profile_margin(
            load_json(profile),
            expected_map_checkpoint_sha256=map_hash,
            expected_card_checkpoint_sha256=card_hash,
            quantile="p80",
            expected_range_name="map_act1_value_profile_v2",
            expected_trained_acts=frozenset(TRAINED_ACTS),
        )
    except ValueError as error:
        return persist("stopped_profile_validation", error=str(error))
    margin["profile"] = {"path": str(profile), "sha256": sha256_file(profile)}
    margin_path = args.output / "margin.json"
    margin_path.write_text(
        json.dumps(margin, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    smoke = args.output / "smoke.json"
    smoke_command = _evaluation_command(
        checkpoint,
        card_checkpoint,
        smoke,
        seed_start=SMOKE_SEED_START,
        seed_count=32,
        range_name="map_act1_value_smoke_v2",
        map_margin=float(margin["override_margin"]),
        device=args.evaluation_device,
        bootstrap_samples=args.bootstrap_samples,
    )
    smoke_step = step("smoke_map_policy", smoke_command)
    smoke_returncode = _run(smoke_command)
    finish_step(smoke_step, returncode=smoke_returncode)
    if smoke_returncode != 0:
        return persist("failed_smoke_command", exit_code=1, returncode=smoke_returncode)
    smoke_gate_step = step("smoke_gate")
    smoke_gate = map_evaluation_gate(
        load_json(smoke),
        expected_range_name="map_act1_value_smoke_v2",
        expected_map_checkpoint_sha256=map_hash,
        expected_card_checkpoint_sha256=card_hash,
        require_effect=False,
        expected_trained_acts=frozenset(TRAINED_ACTS),
    )
    (args.output / "smoke-gate.json").write_text(
        json.dumps(smoke_gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finish_step(smoke_gate_step)
    if not smoke_gate["passed"]:
        return persist("stopped_smoke_gate", smoke_gate=smoke_gate)

    formal = args.output / "formal.json"
    formal_command = _evaluation_command(
        checkpoint,
        card_checkpoint,
        formal,
        seed_start=FORMAL_SEED_START,
        seed_count=512,
        range_name="map_act1_value_formal_v2",
        map_margin=float(margin["override_margin"]),
        device=args.evaluation_device,
        bootstrap_samples=args.bootstrap_samples,
    )
    formal_step = step("formal_map_policy", formal_command)
    formal_returncode = _run(formal_command)
    finish_step(formal_step, returncode=formal_returncode)
    if formal_returncode != 0:
        return persist("failed_formal_command", exit_code=1, returncode=formal_returncode)
    formal_gate_step = step("formal_gate")
    formal_gate = map_evaluation_gate(
        load_json(formal),
        expected_range_name="map_act1_value_formal_v2",
        expected_map_checkpoint_sha256=map_hash,
        expected_card_checkpoint_sha256=card_hash,
        require_effect=True,
        expected_trained_acts=frozenset(TRAINED_ACTS),
    )
    (args.output / "formal-gate.json").write_text(
        json.dumps(formal_gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finish_step(formal_gate_step)
    if not formal_gate["passed"]:
        return persist("stopped_formal_gate", formal_gate=formal_gate)

    replication = args.output / "replication.json"
    replication_command = _evaluation_command(
        checkpoint,
        card_checkpoint,
        replication,
        seed_start=REPLICATION_SEED_START,
        seed_count=512,
        range_name="map_act1_value_replication_v2",
        map_margin=float(margin["override_margin"]),
        device=args.evaluation_device,
        bootstrap_samples=args.bootstrap_samples,
    )
    replication_step = step("replication_map_policy", replication_command)
    replication_returncode = _run(replication_command)
    finish_step(replication_step, returncode=replication_returncode)
    if replication_returncode != 0:
        return persist("failed_replication_command", exit_code=1, returncode=replication_returncode)
    replication_gate_step = step("replication_gate")
    replication_gate = map_evaluation_gate(
        load_json(replication),
        expected_range_name="map_act1_value_replication_v2",
        expected_map_checkpoint_sha256=map_hash,
        expected_card_checkpoint_sha256=card_hash,
        require_effect=True,
        expected_trained_acts=frozenset(TRAINED_ACTS),
    )
    (args.output / "replication-gate.json").write_text(
        json.dumps(replication_gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finish_step(replication_gate_step)
    if not replication_gate["passed"]:
        return persist("stopped_replication_gate", replication_gate=replication_gate)

    audit_step = step("audit")
    audit = audit_map_policy_evaluations(
        formal,
        replication,
        expected_map_checkpoint_sha256=map_hash,
        expected_card_checkpoint_sha256=card_hash,
        bootstrap_samples=args.bootstrap_samples,
        expected_formal_range_name="map_act1_value_formal_v2",
        expected_replication_range_name="map_act1_value_replication_v2",
        expected_trained_acts=frozenset(TRAINED_ACTS),
    )
    (args.output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finish_step(audit_step)
    return persist("complete" if audit["verdict"] == "replicated_improved" else "stopped_audit", audit=audit)


def _frozen_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "card_override_margin": CARD_MARGIN,
        "card_checkpoint_sha256": CARD_CHECKPOINT_SHA256,
        "collection": {
            "seed_start": COLLECTION_SEED_START,
            "seed_count": COLLECTION_SEED_COUNT,
            "acts": list(TRAINED_ACTS),
            "per_act": args.collection_per_act,
            "particles_per_action": args.collection_particles,
            "max_decisions_per_seed": 1,
            "rollout_max_steps": args.collection_rollout_max_steps,
            "device": args.rollout_device,
        },
        "training": {
            "epochs": args.training_epochs,
            "groups_per_batch": args.training_groups_per_batch,
            "learning_rate": args.training_learning_rate,
            "seed": TRAINING_SEED,
            "device": args.training_device,
        },
        "evaluation": {
            "device": args.evaluation_device,
            "bootstrap_samples": args.bootstrap_samples,
            "profile": {"seed_start": PROFILE_SEED_START, "seed_count": 128},
            "smoke": {"seed_start": SMOKE_SEED_START, "seed_count": 32},
            "formal": {"seed_start": FORMAL_SEED_START, "seed_count": 512},
            "replication": {"seed_start": REPLICATION_SEED_START, "seed_count": 512},
            "margin_quantile": "p80",
        },
    }


def _stage_plan() -> list[dict[str, str]]:
    return [
        {"name": "pilot_diagnostic", "gate": "complete validated pilot with eligible diagnostic"},
        {"name": "collect_corpus", "gate": "complete validated 300-decision Act 1 corpus"},
        {"name": "corpus_diagnostic", "gate": "eligible counterfactual contrast diagnostic"},
        {"name": "train_map_model", "gate": "checkpoint and frozen offline metrics exist"},
        {"name": "profile_map_policy", "gate": "record-only identity profile yields a finite p80 margin"},
        {"name": "smoke_map_policy", "gate": "all safety counters are zero"},
        {"name": "formal_map_policy", "gate": "positive final-floor CI and nonnegative Act 1 difference"},
        {"name": "replication_map_policy", "gate": "independent positive final-floor CI and nonnegative Act 1 difference"},
        {"name": "audit", "gate": "two disjoint formal results replicate improvement"},
    ]


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evaluation_command(
    checkpoint: Path,
    card_checkpoint: Path,
    output: Path,
    *,
    seed_start: int,
    seed_count: int,
    range_name: str,
    map_margin: float,
    device: str,
    bootstrap_samples: int,
    record_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate-a20-map-value-policy.py"),
        "--checkpoint",
        str(checkpoint),
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
        str(map_margin),
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


def _run(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def _persist_unhandled_stage_error(status: str, error: BaseException | None = None) -> None:
    if _ACTIVE_STAGE is None:
        return
    state_path, state = _ACTIVE_STAGE
    state["status"] = status
    for step in reversed(state["steps"]):
        if step.get("status") == "running":
            step["status"] = status
            break
    if error is not None:
        state["error"] = {"type": type(error).__name__, "message": str(error)}
    _write_json(state_path, state)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _persist_unhandled_stage_error("interrupted")
        raise SystemExit(130)
    except Exception as error:
        _persist_unhandled_stage_error("failed_exception", error)
        raise
