from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from sts_env.trace import EpisodeTrace
from sts_env.training.m7b_distillation import M7B_SUPERVISED_PHASES
from sts_env.training.m7c_dagger import (
    m7c_corpus_trace_paths,
    m7c_dagger_labels,
    sha256_file,
    verify_m7c_corpus_manifest,
)


M7C_DIAGNOSTIC_PROTOCOL = "m7c-posthoc-diagnostic"


def summarize_m7c_behavior_associations(
    traces: Iterable[EpisodeTrace],
) -> dict[str, Any]:
    trace_count = 0
    traces_without_student_decisions = 0
    final_floors_with_error: list[float] = []
    final_floors_without_error: list[float] = []
    decision_totals = _decision_totals()
    phase_totals = {
        phase.value: _decision_totals() for phase in M7B_SUPERVISED_PHASES
    }
    floor_totals: dict[str, dict[str, float | int]] = {}
    first_error_phases: dict[str, dict[str, Any]] = {}

    for trace in traces:
        trace_count += 1
        metadata = dict(trace.metadata or {})
        final_floor = int(metadata.get("final_floor", -1))
        if final_floor < 0:
            raise ValueError("M7-C trace lacks a valid final floor")
        first_error_phase: str | None = None
        trace_has_error = False
        trace_student_decisions = 0
        for label in m7c_dagger_labels(trace):
            if (
                label.phase not in M7B_SUPERVISED_PHASES
                or label.legal_action_count <= 1
                or label.teacher_mixed
            ):
                continue
            trace_student_decisions += 1
            correct = trace.steps[label.step_index].action == label.teacher_action
            _accumulate_decision(decision_totals, label.policy_entropy, label.policy_margin, correct)
            _accumulate_decision(
                phase_totals[label.phase.value],
                label.policy_entropy,
                label.policy_margin,
                correct,
            )
            floor_entry = floor_totals.setdefault(str(label.floor), _decision_totals())
            _accumulate_decision(
                floor_entry,
                label.policy_entropy,
                label.policy_margin,
                correct,
            )
            if not correct and first_error_phase is None:
                trace_has_error = True
                first_error_phase = label.phase.value
                phase_entry = first_error_phases.setdefault(
                    first_error_phase,
                    {"count": 0, "final_floors": []},
                )
                phase_entry["count"] = int(phase_entry["count"]) + 1
                phase_entry["final_floors"].append(float(final_floor))
        if trace_student_decisions == 0:
            traces_without_student_decisions += 1
        elif trace_has_error:
            final_floors_with_error.append(float(final_floor))
        else:
            final_floors_without_error.append(float(final_floor))

    if trace_count == 0:
        raise ValueError("M7-C behavior diagnostic requires traces")

    with_error_mean = _mean_or_none(final_floors_with_error)
    without_error_mean = _mean_or_none(final_floors_without_error)
    return {
        "trace_count": trace_count,
        "student_decisions": _finalize_decisions(decision_totals),
        "phases": {
            phase: _finalize_decisions(totals)
            for phase, totals in sorted(phase_totals.items())
        },
        "floors": {
            floor: _finalize_decisions(totals)
            for floor, totals in sorted(
                floor_totals.items(), key=lambda item: int(item[0])
            )
        },
        "episodes": {
            "with_student_decisions": (
                len(final_floors_with_error) + len(final_floors_without_error)
            ),
            "without_student_decisions": traces_without_student_decisions,
            "with_any_disagreement": len(final_floors_with_error),
            "without_disagreement": len(final_floors_without_error),
            "mean_final_floor_with_any_disagreement": with_error_mean,
            "mean_final_floor_without_disagreement": without_error_mean,
            "observed_floor_association": (
                None
                if with_error_mean is None or without_error_mean is None
                else with_error_mean - without_error_mean
            ),
        },
        "first_disagreement_phases": {
            phase: {
                "count": int(entry["count"]),
                "mean_final_floor": statistics.fmean(entry["final_floors"]),
            }
            for phase, entry in sorted(first_error_phases.items())
        },
        "causal_regret_estimated": False,
    }


def build_m7c_diagnostic_report(
    run_root: str | Path,
    *,
    run_seed: int = 17,
    verify_trace_hashes: bool = True,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    if run_seed < 0 or not root.is_dir():
        raise ValueError("M7-C diagnostic run root is invalid")

    rounds = []
    input_hashes: dict[str, str] = {}
    for round_index in range(3):
        dagger_root = root / f"dagger-round-{round_index}"
        on_policy_root = root / f"on-policy-round-{round_index}"
        dagger_manifest_path = dagger_root / "manifest.json"
        on_policy_manifest_path = on_policy_root / "manifest.json"
        dagger_manifest = verify_m7c_corpus_manifest(
            dagger_manifest_path,
            expected_round_index=round_index,
            verify_file_hashes=verify_trace_hashes,
        )
        on_policy_manifest = verify_m7c_corpus_manifest(
            on_policy_manifest_path,
            expected_round_index=round_index,
            verify_file_hashes=verify_trace_hashes,
        )
        dagger_diagnostic_path = dagger_root / "on-policy-diagnostic.json"
        on_policy_diagnostic_path = on_policy_root / "on-policy-diagnostic.json"
        dagger_diagnostic = _load_diagnostic(
            dagger_diagnostic_path,
            expected_corpus_sha256=str(dagger_manifest["aggregate_sha256"]),
        )
        on_policy_diagnostic = _load_diagnostic(
            on_policy_diagnostic_path,
            expected_corpus_sha256=str(on_policy_manifest["aggregate_sha256"]),
        )
        training_root = root / "training" / f"seed-{run_seed}" / f"round-{round_index}"
        training_manifest_path = training_root / "manifest.json"
        best_validation_path = training_root / "best-validation.json"
        metrics_path = training_root / "metrics.jsonl"
        best_checkpoint_path = training_root / "best-evaluation-checkpoint.pt"
        training_manifest = _load_json(training_manifest_path)
        if (
            training_manifest.get("protocol") != "m7c-dagger"
            or int(training_manifest.get("run_seed", -1)) != run_seed
            or int(training_manifest.get("round_index", -1)) != round_index
        ):
            raise ValueError("M7-C training manifest differs from the diagnostic request")
        best_validation = _load_json(best_validation_path)
        if best_validation.get("type") != "validation" or not best_checkpoint_path.is_file():
            raise ValueError("M7-C round lacks a selected validation checkpoint")
        validations = tuple(
            record
            for record in _load_jsonl(metrics_path)
            if record.get("type") == "validation"
        )
        if not validations:
            raise ValueError("M7-C round contains no validation metrics")

        behavior = summarize_m7c_behavior_associations(
            EpisodeTrace.read_jsonl(path)
            for path in m7c_corpus_trace_paths(on_policy_manifest)
        )
        rounds.append(
            {
                "round_index": round_index,
                "dagger": _corpus_summary(dagger_manifest, dagger_diagnostic),
                "on_policy": _corpus_summary(
                    on_policy_manifest,
                    on_policy_diagnostic,
                ),
                "on_policy_behavior_associations": behavior,
                "training": {
                    "validation_count": len(validations),
                    "selected_epoch": int(best_validation["epoch"]),
                    "selection_key": list(best_validation["selection_key"]),
                    "on_policy": dict(best_validation["on_policy"]),
                    "teacher_anchor": dict(best_validation["teacher_anchor"]),
                    "best_checkpoint": str(best_checkpoint_path),
                    "best_checkpoint_sha256": sha256_file(best_checkpoint_path),
                },
            }
        )
        for path in (
            dagger_manifest_path,
            on_policy_manifest_path,
            dagger_diagnostic_path,
            on_policy_diagnostic_path,
            training_manifest_path,
            best_validation_path,
            metrics_path,
        ):
            input_hashes[str(path)] = sha256_file(path)

    promotion_root = root / "promotion"
    promotion_paths = {
        name: promotion_root / name
        for name in (
            "m7c-dagger.json",
            "m6-initial.json",
            "heuristic.json",
            "summary.json",
            "audit.json",
        )
    }
    promotion = {name: _load_json(path) for name, path in promotion_paths.items()}
    summary_comparisons = dict(
        promotion["summary.json"].get("paired_comparisons") or {}
    )
    audit = promotion["audit.json"]
    for path in promotion_paths.values():
        input_hashes[str(path)] = sha256_file(path)

    first_round = rounds[0]
    last_round = rounds[-1]
    first_behavior = first_round["on_policy_behavior_associations"]
    last_behavior = last_round["on_policy_behavior_associations"]
    first_agreement = first_behavior["student_decisions"]["agreement"]
    last_agreement = last_behavior["student_decisions"]["agreement"]
    if first_agreement is None or last_agreement is None:
        agreement_change = None
    else:
        agreement_change = float(last_agreement) - float(first_agreement)

    return {
        "protocol": M7C_DIAGNOSTIC_PROTOCOL,
        "schema_version": 1,
        "descriptive_only": True,
        "model_selection_allowed": False,
        "causal_regret_estimated": False,
        "run_root": str(root),
        "run_seed": run_seed,
        "trace_hashes_verified": verify_trace_hashes,
        "rounds": rounds,
        "round_2_minus_round_0": {
            "student_agreement": agreement_change,
            "mean_final_floor": (
                float(last_round["on_policy"]["mean_final_floor"])
                - float(first_round["on_policy"]["mean_final_floor"])
            ),
        },
        "promotion": {
            "audit_verdict": str(audit.get("verdict") or ""),
            "audit_complete": bool(audit.get("complete")),
            "safety_clear": bool(audit.get("safety_clear")),
            "summary_has_m6_comparison": (
                "m7c-dagger_minus_m6-initial" in summary_comparisons
            ),
            "summary_has_heuristic_comparison": (
                "m7c-dagger_minus_heuristic" in summary_comparisons
            ),
            "m7c_minus_m6": dict(audit.get("m7c_minus_m6") or {}),
            "m7c_minus_heuristic": dict(
                audit.get("m7c_minus_heuristic") or {}
            ),
        },
        "input_sha256": dict(sorted(input_hashes.items())),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSONL objects: {path}")
            records.append(payload)
    return tuple(records)


def _load_diagnostic(
    path: Path,
    *,
    expected_corpus_sha256: str,
) -> dict[str, Any]:
    payload = _load_json(path)
    if (
        payload.get("protocol") != "m7c-on-policy-diagnostic"
        or str(payload.get("corpus_sha256") or "") != expected_corpus_sha256
    ):
        raise ValueError("M7-C on-policy diagnostic differs from its corpus")
    return payload


def _corpus_summary(
    manifest: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed_range": list(manifest["seed_range"]),
        "trace_count": int(manifest["trace_count"]),
        "aggregate_sha256": str(manifest["aggregate_sha256"]),
        "teacher_mix_probability": float(manifest["teacher_mix_probability"]),
        "wins": int(manifest["wins"]),
        "mean_final_floor": float(manifest["mean_final_floor"]),
        "horizon_truncations": int(manifest["horizon_truncations"]),
        "phase_supervision_counts": dict(manifest["phase_supervision_counts"]),
        "student_behavior": dict(diagnostic["student_behavior"]),
        "phases": dict(diagnostic["phases"]),
        "floors": dict(diagnostic["floors"]),
    }


def _decision_totals() -> dict[str, float | int]:
    return {
        "count": 0,
        "correct": 0,
        "correct_entropy": 0.0,
        "incorrect_entropy": 0.0,
        "correct_margin": 0.0,
        "incorrect_margin": 0.0,
    }


def _accumulate_decision(
    totals: dict[str, float | int],
    entropy: float,
    margin: float,
    correct: bool,
) -> None:
    totals["count"] = int(totals["count"]) + 1
    totals["correct"] = int(totals["correct"]) + int(correct)
    prefix = "correct" if correct else "incorrect"
    totals[f"{prefix}_entropy"] = float(totals[f"{prefix}_entropy"]) + entropy
    totals[f"{prefix}_margin"] = float(totals[f"{prefix}_margin"]) + margin


def _finalize_decisions(
    totals: dict[str, float | int],
) -> dict[str, float | int | None]:
    count = int(totals["count"])
    correct = int(totals["correct"])
    incorrect = count - correct
    return {
        "count": count,
        "correct": correct,
        "incorrect": incorrect,
        "agreement": None if count == 0 else correct / count,
        "mean_entropy_correct": (
            None if correct == 0 else float(totals["correct_entropy"]) / correct
        ),
        "mean_entropy_incorrect": (
            None if incorrect == 0 else float(totals["incorrect_entropy"]) / incorrect
        ),
        "mean_margin_correct": (
            None if correct == 0 else float(totals["correct_margin"]) / correct
        ),
        "mean_margin_incorrect": (
            None if incorrect == 0 else float(totals["incorrect_margin"]) / incorrect
        ),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)
