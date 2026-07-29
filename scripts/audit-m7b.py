from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SAFETY_METRICS = (
    "errors",
    "crashes",
    "illegal_actions",
    "recovery_failures",
    "timeouts",
    "cycles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit M7-B teacher imitation and end-to-end promotion gates."
    )
    parser.add_argument("--imitation-summary", type=Path, required=True)
    parser.add_argument("--end-to-end-summary", type=Path, required=True)
    parser.add_argument("--candidate-method", default="m7b")
    parser.add_argument("--baseline-method", default="m6-initial")
    parser.add_argument("--gate-seed-start", type=int, default=1_600_000)
    parser.add_argument("--gate-seed-count", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(
    imitation: dict[str, Any],
    end_to_end: dict[str, Any],
    *,
    candidate_method: str,
    baseline_method: str,
    gate_seed_start: int,
    gate_seed_count: int,
) -> dict[str, Any]:
    if gate_seed_start < 0 or gate_seed_count <= 0:
        raise ValueError("M7-B gate seed range is invalid")
    if imitation.get("protocol") != "m7b-imitation-comparison":
        raise ValueError("M7-B imitation summary has the wrong protocol")
    if end_to_end.get("protocol") != "m7":
        raise ValueError("M7-B end-to-end summary has the wrong protocol")
    if (
        imitation.get("candidate") != candidate_method
        or imitation.get("baseline") != baseline_method
    ):
        raise ValueError("M7-B imitation methods differ from the audit request")

    aggregate = dict(end_to_end.get("aggregate") or {})
    candidate = dict(aggregate.get(candidate_method) or {})
    baseline = dict(aggregate.get(baseline_method) or {})
    if not candidate or not baseline:
        raise ValueError("M7-B end-to-end summary lacks candidate or baseline")
    comparison_name = f"{candidate_method}_minus_{baseline_method}"
    comparison = dict(
        dict(end_to_end.get("paired_comparisons") or {}).get(comparison_name) or {}
    )
    if not comparison:
        raise ValueError("M7-B end-to-end summary lacks the paired comparison")

    expected_range = [gate_seed_start, gate_seed_start + gate_seed_count - 1]
    exact_seed_protocol = (
        comparison.get("seed_range") == expected_range
        and int(comparison.get("environment_seed_count", -1)) == gate_seed_count
        and int(candidate.get("unique_environment_seed_count", -1)) == gate_seed_count
        and int(baseline.get("unique_environment_seed_count", -1)) == gate_seed_count
    )
    safety = {}
    for method, payload in ((candidate_method, candidate), (baseline_method, baseline)):
        metrics = dict(payload.get("metrics") or {})
        safety[method] = {
            metric: float(dict(metrics.get(metric) or {}).get("mean", float("inf")))
            for metric in SAFETY_METRICS
        }
    safety_clear = all(
        value == 0.0
        for method_metrics in safety.values()
        for value in method_metrics.values()
    )

    metrics = dict(comparison.get("metrics") or {})
    floor = dict(metrics.get("final_floor") or {})
    act1 = dict(metrics.get("act1_clear") or {})
    floor_ci = list(floor.get("hierarchical_bootstrap_ci95") or ())
    if len(floor_ci) != 2:
        raise ValueError("M7-B end-to-end comparison lacks a floor confidence interval")
    floor_superiority = float(floor_ci[0]) > 0.0
    act1_nonregression = float(act1.get("mean_difference", float("-inf"))) >= 0.0
    teacher_gate = bool(imitation.get("complete")) and int(
        imitation.get("errors", -1)
    ) == 0
    end_to_end_gate = (
        exact_seed_protocol
        and safety_clear
        and floor_superiority
        and act1_nonregression
    )
    passed = teacher_gate and end_to_end_gate
    return {
        "protocol": "m7b-audit",
        "schema_version": 1,
        "candidate": candidate_method,
        "baseline": baseline_method,
        "gate_seed_range": expected_range,
        "teacher_action_gate": {
            "passed": teacher_gate,
            "phase_accuracy_nonregression": bool(
                imitation.get("phase_accuracy_nonregression")
            ),
            "overall_accuracy_improved": bool(
                imitation.get("overall_accuracy_improved")
            ),
            "overall_cross_entropy_improved": bool(
                imitation.get("overall_cross_entropy_improved")
            ),
        },
        "end_to_end_gate": {
            "passed": end_to_end_gate,
            "exact_seed_protocol": exact_seed_protocol,
            "safety_clear": safety_clear,
            "safety": safety,
            "final_floor_mean_difference": float(floor["mean_difference"]),
            "final_floor_ci95": [float(value) for value in floor_ci],
            "final_floor_superiority": floor_superiority,
            "act1_clear_mean_difference": float(act1["mean_difference"]),
            "act1_clear_nonregression": act1_nonregression,
        },
        "verdict": "PASS" if passed else "FAIL",
        "complete": passed,
        "errors": 0,
    }


def main() -> int:
    args = parse_args()
    result = audit(
        load_json(args.imitation_summary),
        load_json(args.end_to_end_summary),
        candidate_method=args.candidate_method,
        baseline_method=args.baseline_method,
        gate_seed_start=args.gate_seed_start,
        gate_seed_count=args.gate_seed_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
