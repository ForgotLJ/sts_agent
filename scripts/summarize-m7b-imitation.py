from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare M7-B candidate and baseline teacher-action agreement."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_evaluation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "m7b-imitation-evaluation":
        raise ValueError(f"unsupported M7-B imitation evaluation: {path}")
    return payload


def summarize(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if baseline["corpus_sha256"] != candidate["corpus_sha256"]:
        raise ValueError("M7-B imitation comparisons require the same corpus")
    baseline_metrics = dict(baseline["metrics"])
    candidate_metrics = dict(candidate["metrics"])
    baseline_phases = dict(baseline_metrics["phases"])
    candidate_phases = dict(candidate_metrics["phases"])
    if baseline_phases.keys() != candidate_phases.keys():
        raise ValueError("M7-B imitation phase sets differ")
    phases = {}
    for phase in sorted(baseline_phases):
        reference = dict(baseline_phases[phase])
        observed = dict(candidate_phases[phase])
        if int(reference["count"]) != int(observed["count"]):
            raise ValueError("M7-B imitation phase counts differ")
        phases[phase] = {
            "count": int(reference["count"]),
            "baseline_accuracy": float(reference["accuracy"]),
            "candidate_accuracy": float(observed["accuracy"]),
            "accuracy_difference": float(observed["accuracy"])
            - float(reference["accuracy"]),
            "baseline_cross_entropy": float(reference["cross_entropy"]),
            "candidate_cross_entropy": float(observed["cross_entropy"]),
            "cross_entropy_difference": float(observed["cross_entropy"])
            - float(reference["cross_entropy"]),
        }
    nonregression = all(
        phase["accuracy_difference"] >= 0 for phase in phases.values()
    )
    overall_accuracy_improved = (
        float(candidate_metrics["accuracy"]) > float(baseline_metrics["accuracy"])
    )
    overall_cross_entropy_improved = (
        float(candidate_metrics["cross_entropy"])
        < float(baseline_metrics["cross_entropy"])
    )
    passed = (
        nonregression
        and overall_accuracy_improved
        and overall_cross_entropy_improved
    )
    return {
        "protocol": "m7b-imitation-comparison",
        "schema_version": 1,
        "baseline": baseline["method"],
        "candidate": candidate["method"],
        "corpus_sha256": baseline["corpus_sha256"],
        "seed_range": baseline["seed_range"],
        "overall": {
            "baseline_accuracy": float(baseline_metrics["accuracy"]),
            "candidate_accuracy": float(candidate_metrics["accuracy"]),
            "accuracy_difference": float(candidate_metrics["accuracy"])
            - float(baseline_metrics["accuracy"]),
            "baseline_cross_entropy": float(baseline_metrics["cross_entropy"]),
            "candidate_cross_entropy": float(candidate_metrics["cross_entropy"]),
            "cross_entropy_difference": float(candidate_metrics["cross_entropy"])
            - float(baseline_metrics["cross_entropy"]),
        },
        "phases": phases,
        "phase_accuracy_nonregression": nonregression,
        "overall_accuracy_improved": overall_accuracy_improved,
        "overall_cross_entropy_improved": overall_cross_entropy_improved,
        "verdict": "PASS" if passed else "FAIL",
        "complete": passed,
        "errors": 0,
    }


def main() -> int:
    args = parse_args()
    summary = summarize(
        load_evaluation(args.baseline),
        load_evaluation(args.candidate),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
