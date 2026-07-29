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
        description="Audit M7-C DAgger promotion and formal paired gates."
    )
    parser.add_argument("--end-to-end-summary", type=Path, required=True)
    parser.add_argument("--candidate-method", default="m7c-dagger")
    parser.add_argument("--m6-baseline-method", default="m6-initial")
    parser.add_argument("--heuristic-baseline-method", default="heuristic")
    parser.add_argument("--gate-seed-start", type=int, required=True)
    parser.add_argument("--gate-seed-count", type=int, required=True)
    parser.add_argument("--heuristic-floor-noninferiority-margin", type=float, default=1.0)
    parser.add_argument("--heuristic-act1-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(
    end_to_end: dict[str, Any],
    *,
    candidate_method: str,
    m6_baseline_method: str,
    heuristic_baseline_method: str,
    gate_seed_start: int,
    gate_seed_count: int,
    heuristic_floor_noninferiority_margin: float = 1.0,
    heuristic_act1_noninferiority_margin: float = 0.02,
) -> dict[str, Any]:
    if (
        gate_seed_start < 0
        or gate_seed_count <= 0
        or heuristic_floor_noninferiority_margin < 0.0
        or heuristic_act1_noninferiority_margin < 0.0
    ):
        raise ValueError("M7-C gate seed range is invalid")
    if end_to_end.get("protocol") != "m7":
        raise ValueError("M7-C end-to-end summary has the wrong protocol")
    aggregate = dict(end_to_end.get("aggregate") or {})
    methods = {
        candidate_method: dict(aggregate.get(candidate_method) or {}),
        m6_baseline_method: dict(aggregate.get(m6_baseline_method) or {}),
        heuristic_baseline_method: dict(aggregate.get(heuristic_baseline_method) or {}),
    }
    if any(not method for method in methods.values()):
        raise ValueError("M7-C summary lacks a candidate or required baseline")
    comparisons = dict(end_to_end.get("paired_comparisons") or {})
    m6_comparison = dict(
        comparisons.get(f"{candidate_method}_minus_{m6_baseline_method}") or {}
    )
    heuristic_comparison = dict(
        comparisons.get(f"{candidate_method}_minus_{heuristic_baseline_method}")
        or {}
    )
    if not m6_comparison or not heuristic_comparison:
        raise ValueError("M7-C summary lacks required paired comparisons")
    expected_range = [gate_seed_start, gate_seed_start + gate_seed_count - 1]

    def exact_seed_protocol(comparison: dict[str, Any]) -> bool:
        return (
            comparison.get("seed_range") == expected_range
            and int(comparison.get("environment_seed_count", -1)) == gate_seed_count
            and all(
                int(method.get("unique_environment_seed_count", -1))
                == gate_seed_count
                for method in methods.values()
            )
        )

    safety = {
        name: {
            metric: float(
                dict(dict(payload.get("metrics") or {}).get(metric) or {}).get(
                    "mean",
                    float("inf"),
                )
            )
            for metric in SAFETY_METRICS
        }
        for name, payload in methods.items()
    }
    safety_clear = all(
        value == 0.0
        for method_metrics in safety.values()
        for value in method_metrics.values()
    )

    def comparison_gate(
        comparison: dict[str, Any],
        *,
        floor_margin: float,
        act1_margin: float,
    ) -> dict[str, Any]:
        metrics = dict(comparison.get("metrics") or {})
        floor = dict(metrics.get("final_floor") or {})
        act1 = dict(metrics.get("act1_clear") or {})
        interval = list(floor.get("hierarchical_bootstrap_ci95") or ())
        if len(interval) != 2:
            raise ValueError("M7-C comparison lacks a floor confidence interval")
        return {
            "exact_seed_protocol": exact_seed_protocol(comparison),
            "final_floor_mean_difference": float(floor["mean_difference"]),
            "final_floor_ci95": [float(value) for value in interval],
            "final_floor_lower_bound_requirement": -floor_margin,
            "final_floor_requirement_met": float(interval[0]) > -floor_margin,
            "act1_clear_mean_difference": float(act1["mean_difference"]),
            "act1_clear_lower_bound_requirement": -act1_margin,
            "act1_clear_requirement_met": float(act1["mean_difference"]) >= -act1_margin,
        }

    m6_gate = comparison_gate(
        m6_comparison,
        floor_margin=0.0,
        act1_margin=0.0,
    )
    heuristic_gate = comparison_gate(
        heuristic_comparison,
        floor_margin=heuristic_floor_noninferiority_margin,
        act1_margin=heuristic_act1_noninferiority_margin,
    )
    passed = (
        safety_clear
        and m6_gate["exact_seed_protocol"]
        and heuristic_gate["exact_seed_protocol"]
        and m6_gate["final_floor_requirement_met"]
        and m6_gate["act1_clear_requirement_met"]
        and heuristic_gate["final_floor_requirement_met"]
        and heuristic_gate["act1_clear_requirement_met"]
    )
    return {
        "protocol": "m7c-audit",
        "schema_version": 1,
        "candidate": candidate_method,
        "m6_baseline": m6_baseline_method,
        "heuristic_baseline": heuristic_baseline_method,
        "gate_seed_range": expected_range,
        "heuristic_floor_noninferiority_margin": heuristic_floor_noninferiority_margin,
        "heuristic_act1_noninferiority_margin": heuristic_act1_noninferiority_margin,
        "safety_clear": safety_clear,
        "safety": safety,
        "m7c_minus_m6": m6_gate,
        "m7c_minus_heuristic": heuristic_gate,
        "verdict": "PASS" if passed else "FAIL",
        "complete": passed,
        "errors": 0,
    }


def main() -> int:
    args = parse_args()
    result = audit(
        load_json(args.end_to_end_summary),
        candidate_method=args.candidate_method,
        m6_baseline_method=args.m6_baseline_method,
        heuristic_baseline_method=args.heuristic_baseline_method,
        gate_seed_start=args.gate_seed_start,
        gate_seed_count=args.gate_seed_count,
        heuristic_floor_noninferiority_margin=args.heuristic_floor_noninferiority_margin,
        heuristic_act1_noninferiority_margin=args.heuristic_act1_noninferiority_margin,
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
