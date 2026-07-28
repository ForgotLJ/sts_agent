from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import bootstrap_mean_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired M7 combat benchmarks.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--win-noninferiority-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--hp-noninferiority-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stable_seed(name: str) -> int:
    digest = hashlib.sha256(f"m7-combat-summary:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("combat summary bootstrap count must be positive")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("method") != "heuristic":
        raise ValueError("M7 combat baseline must be heuristic")
    baseline_records = {record["trace"]: record for record in baseline["records"]}
    comparisons: dict[str, Any] = {}
    extractors: dict[str, Callable[[dict[str, Any]], float]] = {
        "combat_win": lambda record: float(bool(record["won_combat"])),
        "hp_delta": lambda record: float(record["hp_delta"]),
        "decisions": lambda record: float(record["decisions"]),
        "simulator_calls": lambda record: float(record["simulator_calls"]),
        "wall_seconds": lambda record: float(record["wall_seconds"]),
    }
    for candidate_path in args.candidate:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        method = str(candidate["method"])
        candidate_records = {record["trace"]: record for record in candidate["records"]}
        if candidate_records.keys() != baseline_records.keys():
            raise ValueError(f"M7 combat seed set differs for {method}")
        metrics = {}
        for metric_index, (metric, extractor) in enumerate(extractors.items()):
            differences = [
                extractor(candidate_records[trace]) - extractor(baseline_records[trace])
                for trace in sorted(baseline_records)
            ]
            metrics[metric] = {
                "mean_difference": statistics.mean(differences),
                "bootstrap_ci95": bootstrap_mean_interval(
                    differences,
                    samples=args.bootstrap_samples,
                    seed=stable_seed(method) + metric_index,
                ),
            }
        win_lower = metrics["combat_win"]["bootstrap_ci95"][0]
        hp_lower = metrics["hp_delta"]["bootstrap_ci95"][0]
        passed = (
            int(candidate["summary"]["errors"]) == 0
            and win_lower >= -args.win_noninferiority_margin
            and hp_lower >= -args.hp_noninferiority_margin
        )
        comparisons[method] = {
            "episodes": len(candidate_records),
            "errors": int(candidate["summary"]["errors"]),
            "metrics": metrics,
            "noninferiority": {
                "passed": passed,
                "win_margin": args.win_noninferiority_margin,
                "hp_margin": args.hp_noninferiority_margin,
            },
        }
    payload = {
        "protocol": "m7",
        "schema_version": 1,
        "baseline": "heuristic",
        "bootstrap_samples": args.bootstrap_samples,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                method: comparison["noninferiority"]["passed"]
                for method, comparison in comparisons.items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
