from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import summarize_m7_evaluations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize M7 evaluations.")
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--reference-method", default="heuristic")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluations = tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in args.evaluation
    )
    summary = summarize_m7_evaluations(
        evaluations,
        reference_method=args.reference_method,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "methods": sorted(summary["aggregate"]),
                "comparisons": sorted(summary["paired_comparisons"]),
                "warnings": len(summary["warnings"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
