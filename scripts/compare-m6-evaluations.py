from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import paired_evaluation_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare paired M6 evaluations.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    comparison = paired_evaluation_difference(
        candidate,
        reference,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload = {
        "candidate": str(args.candidate),
        "candidate_summary": {
            key: value
            for key, value in candidate["summary"].items()
            if key != "episodes"
        },
        "reference": str(args.reference),
        "reference_summary": {
            key: value
            for key, value in reference["summary"].items()
            if key != "episodes"
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "paired_difference": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
