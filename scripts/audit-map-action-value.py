#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.map_action_audit import audit_map_policy_evaluations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen paired map-policy evaluations and promotion criteria."
    )
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-checkpoint-sha256", required=True)
    parser.add_argument("--card-checkpoint-sha256", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--formal-range-name", default="map_value_formal")
    parser.add_argument("--replication-range-name", default="map_value_replication")
    parser.add_argument("--trained-acts", type=int, nargs="+")
    parser.add_argument("--trained-floor-range", type=int, nargs=2)
    parser.add_argument("--label-mode")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    result = audit_map_policy_evaluations(
        args.formal,
        args.replication,
        expected_map_checkpoint_sha256=args.map_checkpoint_sha256,
        expected_card_checkpoint_sha256=args.card_checkpoint_sha256,
        bootstrap_samples=args.bootstrap_samples,
        expected_formal_range_name=args.formal_range_name,
        expected_replication_range_name=args.replication_range_name,
        expected_trained_acts=(
            frozenset(args.trained_acts) if args.trained_acts is not None else None
        ),
        expected_trained_floor_range=(
            tuple(args.trained_floor_range) if args.trained_floor_range is not None else None
        ),
        expected_label_mode=args.label_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "replicated_improved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
