#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.map_action_stage import (
    load_json,
    select_profile_margin,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a pre-specified map override margin from a record-only profile."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-checkpoint-sha256", required=True)
    parser.add_argument("--card-checkpoint-sha256", required=True)
    parser.add_argument("--quantile", choices=("p50", "p75", "p80", "p90", "p95"), default="p80")
    parser.add_argument("--label-mode")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"margin output already exists: {args.output}")
    profile = load_json(args.profile)
    result = select_profile_margin(
        profile,
        expected_map_checkpoint_sha256=args.map_checkpoint_sha256,
        expected_card_checkpoint_sha256=args.card_checkpoint_sha256,
        quantile=args.quantile,
        expected_label_mode=args.label_mode,
    )
    result["profile"] = {
        "path": str(args.profile.resolve()),
        "sha256": sha256_file(args.profile),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
