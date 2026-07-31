#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.map_counterfactual_diagnostics import diagnose_map_counterfactual_corpus


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only diagnosis for a validated map counterfactual corpus."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-records", type=int, default=16)
    parser.add_argument("--min-contrasting-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")
    result = diagnose_map_counterfactual_corpus(
        args.input,
        min_records=args.min_records,
        min_contrasting_fraction=args.min_contrasting_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["scale_gate"]["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
