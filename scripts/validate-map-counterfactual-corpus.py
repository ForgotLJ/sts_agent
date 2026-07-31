#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.map_counterfactual import validate_map_counterfactual_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a map counterfactual rollout corpus.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = validate_map_counterfactual_corpus(args.input)
    if args.require_complete and not result["complete"]:
        result["valid"] = False
        result["errors"].append("corpus is not complete")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
