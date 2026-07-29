from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.m7c_inputs import verify_m7c_frozen_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify installed immutable M7-C inputs.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    verified = verify_m7c_frozen_inputs(args.root)
    print(
        json.dumps(
            {
                "root": verified["root"],
                "teacher_corpus_manifest": verified["teacher_corpus_manifest"],
                "initialization_checkpoint": verified["initialization_checkpoint"],
                "m6_baseline_checkpoint": verified["m6_baseline_checkpoint"],
                "files": len(dict(verified["files"])),
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
