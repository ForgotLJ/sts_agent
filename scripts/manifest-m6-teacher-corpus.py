from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.teacher_corpus import build_teacher_corpus_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and hash the formal M6 teacher corpus.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--act1-count", type=int, default=1024)
    parser.add_argument("--act2-count", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_teacher_corpus_manifest(
        args.corpus,
        {
            "act1_clear": args.act1_count,
            "act2_clear": args.act2_count,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
