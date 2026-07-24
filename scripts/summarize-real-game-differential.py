from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize real-game differential traces.")
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = []
    for path in args.trace:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    phases = Counter(str(record.get("reference_phase")) for record in records)
    differences = [
        difference
        for record in records
        for difference in record.get("differences", [])
    ]
    unallowed = [difference for difference in differences if not difference.get("allowed")]
    payload = {
        "complete": bool(records) and not unallowed,
        "errors": len(unallowed),
        "records": len(records),
        "traces": [str(path.resolve()) for path in args.trace],
        "phase_counts": dict(sorted(phases.items())),
        "differences": len(differences),
        "allowed_differences": len(differences) - len(unallowed),
        "unallowed_differences": unallowed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
