from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select M6 teacher traces that reached a target act."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-act", type=int, required=True)
    parser.add_argument("--forbid-action-source", action="append", default=[])
    return parser.parse_args()


def inspect_trace(path: Path) -> tuple[int, int, set[str]]:
    seed = -1
    maximum_floor = -1
    action_sources: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "reset":
                seed = int(record["seed"])
                continue
            info = dict(record.get("info") or {})
            maximum_floor = max(maximum_floor, int(info.get("floor", -1)))
            source_id = (record.get("action") or {}).get("source_id")
            if source_id is not None:
                action_sources.add(str(source_id).lower())
    return seed, maximum_floor, action_sources


def main() -> int:
    args = parse_args()
    if args.target_act < 2 or not args.input.is_dir():
        raise ValueError("teacher selection arguments are invalid")
    minimum_floor = 17 * (args.target_act - 1)
    forbidden = {str(value).lower() for value in args.forbid_action_source}
    selected: list[dict[str, object]] = []
    rejected_forbidden: list[dict[str, object]] = []
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input.glob("*.jsonl")):
        seed, maximum_floor, action_sources = inspect_trace(path)
        if maximum_floor < minimum_floor:
            continue
        overlap = sorted(forbidden.intersection(action_sources))
        if overlap:
            rejected_forbidden.append(
                {
                    "path": str(path),
                    "seed": seed,
                    "forbidden_action_sources": overlap,
                }
            )
            continue
        destination = args.output / f"act{args.target_act - 1}-success-seed-{seed}.jsonl"
        shutil.copyfile(path, destination)
        selected.append(
            {
                "path": str(destination),
                "source": str(path),
                "seed": seed,
                "maximum_floor": maximum_floor,
            }
        )
    payload = {
        "target_act": args.target_act,
        "minimum_floor": minimum_floor,
        "input": str(args.input),
        "selected_count": len(selected),
        "selected": selected,
        "forbidden_action_sources": sorted(forbidden),
        "rejected_forbidden": rejected_forbidden,
    }
    (args.output / "selection-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not selected:
        raise RuntimeError("no teacher trace reached the target act")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
