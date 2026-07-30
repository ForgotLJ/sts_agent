from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.m7c_analysis import build_m7c_diagnostic_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a descriptive post-hoc diagnostic from frozen M7-C outputs."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-seed", type=int, default=17)
    parser.add_argument("--skip-trace-hashes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    report = build_m7c_diagnostic_report(
        args.run_root,
        run_seed=args.run_seed,
        verify_trace_hashes=not args.skip_trace_hashes,
    )
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "round_2_minus_round_0": report["round_2_minus_round_0"],
                "promotion": report["promotion"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
