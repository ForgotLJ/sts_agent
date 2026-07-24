from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import LightspeedBackend, StsEnv, record_episode
from sts_env.training import HeuristicPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public-information M6 teacher traces.")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed_start < 0 or args.seed_count <= 0 or args.max_steps <= 0:
        raise ValueError("teacher trace collection arguments are invalid")
    args.output.mkdir(parents=True, exist_ok=True)
    heuristic = HeuristicPolicy()
    started = time.perf_counter()
    collected = 0
    wins = 0
    act1_clears = 0
    act2_clears = 0
    errors: list[dict[str, object]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        try:
            environment = StsEnv(LightspeedBackend())
            trace = record_episode(
                environment,
                seed=seed,
                policy=heuristic,
                max_steps=args.max_steps,
            )
        except Exception as error:
            errors.append(
                {
                    "seed": seed,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue
        trace.write_jsonl(args.output / f"seed-{seed:08d}.jsonl")
        final_act = environment.observation.act
        final_reward = trace.steps[-1].reward if trace.steps else 0.0
        wins += int(final_reward > 0)
        act1_clears += int(final_act >= 2 or final_reward > 0)
        act2_clears += int(final_act >= 3 or final_reward > 0)
        collected += 1
    payload = {
        "policy": "improved public-information heuristic",
        "seed_range": [args.seed_start, args.seed_start + args.seed_count - 1],
        "requested": args.seed_count,
        "collected": collected,
        "wins": wins,
        "act1_clears": act1_clears,
        "act2_clears": act2_clears,
        "errors": errors,
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output / "collection-summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
