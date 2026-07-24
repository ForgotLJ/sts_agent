from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import LightspeedBackend, StsEnv, record_episode
from sts_env.training import HeuristicPolicy, PrefixCorpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect self-generated M6 act prefixes.")
    parser.add_argument("--target-act", type=int, required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_act < 2 or min(args.target_count, args.seed_count, args.max_steps) <= 0:
        raise ValueError("prefix collection arguments are invalid")
    prefixes = []
    attempted = 0
    started = time.perf_counter()
    heuristic = HeuristicPolicy()
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        attempted += 1
        trace = record_episode(
            StsEnv(LightspeedBackend()),
            seed=seed,
            policy=heuristic,
            max_steps=args.max_steps,
        )
        try:
            corpus = PrefixCorpus.extract(
                lambda: StsEnv(LightspeedBackend()),
                (trace,),
                args.target_act,
            )
        except ValueError:
            continue
        prefixes.extend(corpus.traces)
        if len(prefixes) >= args.target_count:
            break
    if not prefixes:
        raise RuntimeError(f"no evaluated seed reached Act {args.target_act}")
    corpus = PrefixCorpus(tuple(prefixes[: args.target_count]), args.target_act)
    corpus.write(args.output)
    summary = {
        "target_act": args.target_act,
        "target_count": args.target_count,
        "collected": len(corpus.traces),
        "attempted_seeds": attempted,
        "seed_range": [args.seed_start, args.seed_start + attempted - 1],
        "policy": "improved public-information heuristic",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "collection-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
