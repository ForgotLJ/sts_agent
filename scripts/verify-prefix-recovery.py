from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import (
    LightspeedBackend,
    StsEnv,
    observation_digest,
    record_episode,
    replay_trace,
)


@dataclass(frozen=True, slots=True)
class RecoveryAuditSummary:
    complete: bool
    errors: int
    checks: int
    episodes: int
    seed_start: int
    policy_seed: int
    recorded_steps: int
    replayed_steps: int
    max_prefix: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify deterministic seed + action-prefix recovery."
    )
    parser.add_argument("--checks", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=300000)
    parser.add_argument("--policy-seed", type=int, default=20260716)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m6_prefix_recovery" / "summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.checks <= 0 or args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("checks, episodes, and max-steps must be positive")
    started = time.perf_counter()
    traces = []
    recorded_steps = 0
    for episode_index in range(args.episodes):
        seed = args.seed_start + episode_index
        policy_random = random.Random(args.policy_seed + seed)
        trace = record_episode(
            StsEnv(LightspeedBackend()),
            seed=seed,
            policy=lambda observation, source=policy_random: source.choice(
                observation.legal_actions
            ),
            max_steps=args.max_steps,
        )
        traces.append(trace)
        recorded_steps += len(trace.steps)

    prefix_random = random.Random(args.policy_seed)
    replayed_steps = 0
    max_prefix = 0
    for check_index in range(args.checks):
        trace = traces[check_index % len(traces)]
        prefix_length = prefix_random.randrange(len(trace.steps) + 1)
        prefix = trace.prefix(prefix_length)
        observation = replay_trace(StsEnv(LightspeedBackend()), prefix)
        expected_digest = (
            prefix.initial_observation_digest
            if prefix_length == 0
            else prefix.steps[-1].observation_digest
        )
        actual_digest = observation_digest(observation)
        if actual_digest != expected_digest:
            raise AssertionError(
                f"prefix recovery mismatch at check {check_index}: "
                f"seed={trace.seed} prefix={prefix_length}"
            )
        replayed_steps += prefix_length
        max_prefix = max(max_prefix, prefix_length)

    summary = RecoveryAuditSummary(
        complete=True,
        errors=0,
        checks=args.checks,
        episodes=args.episodes,
        seed_start=args.seed_start,
        policy_seed=args.policy_seed,
        recorded_steps=recorded_steps,
        replayed_steps=replayed_steps,
        max_prefix=max_prefix,
        elapsed_seconds=time.perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
