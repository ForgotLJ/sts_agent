#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import LightspeedBackend, StsEnv
from sts_env.training.a20_online_card_ranking import (
    A20CloneValueCardRewardPolicy,
    load_online_value_model,
)
from sts_env.training.full_run_evaluation import evaluate_full_runs
from sts_env.training.m6_reporting import paired_evaluation_difference
from sts_env.training.policies import HeuristicPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired Lightspeed evaluation for a clone-value Ironclad card policy."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--override-margin", type=float, default=0.05)
    parser.add_argument(
        "--neow-history",
        choices=("full", "limited", "skipped"),
        default="full",
    )
    parser.add_argument(
        "--act1-boss-history",
        choices=("guardian_unseen", "hexaghost_unseen", "slime_boss_unseen", "all_seen"),
        default="all_seen",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.seed_start < 0 or args.seed_count <= 0 or args.seed_start + args.seed_count > 2**64:
        raise ValueError("evaluation seeds must stay within [0, 2**64)")
    if args.max_steps <= 0 or args.bootstrap_samples <= 0 or args.override_margin < 0:
        raise ValueError("evaluation limits and margin must be valid")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    value_model = load_online_value_model(checkpoint, args.device)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    candidate_policies: list[A20CloneValueCardRewardPolicy] = []

    def environment_factory() -> StsEnv:
        return StsEnv(
            LightspeedBackend(
                ascension=20,
                neow_history=args.neow_history,
                act1_boss_history=args.act1_boss_history,
                final_act_unlocked=True,
            )
        )

    def candidate_policy_factory(_: int, __: int) -> A20CloneValueCardRewardPolicy:
        policy = A20CloneValueCardRewardPolicy(
            value_model,
            override_margin=args.override_margin,
        )
        candidate_policies.append(policy)
        return policy

    candidate_summary = evaluate_full_runs(
        environment_factory,
        candidate_policy_factory,
        seeds,
        policy_seed=args.policy_seed,
        max_steps=args.max_steps,
        bootstrap_samples=args.bootstrap_samples,
    )
    reference_summary = evaluate_full_runs(
        environment_factory,
        lambda _, __: HeuristicPolicy(),
        seeds,
        policy_seed=args.policy_seed,
        max_steps=args.max_steps,
        bootstrap_samples=args.bootstrap_samples,
    )
    candidate = {
        "method": "a20_clone_value_card_reward",
        "policy_seed": args.policy_seed,
        "summary": candidate_summary.to_dict(),
    }
    reference = {
        "method": "heuristic",
        "policy_seed": args.policy_seed,
        "summary": reference_summary.to_dict(),
    }
    telemetry_totals: dict[str, float] = {}
    for policy in candidate_policies:
        for key, value in policy.telemetry().items():
            telemetry_totals[key] = telemetry_totals.get(key, 0.0) + float(value)
    decision_count = telemetry_totals.get("card_reward_decisions", 0.0)
    override_count = telemetry_totals.get("overrides", 0.0)
    candidate_telemetry = {
        **telemetry_totals,
        "episodes": len(candidate_policies),
        "mean_candidates_per_card_reward": (
            telemetry_totals.get("candidate_actions_scored", 0.0) / decision_count
            if decision_count
            else 0.0
        ),
        "override_rate": override_count / decision_count if decision_count else 0.0,
        "mean_override_advantage": (
            telemetry_totals.get("override_advantage_total", 0.0) / override_count
            if override_count
            else 0.0
        ),
    }
    payload: dict[str, Any] = {
        "protocol": "a20-clone-value-card-reward-paired-lightspeed-evaluation",
        "schema_version": 1,
        "evaluation_scope": "diagnostic; not a promotion gate",
        "character": "IRONCLAD",
        "ascension": 20,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "seed_range": [args.seed_start, args.seed_start + args.seed_count - 1],
        "seed_count": args.seed_count,
        "policy_seed": args.policy_seed,
        "override_margin": args.override_margin,
        "max_steps": args.max_steps,
        "bootstrap_samples": args.bootstrap_samples,
        "neow_history": args.neow_history,
        "act1_boss_history": args.act1_boss_history,
        "candidate": candidate,
        "candidate_policy_telemetry": candidate_telemetry,
        "reference": reference,
        "paired_difference": paired_evaluation_difference(
            candidate,
            reference,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.policy_seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate": {
                    key: value
                    for key, value in candidate["summary"].items()
                    if key != "episodes"
                },
                "reference": {
                    key: value
                    for key, value in reference["summary"].items()
                    if key != "episodes"
                },
                "final_floor_difference": payload["paired_difference"]["metrics"]["final_floor"],
                "candidate_policy_telemetry": candidate_telemetry,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
