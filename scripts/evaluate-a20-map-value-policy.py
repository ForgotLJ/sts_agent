#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
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
from sts_env.training.map_action_value import (
    A20MapActionValuePolicy,
    load_map_action_value_model,
    sha256_file,
)
from sts_env.training.map_action_protocol import (
    MAP_ACTION_EVALUATION_RANGE_NAMES,
    require_map_action_seed_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired A20 Ironclad evaluation for a map-action value policy."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--card-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--seed-range-name", choices=sorted(MAP_ACTION_EVALUATION_RANGE_NAMES), required=True)
    parser.add_argument("--override-margin", type=float, required=True)
    parser.add_argument("--card-override-margin", type=float, required=True)
    parser.add_argument("--policy-seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--record-only", action="store_true")
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


def _aggregate_telemetry(policies: list[Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for policy in policies:
        for key, value in policy.telemetry().items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("p50", "p75", "p80", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p80": percentile(0.80),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _assert_record_only_identity(candidate: Any, reference: Any) -> None:
    fields = (
        "seed",
        "outcome",
        "won",
        "final_act",
        "final_floor",
        "final_hp",
        "max_hp",
        "environment_return",
        "decisions",
        "error",
        "error_category",
    )
    if len(candidate.episodes) != len(reference.episodes):
        raise AssertionError("record-only candidate and reference episode counts differ")
    for candidate_episode, reference_episode in zip(candidate.episodes, reference.episodes):
        if any(
            getattr(candidate_episode, field) != getattr(reference_episode, field)
            for field in fields
        ):
            raise AssertionError(
                f"record-only policy changed the episode for seed {candidate_episode.seed}"
            )


def _trained_acts(checkpoint_metadata: dict[str, Any]) -> frozenset[int] | None:
    metadata = dict(checkpoint_metadata.get("metadata") or {})
    values = metadata.get("trained_acts")
    if values is None:
        return None
    try:
        acts = frozenset(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"map checkpoint trained acts are invalid: {error}") from error
    if not acts or not acts.issubset({1, 2, 3}):
        raise ValueError("map checkpoint trained acts must be a non-empty subset of 1, 2, and 3")
    return acts


def main() -> int:
    args = parse_args()
    if args.seed_start < 0 or args.seed_count <= 0 or args.seed_start + args.seed_count > 2**64:
        raise ValueError("evaluation seeds must stay within [0, 2**64)")
    if (
        args.override_margin < 0
        or args.card_override_margin < 0
        or args.max_steps <= 0
        or args.bootstrap_samples <= 0
    ):
        raise ValueError("evaluation margins and limits must be valid")
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    require_map_action_seed_range(
        args.seed_range_name,
        start=args.seed_start,
        count=args.seed_count,
        allowed_names=MAP_ACTION_EVALUATION_RANGE_NAMES,
    )
    checkpoint = args.checkpoint.resolve()
    card_checkpoint = args.card_checkpoint.resolve()
    if not checkpoint.is_file() or not card_checkpoint.is_file():
        raise FileNotFoundError("map and card checkpoints must both exist")
    map_model, map_encoder, map_checkpoint_metadata = load_map_action_value_model(
        checkpoint,
        args.device,
    )
    trained_acts = _trained_acts(map_checkpoint_metadata)
    card_model = load_online_value_model(card_checkpoint, args.device)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    candidate_map_policies: list[A20MapActionValuePolicy] = []
    candidate_card_policies: list[A20CloneValueCardRewardPolicy] = []
    reference_card_policies: list[A20CloneValueCardRewardPolicy] = []

    def environment_factory() -> StsEnv:
        return StsEnv(
            LightspeedBackend(
                ascension=20,
                neow_history=args.neow_history,
                act1_boss_history=args.act1_boss_history,
                final_act_unlocked=True,
            )
        )

    def candidate_factory(_: int, __: int) -> A20MapActionValuePolicy:
        card_policy = A20CloneValueCardRewardPolicy(
            card_model,
            override_margin=args.card_override_margin,
        )
        map_policy = A20MapActionValuePolicy(
            map_model,
            map_encoder,
            fallback=card_policy,
            override_margin=args.override_margin,
            record_only=args.record_only,
            allowed_acts=trained_acts,
        )
        candidate_card_policies.append(card_policy)
        candidate_map_policies.append(map_policy)
        return map_policy

    def reference_factory(_: int, __: int) -> A20CloneValueCardRewardPolicy:
        card_policy = A20CloneValueCardRewardPolicy(
            card_model,
            override_margin=args.card_override_margin,
        )
        reference_card_policies.append(card_policy)
        return card_policy

    candidate_summary = evaluate_full_runs(
        environment_factory,
        candidate_factory,
        seeds,
        policy_seed=args.policy_seed,
        max_steps=args.max_steps,
        bootstrap_samples=args.bootstrap_samples,
    )
    reference_summary = evaluate_full_runs(
        environment_factory,
        reference_factory,
        seeds,
        policy_seed=args.policy_seed,
        max_steps=args.max_steps,
        bootstrap_samples=args.bootstrap_samples,
    )
    if args.record_only:
        _assert_record_only_identity(candidate_summary, reference_summary)
    map_telemetry = _aggregate_telemetry(candidate_map_policies)
    map_advantages = [
        advantage
        for policy in candidate_map_policies
        for advantage in policy.best_advantages
    ]
    map_decisions = map_telemetry.get("map_decisions", 0.0)
    map_overrides = map_telemetry.get("overrides", 0.0)
    candidate = {
        "method": "a20-map-action-value",
        "policy_seed": args.policy_seed,
        "summary": candidate_summary.to_dict(),
    }
    reference = {
        "method": "a20-clone-value-card-reward",
        "policy_seed": args.policy_seed,
        "summary": reference_summary.to_dict(),
    }
    payload: dict[str, Any] = {
        "protocol": "a20-map-action-value-paired-lightspeed-evaluation",
        "schema_version": 1,
        "evaluation_scope": "record-only profile" if args.record_only else "diagnostic; not a promotion gate",
        "character": "IRONCLAD",
        "ascension": 20,
        "map_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "card_checkpoint": {"path": str(card_checkpoint), "sha256": sha256_file(card_checkpoint)},
        "map_checkpoint_metadata": map_checkpoint_metadata,
        "map_policy_trained_acts": sorted(trained_acts) if trained_acts is not None else None,
        "seed_range": [args.seed_start, args.seed_start + args.seed_count - 1],
        "seed_range_name": args.seed_range_name,
        "seed_count": args.seed_count,
        "policy_seed": args.policy_seed,
        "override_margin": args.override_margin,
        "card_override_margin": args.card_override_margin,
        "record_only": args.record_only,
        "max_steps": args.max_steps,
        "bootstrap_samples": args.bootstrap_samples,
        "neow_history": args.neow_history,
        "act1_boss_history": args.act1_boss_history,
        "candidate": candidate,
        "reference": reference,
        "candidate_map_telemetry": {
            **map_telemetry,
            "episodes": len(candidate_map_policies),
            "override_rate": map_overrides / map_decisions if map_decisions else 0.0,
            "mean_override_advantage": (
                map_telemetry.get("override_advantage_total", 0.0) / map_overrides
                if map_overrides
                else 0.0
            ),
            "mean_best_advantage": statistics.fmean(map_advantages) if map_advantages else 0.0,
            "best_advantage_quantiles": _quantiles(map_advantages),
        },
        "candidate_card_telemetry": _aggregate_telemetry(candidate_card_policies),
        "reference_card_telemetry": _aggregate_telemetry(reference_card_policies),
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
                "candidate_map_telemetry": payload["candidate_map_telemetry"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
