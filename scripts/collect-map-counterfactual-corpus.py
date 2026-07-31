#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import LightspeedBackend, Phase, StsEnv
from sts_env.training.a20_online_card_ranking import (
    A20CloneValueCardRewardPolicy,
    load_online_value_model,
)
from sts_env.training.map_counterfactual import (
    MapCounterfactualConfig,
    evaluate_map_counterfactuals,
    map_candidate_actions,
)
from sts_env.training.map_action_protocol import (
    MAP_ACTION_COLLECTION_RANGE_NAMES,
    require_map_action_seed_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect redeterminized map-action counterfactual rollouts."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--seed-range-name", choices=sorted(MAP_ACTION_COLLECTION_RANGE_NAMES), required=True)
    parser.add_argument("--per-act", type=int, required=True)
    parser.add_argument("--acts", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--particles-per-action", type=int, default=2)
    parser.add_argument("--max-decisions-per-seed", type=int, default=1)
    parser.add_argument("--rollout-max-steps", type=int, default=5000)
    parser.add_argument("--override-margin", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neow-history", choices=("full", "limited", "skipped"), default="full")
    parser.add_argument(
        "--act1-boss-history",
        choices=("guardian_unseen", "hexaghost_unseen", "slime_boss_unseen", "all_seen"),
        default="all_seen",
    )
    parser.add_argument("--final-act-unlocked", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_action(policy: Any, environment: StsEnv, observation: Any, step_index: int):
    if hasattr(policy, "select"):
        return policy.select(environment)
    return policy(observation, step_index)


def main() -> int:
    args = parse_args()
    if args.seed_start < 0 or args.seed_count <= 0 or args.per_act <= 0:
        raise ValueError("counterfactual corpus seed and quota arguments must be positive")
    if args.max_decisions_per_seed <= 0 or args.override_margin < 0:
        raise ValueError("counterfactual corpus limits must be valid")
    if not args.acts or any(act not in {1, 2, 3} for act in args.acts):
        raise ValueError("counterfactual corpus acts must be selected from 1, 2, and 3")
    require_map_action_seed_range(
        args.seed_range_name,
        start=args.seed_start,
        count=args.seed_count,
        allowed_names=MAP_ACTION_COLLECTION_RANGE_NAMES,
    )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("counterfactual output directory must be empty")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    value_model = load_online_value_model(checkpoint, args.device)
    config = MapCounterfactualConfig(
        particles_per_action=args.particles_per_action,
        rollout_max_steps=args.rollout_max_steps,
    )
    counts = {act: 0 for act in sorted(set(args.acts))}
    errors: list[dict[str, Any]] = []
    records_path = args.output / "records.jsonl"
    started = time.perf_counter()
    completed = False
    with records_path.open("w", encoding="utf-8", newline="\n") as stream:
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            if all(count >= args.per_act for count in counts.values()):
                completed = True
                break
            environment = StsEnv(
                LightspeedBackend(
                    ascension=20,
                    neow_history=args.neow_history,
                    act1_boss_history=args.act1_boss_history,
                    final_act_unlocked=args.final_act_unlocked,
                )
            )
            behavior_policy = A20CloneValueCardRewardPolicy(
                value_model,
                override_margin=args.override_margin,
            )
            try:
                observation, _ = environment.reset(seed=seed)
                decisions_collected = 0
                for step_index in range(args.rollout_max_steps):
                    if observation.phase is Phase.TERMINAL:
                        break
                    map_actions = map_candidate_actions(observation)
                    if (
                        len(map_actions) >= 2
                        and observation.act in counts
                        and counts[observation.act] < args.per_act
                        and decisions_collected < args.max_decisions_per_seed
                    ):
                        behavior_action = select_action(
                            behavior_policy,
                            environment,
                            observation,
                            step_index,
                        )
                        record = evaluate_map_counterfactuals(
                            environment,
                            seed=seed,
                            decision_index=step_index,
                            behavior_action=behavior_action,
                            rollout_policy_factory=lambda: A20CloneValueCardRewardPolicy(
                                value_model,
                                override_margin=args.override_margin,
                            ),
                            config=config,
                        )
                        stream.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                        stream.flush()
                        counts[observation.act] += 1
                        decisions_collected += 1
                    action = select_action(behavior_policy, environment, observation, step_index)
                    observation, _, terminated, truncated, _ = environment.step(action)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("primary collection episode exceeded its maximum step count")
            except Exception as error:
                errors.append(
                    {
                        "seed": seed,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                break
    if not completed and not errors and all(count >= args.per_act for count in counts.values()):
        completed = True
    manifest = {
        "protocol": "map-counterfactual-rollouts",
        "schema_version": 1,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "ascension": 20,
        "seed_range": [args.seed_start, args.seed_start + args.seed_count - 1],
        "seed_range_name": args.seed_range_name,
        "neow_history": args.neow_history,
        "act1_boss_history": args.act1_boss_history,
        "final_act_unlocked": args.final_act_unlocked,
        "per_act": args.per_act,
        "acts": list(counts),
        "counts": {str(act): count for act, count in counts.items()},
        "particles_per_action": args.particles_per_action,
        "max_decisions_per_seed": args.max_decisions_per_seed,
        "rollout_max_steps": args.rollout_max_steps,
        "override_margin": args.override_margin,
        "records": {"path": records_path.name, "sha256": sha256_file(records_path)},
        "errors": errors,
        "complete": completed and not errors,
        "wall_seconds": time.perf_counter() - started,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
