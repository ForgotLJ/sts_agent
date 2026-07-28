from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any, Callable

from sts_env.differential import canonical_observation
from sts_env.env import StsEnv
from sts_env.search import BeliefSearchConfig, ParticleSearchPolicy
from sts_env.training.full_run_evaluation import (
    FullRunEvaluationSummary,
    evaluate_full_runs,
)
from sts_env.training.policies import HeuristicPolicy
from sts_env.training.recurrent_ppo import (
    HierarchicalRecurrentPolicy,
    RecurrentPPOTrainer,
)


def recurrent_policy_factory(
    trainer: RecurrentPPOTrainer,
    *,
    combat_policy: str,
    search_budget: int = 64,
    deterministic: bool = True,
) -> Callable[[int, int], HierarchicalRecurrentPolicy]:
    if combat_policy not in {"network", "heuristic", "belief-search"}:
        raise ValueError("unsupported M7 combat policy")
    if search_budget <= 0:
        raise ValueError("M7 search budget must be positive")

    def factory(policy_seed: int, _: int) -> HierarchicalRecurrentPolicy:
        combat_selector = None
        if combat_policy == "heuristic":
            heuristic = HeuristicPolicy()
            combat_selector = lambda environment: heuristic(environment.observation)
        elif combat_policy == "belief-search":
            search = ParticleSearchPolicy(
                BeliefSearchConfig(
                    simulations=max(search_budget, 64),
                    simulator_call_budget=search_budget,
                    max_depth=32,
                    rollout_depth=8,
                ),
                seed=policy_seed,
                rollout_policy=HeuristicPolicy(),
            )
            combat_selector = search.select
        return HierarchicalRecurrentPolicy(
            trainer,
            combat_selector=combat_selector,
            deterministic=deterministic,
        )

    return factory


def evaluate_m7_full_run(
    trainer: RecurrentPPOTrainer,
    environment_factory: Callable[[], StsEnv],
    seeds: tuple[int, ...],
    *,
    policy_seed: int,
    combat_policy: str,
    search_budget: int = 64,
    max_steps: int = 5_000,
    bootstrap_samples: int = 2_000,
) -> FullRunEvaluationSummary:
    return evaluate_full_runs(
        environment_factory,
        recurrent_policy_factory(
            trainer,
            combat_policy=combat_policy,
            search_budget=search_budget,
        ),
        seeds,
        policy_seed=policy_seed,
        max_steps=max_steps,
        bootstrap_samples=bootstrap_samples,
    )


def compact_full_run_summary(summary: FullRunEvaluationSummary) -> dict[str, Any]:
    payload = summary.to_dict()
    payload.pop("episodes")
    return payload


def evaluate_m7_curriculum_stage(
    trainer: RecurrentPPOTrainer,
    environment_factory: Callable[[], StsEnv],
    seeds: tuple[int, ...],
    *,
    policy_seed: int,
    max_steps: int,
    combat_policy: str = "heuristic",
    search_budget: int = 64,
    cycle_limit: int = 16,
) -> dict[str, float]:
    if not seeds or max_steps <= 0 or cycle_limit <= 0:
        raise ValueError("M7 curriculum validation configuration is invalid")
    policy_builder = recurrent_policy_factory(
        trainer,
        combat_policy=combat_policy,
        search_budget=search_budget,
    )
    completed = 0
    wins = 0
    floors: list[int] = []
    lengths: list[int] = []
    for episode_index, seed in enumerate(seeds):
        environment = environment_factory()
        observation, _ = environment.reset(seed=seed)
        policy = policy_builder(_split_seed(policy_seed, seed, episode_index), episode_index)
        repeated_decisions: dict[str, int] = {}
        episode_completed = False
        for step_index in range(max_steps):
            action = policy.select(environment)
            decision_key = json.dumps(
                {
                    "observation": canonical_observation(observation),
                    "action": action.to_dict(),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            repeated_decisions[decision_key] = repeated_decisions.get(decision_key, 0) + 1
            if repeated_decisions[decision_key] > cycle_limit:
                lengths.append(step_index)
                floors.append(observation.floor)
                break
            observation, _, terminated, truncated, info = environment.step(action)
            episode_completed = episode_completed or bool(
                info.get("curriculum_completed", False)
            )
            if terminated or truncated:
                raw_reward = float(info.get("raw_reward", 0.0))
                completed += int(episode_completed or raw_reward > 0)
                wins += int(terminated and raw_reward > 0)
                lengths.append(step_index + 1)
                floors.append(observation.floor)
                break
        else:
            lengths.append(max_steps)
            floors.append(observation.floor)
    return {
        "completion_rate": completed / len(seeds),
        "win_rate": wins / len(seeds),
        "mean_floor": statistics.mean(floors),
        "median_floor": statistics.median(floors),
        "mean_length": statistics.mean(lengths),
    }


def _split_seed(policy_seed: int, environment_seed: int, episode_index: int) -> int:
    digest = hashlib.sha256(
        f"m7:{policy_seed}:{environment_seed}:{episode_index}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little")
