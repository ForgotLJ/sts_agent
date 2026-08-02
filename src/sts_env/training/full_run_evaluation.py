from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
import statistics
import time
from typing import Any, Callable

from sts_env.differential import canonical_observation
from sts_env.env import StsEnv
from sts_env.types import Action, Observation


@dataclass(frozen=True, slots=True)
class FullRunEvaluationEpisode:
    seed: int
    policy_seed: int
    outcome: str
    won: bool
    final_act: int
    final_floor: int
    final_hp: int
    max_hp: int
    environment_return: float
    proxy_score: float
    game_score: int | None
    decisions: int
    simulator_calls: int
    wall_seconds: float
    error: str = ""
    error_category: str = ""


@dataclass(frozen=True, slots=True)
class FullRunEvaluationSummary:
    episodes: tuple[FullRunEvaluationEpisode, ...]
    win_rate: float
    win_rate_ci95: tuple[float, float]
    act1_clear_rate: float
    act2_clear_rate: float
    act3_clear_rate: float
    mean_floor: float
    median_floor: float
    mean_floor_ci95: tuple[float, float]
    mean_final_hp: float
    mean_proxy_score: float
    mean_decisions: float
    total_simulator_calls: int
    total_wall_seconds: float
    errors: int
    crashes: int
    illegal_actions: int
    recovery_failures: int
    truncations: int
    timeouts: int
    cycles: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["episodes"] = [asdict(episode) for episode in self.episodes]
        return payload


PolicyFactory = Callable[[int, int], Any]
SeedAwarePolicyFactory = Callable[[int, int, int], Any]


def evaluate_full_runs(
    environment_factory: Callable[[], StsEnv],
    policy_factory: PolicyFactory,
    seeds: tuple[int, ...],
    *,
    policy_seed: int,
    max_steps: int = 5000,
    cycle_limit: int = 16,
    bootstrap_samples: int = 10000,
    seed_aware_policy_factory: SeedAwarePolicyFactory | None = None,
) -> FullRunEvaluationSummary:
    if not seeds or max_steps <= 0 or cycle_limit <= 0 or bootstrap_samples <= 0:
        raise ValueError("full-run evaluation configuration is invalid")
    episodes: list[FullRunEvaluationEpisode] = []
    for episode_index, seed in enumerate(seeds):
        episode_policy_seed = _split_seed(policy_seed, seed, episode_index)
        environment = environment_factory()
        policy = (
            seed_aware_policy_factory(episode_policy_seed, episode_index, seed)
            if seed_aware_policy_factory is not None
            else policy_factory(episode_policy_seed, episode_index)
        )
        started = time.perf_counter()
        observation: Observation | None = None
        environment_return = 0.0
        outcome = "error"
        error_message = ""
        error_category = ""
        game_score: int | None = None
        repeated_decisions: dict[str, int] = {}
        calls_before = _simulator_calls(policy)
        decisions = 0
        try:
            observation, _ = environment.reset(seed=seed)
            if hasattr(policy, "reset"):
                policy.reset()
            for step_index in range(max_steps):
                action = _select_action(policy, environment, observation, episode_index)
                decision_key = _decision_key(observation, action)
                repeated_decisions[decision_key] = repeated_decisions.get(decision_key, 0) + 1
                if repeated_decisions[decision_key] > cycle_limit:
                    outcome = "cycle"
                    decisions = step_index
                    break
                observation, reward, terminated, truncated, info = environment.step(action)
                environment_return += reward
                decisions = step_index + 1
                if terminated or truncated:
                    outcome = str(
                        info.get("outcome") or ("truncated" if truncated else "terminal")
                    )
                    score_value = info.get("score")
                    game_score = int(score_value) if score_value is not None else None
                    break
            else:
                outcome = "timeout"
                decisions = max_steps
        except Exception as error:
            outcome = "error"
            error_message = f"{type(error).__name__}: {error}"
            error_category = _error_category(error)
        won = environment_return > 0 and outcome not in {"error", "timeout", "cycle"}
        final_act = observation.act if observation is not None else 0
        final_floor = observation.floor if observation is not None else 0
        final_hp = observation.player.hp if observation is not None else 0
        max_hp = observation.player.max_hp if observation is not None else 1
        proxy_score = environment_return + final_floor / 60.0 + final_hp / max(1, max_hp)
        episodes.append(
            FullRunEvaluationEpisode(
                seed=seed,
                policy_seed=episode_policy_seed,
                outcome=outcome,
                won=won,
                final_act=final_act,
                final_floor=final_floor,
                final_hp=final_hp,
                max_hp=max_hp,
                environment_return=environment_return,
                proxy_score=proxy_score,
                game_score=game_score,
                decisions=decisions,
                simulator_calls=max(0, _simulator_calls(policy) - calls_before),
                wall_seconds=time.perf_counter() - started,
                error=error_message,
                error_category=error_category,
            )
        )

    floors = [episode.final_floor for episode in episodes]
    wins = sum(episode.won for episode in episodes)
    return FullRunEvaluationSummary(
        episodes=tuple(episodes),
        win_rate=wins / len(episodes),
        win_rate_ci95=wilson_interval(wins, len(episodes)),
        act1_clear_rate=sum(episode.final_act >= 2 or episode.won for episode in episodes)
        / len(episodes),
        act2_clear_rate=sum(episode.final_act >= 3 or episode.won for episode in episodes)
        / len(episodes),
        act3_clear_rate=sum(episode.won for episode in episodes) / len(episodes),
        mean_floor=statistics.mean(floors),
        median_floor=statistics.median(floors),
        mean_floor_ci95=bootstrap_mean_interval(
            floors,
            samples=bootstrap_samples,
            seed=policy_seed ^ 0xB0057A9,
        ),
        mean_final_hp=statistics.mean(episode.final_hp for episode in episodes),
        mean_proxy_score=statistics.mean(episode.proxy_score for episode in episodes),
        mean_decisions=statistics.mean(episode.decisions for episode in episodes),
        total_simulator_calls=sum(episode.simulator_calls for episode in episodes),
        total_wall_seconds=sum(episode.wall_seconds for episode in episodes),
        errors=sum(bool(episode.error) for episode in episodes),
        crashes=sum(episode.error_category == "crash" for episode in episodes),
        illegal_actions=sum(
            episode.error_category == "illegal_action" for episode in episodes
        ),
        recovery_failures=sum(
            episode.error_category == "recovery_failure" for episode in episodes
        ),
        truncations=sum(episode.outcome == "truncated" for episode in episodes),
        timeouts=sum(episode.outcome == "timeout" for episode in episodes),
        cycles=sum(episode.outcome == "cycle" for episode in episodes),
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval counts are invalid")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def bootstrap_mean_interval(
    values: list[int] | tuple[int, ...],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values or samples <= 0:
        raise ValueError("bootstrap requires values and a positive sample count")
    source = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[source.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    lower_index = max(0, math.floor(0.025 * (samples - 1)))
    upper_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return means[lower_index], means[upper_index]


def _select_action(
    policy: Any,
    environment: StsEnv,
    observation: Observation,
    episode_index: int,
) -> Action:
    if hasattr(policy, "select"):
        return policy.select(environment)
    return policy(observation, episode_index)


def _decision_key(observation: Observation, action: Action) -> str:
    return json.dumps(
        {
            "observation": canonical_observation(observation),
            "action": action.to_dict(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _simulator_calls(policy: Any) -> int:
    if hasattr(policy, "total_simulator_calls"):
        return int(policy.total_simulator_calls)
    selector = getattr(policy, "combat_selector", None)
    owner = getattr(selector, "__self__", None)
    return int(getattr(owner, "total_simulator_calls", 0))


def _error_category(error: Exception) -> str:
    description = f"{type(error).__name__}: {error}".lower()
    if "illegal" in description and "action" in description:
        return "illegal_action"
    if "recover" in description or "replay" in description:
        return "recovery_failure"
    return "crash"


def _split_seed(policy_seed: int, environment_seed: int, episode_index: int) -> int:
    value = policy_seed ^ environment_seed ^ (episode_index * 0x9E3779B97F4A7C15)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB
    return (value ^ (value >> 31)) & ((1 << 63) - 1)
