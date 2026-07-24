from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Callable

from sts_env.env import StsEnv
from sts_env.types import Action, Observation


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    seed: int
    environment_return: float
    score: float
    length: int
    final_hp: int
    won: bool


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    episodes: tuple[EvaluationEpisode, ...]
    mean_score: float
    score_ci95: float
    win_rate: float
    mean_length: float
    mean_final_hp: float

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": [asdict(episode) for episode in self.episodes],
            "mean_score": self.mean_score,
            "score_ci95": self.score_ci95,
            "win_rate": self.win_rate,
            "mean_length": self.mean_length,
            "mean_final_hp": self.mean_final_hp,
        }


def evaluate_policy(
    environment_factory: Callable[[], StsEnv],
    policy: Any,
    seeds: tuple[int, ...],
    max_steps: int = 20_000,
) -> EvaluationSummary:
    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    episodes: list[EvaluationEpisode] = []
    for episode_index, seed in enumerate(seeds):
        environment = environment_factory()
        observation, _ = environment.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        environment_return = 0.0
        for step_index in range(max_steps):
            if hasattr(policy, "select"):
                action = policy.select(environment)
            else:
                action = policy(observation, episode_index)
            observation, reward, terminated, truncated, _ = environment.step(action)
            environment_return += reward
            if terminated or truncated:
                length = step_index + 1
                break
        else:
            raise RuntimeError(f"evaluation seed {seed} exceeded max_steps")
        won = environment_return > 0
        score = (
            environment_return
            + observation.player.hp / max(1, observation.player.max_hp)
            - 0.01 * length
        )
        episodes.append(
            EvaluationEpisode(
                seed=seed,
                environment_return=environment_return,
                score=score,
                length=length,
                final_hp=observation.player.hp,
                won=won,
            )
        )
    scores = [episode.score for episode in episodes]
    score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return EvaluationSummary(
        episodes=tuple(episodes),
        mean_score=statistics.mean(scores),
        score_ci95=1.96 * score_std / math.sqrt(len(scores)),
        win_rate=sum(episode.won for episode in episodes) / len(episodes),
        mean_length=statistics.mean(episode.length for episode in episodes),
        mean_final_hp=statistics.mean(episode.final_hp for episode in episodes),
    )
