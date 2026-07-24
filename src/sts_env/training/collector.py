from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sts_env.env import StsEnv
from sts_env.training.replay import ReplayTransition
from sts_env.types import Action, Observation


Policy = Callable[[Observation, int], Action]


@dataclass(frozen=True, slots=True)
class CompletedEpisode:
    environment_index: int
    seed: int
    length: int
    environment_return: float
    final_hp: int
    won: bool


@dataclass(frozen=True, slots=True)
class CollectionBatch:
    transitions: tuple[ReplayTransition, ...]
    completed_episodes: tuple[CompletedEpisode, ...]


class SynchronousVectorCollector:
    def __init__(
        self,
        environment_factory: Callable[[], StsEnv],
        num_environments: int,
        seeds: tuple[int, ...],
    ):
        if num_environments <= 0:
            raise ValueError("num_environments must be positive")
        if len(seeds) < num_environments:
            raise ValueError("seed stream must initialize every environment")
        self._environments = [environment_factory() for _ in range(num_environments)]
        self._seeds = seeds
        self._next_seed_index = 0
        self._observations: list[Observation] = []
        self._active_seeds: list[int] = []
        self._episode_lengths = [0] * num_environments
        self._episode_returns = [0.0] * num_environments
        for environment in self._environments:
            seed = self._next_seed()
            observation, _ = environment.reset(seed=seed)
            self._observations.append(observation)
            self._active_seeds.append(seed)

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def collect(self, steps: int, policy: Policy) -> CollectionBatch:
        if steps <= 0:
            raise ValueError("steps must be positive")
        transitions: list[ReplayTransition] = []
        completed: list[CompletedEpisode] = []
        while len(transitions) < steps:
            for environment_index, environment in enumerate(self._environments):
                if len(transitions) >= steps:
                    break
                observation = self._observations[environment_index]
                action = policy(observation, environment_index)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                transitions.append(
                    ReplayTransition(
                        observation=observation,
                        action=action,
                        reward=reward,
                        next_observation=next_observation,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )
                self._episode_lengths[environment_index] += 1
                self._episode_returns[environment_index] += reward
                self._observations[environment_index] = next_observation
                if terminated or truncated:
                    completed.append(
                        CompletedEpisode(
                            environment_index=environment_index,
                            seed=self._active_seeds[environment_index],
                            length=self._episode_lengths[environment_index],
                            environment_return=self._episode_returns[environment_index],
                            final_hp=next_observation.player.hp,
                            won=reward > 0,
                        )
                    )
                    seed = self._next_seed()
                    reset_observation, _ = environment.reset(seed=seed)
                    self._observations[environment_index] = reset_observation
                    self._active_seeds[environment_index] = seed
                    self._episode_lengths[environment_index] = 0
                    self._episode_returns[environment_index] = 0.0
        return CollectionBatch(tuple(transitions), tuple(completed))

    def _next_seed(self) -> int:
        if self._next_seed_index >= len(self._seeds):
            raise RuntimeError("collector exhausted its training seed stream")
        seed = self._seeds[self._next_seed_index]
        self._next_seed_index += 1
        return seed
