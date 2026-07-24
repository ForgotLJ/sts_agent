from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Protocol

from sts_env.env import StsEnv
from sts_env.types import Action, Observation


@dataclass(frozen=True, slots=True)
class BeliefConstraints:
    known_top: tuple[str, ...] = ()
    known_bottom: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not card_id for card_id in self.known_top + self.known_bottom):
            raise ValueError("draw-order constraints require non-empty card ids")

    def after_draw(self, count: int) -> BeliefConstraints:
        if count < 0:
            raise ValueError("draw count must be non-negative")
        return replace(self, known_top=self.known_top[count:])

    def after_shuffle(self) -> BeliefConstraints:
        return BeliefConstraints()

    def with_known_top(self, card_ids: tuple[str, ...]) -> BeliefConstraints:
        return replace(self, known_top=card_ids)

    def with_known_bottom(self, card_ids: tuple[str, ...]) -> BeliefConstraints:
        return replace(self, known_bottom=card_ids)


@dataclass(frozen=True, slots=True)
class PublicHistoryStep:
    action: Action
    observation: Observation


@dataclass(frozen=True, slots=True)
class PublicHistory:
    initial_observation: Observation
    steps: tuple[PublicHistoryStep, ...] = ()
    constraints: BeliefConstraints = BeliefConstraints()

    @property
    def observation(self) -> Observation:
        if self.steps:
            return self.steps[-1].observation
        return self.initial_observation

    def append(
        self,
        action: Action,
        observation: Observation,
        *,
        drawn_cards: int = 0,
        shuffled: bool = False,
        known_top: tuple[str, ...] | None = None,
        known_bottom: tuple[str, ...] | None = None,
    ) -> PublicHistory:
        constraints = self.constraints.after_shuffle() if shuffled else self.constraints
        constraints = constraints.after_draw(drawn_cards)
        if known_top is not None:
            constraints = constraints.with_known_top(known_top)
        if known_bottom is not None:
            constraints = constraints.with_known_bottom(known_bottom)
        return PublicHistory(
            initial_observation=self.initial_observation,
            steps=self.steps + (PublicHistoryStep(action, observation),),
            constraints=constraints,
        )


def public_observation_key(observation: Observation) -> str:
    return json.dumps(
        observation.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class BeliefSource(Protocol):
    @property
    def observation(self) -> Observation:
        ...

    def sample(self, search_seed: int) -> StsEnv:
        ...


class EnvironmentBeliefSource:
    def __init__(self, environment: StsEnv, history: PublicHistory | None = None):
        if history is not None and history.observation != environment.observation:
            raise ValueError("public history does not end at the environment observation")
        self._environment = environment
        self._history = history or PublicHistory(environment.observation)

    @property
    def observation(self) -> Observation:
        return self._history.observation

    @property
    def history(self) -> PublicHistory:
        return self._history

    def sample(self, search_seed: int) -> StsEnv:
        constraints = self._history.constraints
        sampled = self._environment.redeterminized_clone(
            search_seed,
            known_top=constraints.known_top,
            known_bottom=constraints.known_bottom,
        )
        if sampled.observation != self.observation:
            raise RuntimeError("belief sample changed the root public observation")
        return sampled
