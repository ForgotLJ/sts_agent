from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
import random
from typing import Self

from sts_env.backend import Transition
from sts_env.env import StsEnv
from sts_env.search.belief import public_observation_key
from sts_env.types import (
    Action,
    ActionKind,
    CardView,
    EnemyView,
    Observation,
    Phase,
    PlayerView,
)


@dataclass(frozen=True, slots=True)
class _FixtureCard:
    instance_id: int
    card_id: str
    damage: int = 0
    block: int = 0


STRIKE = _FixtureCard(0, "fixture_strike", damage=6)
DEFEND = _FixtureCard(1, "fixture_defend", block=6)
FINISHER = _FixtureCard(2, "fixture_finisher", damage=10)
GUARD = _FixtureCard(3, "fixture_guard", block=10)


class StochasticCombatFixtureBackend:
    def __init__(self, hidden_order: tuple[str, ...] | None = None):
        self._forced_hidden_order = hidden_order
        self._initialized = False
        self._terminated = False
        self._truncated = False

    @property
    def supports_clone(self) -> bool:
        return True

    @property
    def supports_redeterminization(self) -> bool:
        return True

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, int | str]]:
        random_source = random.Random(seed)
        self._initialized = True
        self._terminated = False
        self._truncated = False
        self._turn = 1
        self._player_hp = 12
        self._player_block = 0
        self._enemy_hp = 10
        self._hand = [STRIKE, DEFEND]
        cards = {FINISHER.card_id: FINISHER, GUARD.card_id: GUARD}
        if self._forced_hidden_order is None:
            self._draw_pile = [FINISHER, GUARD]
            random_source.shuffle(self._draw_pile)
        else:
            if Counter(self._forced_hidden_order) != Counter(cards.keys()):
                raise ValueError("fixture hidden order must contain finisher and guard once")
            self._draw_pile = [cards[card_id] for card_id in self._forced_hidden_order]
        return self._observe(), {"seed": seed if seed is not None else "random"}

    def step(self, action: Action) -> Transition:
        self._require_active()
        if action not in self._legal_actions():
            raise ValueError("fixture received an illegal action")
        card = next(card for card in self._hand if card.instance_id == action.source_id)
        self._enemy_hp = max(0, self._enemy_hp - card.damage)
        self._player_block += card.block
        damage_taken = 0
        if self._enemy_hp > 0:
            incoming = self._enemy_intent()
            damage_taken = max(0, incoming - self._player_block)
            self._player_hp = max(0, self._player_hp - damage_taken)
        self._player_block = 0

        reward = 0.0
        if self._enemy_hp <= 0:
            self._terminated = True
            reward = 1.0
        elif self._player_hp <= 0:
            self._terminated = True
            reward = -1.0
        else:
            self._turn += 1
            if self._turn > 4 or not self._draw_pile:
                self._truncated = True
            else:
                self._hand = [self._draw_pile.pop()]

        return Transition(
            observation=self._observe(),
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info={"damage_taken": damage_taken, "damage_dealt": card.damage},
        )

    def clone(self) -> Self:
        return copy.deepcopy(self)

    def redeterminized_clone(
        self,
        search_seed: int,
        known_top: tuple[str, ...] = (),
        known_bottom: tuple[str, ...] = (),
    ) -> Self:
        if search_seed < 0 or search_seed >= 2**64:
            raise ValueError("search_seed must be in [0, 2**64)")
        cloned = self.clone()
        remaining = list(cloned._draw_pile)

        def take(card_id: str) -> _FixtureCard:
            index = next(
                (index for index, card in enumerate(remaining) if card.card_id == card_id),
                None,
            )
            if index is None:
                raise ValueError(f"known draw constraint is absent from fixture pile: {card_id}")
            return remaining.pop(index)

        bottom_cards = [take(card_id) for card_id in known_bottom]
        top_cards = [take(card_id) for card_id in known_top]
        random.Random(search_seed).shuffle(remaining)
        cloned._draw_pile = bottom_cards + remaining + list(reversed(top_cards))
        return cloned

    def _legal_actions(self) -> tuple[Action, ...]:
        if self._terminated or self._truncated:
            return ()
        return tuple(
            Action(
                kind=ActionKind.PLAY_CARD,
                source_id=card.instance_id,
                target_id=0 if card.damage else None,
                label=f"Play {card.card_id}",
            )
            for card in self._hand
        )

    def _enemy_intent(self) -> int:
        return 6 if self._turn == 1 else 10

    def _observe(self) -> Observation:
        terminal = self._terminated or self._truncated
        return Observation(
            phase=Phase.TERMINAL if terminal else Phase.COMBAT,
            turn=self._turn,
            player=PlayerView(
                hp=self._player_hp,
                max_hp=12,
                block=self._player_block,
                energy=1,
            ),
            hand=tuple(
                CardView(
                    instance_id=card.instance_id,
                    card_id=card.card_id,
                    name=card.card_id,
                    cost=1,
                    upgraded=False,
                    playable=True,
                    requires_target=bool(card.damage),
                )
                for card in self._hand
            ) if not terminal else (),
            enemies=(
                EnemyView(
                    enemy_id=0,
                    name="Fixture Enemy",
                    hp=self._enemy_hp,
                    max_hp=10,
                    block=0,
                    intent_damage=0 if terminal else self._enemy_intent(),
                    intent_hits=0 if terminal else 1,
                    monster_id="fixture_enemy",
                ),
            ),
            draw_pile=tuple(sorted(Counter(card.card_id for card in self._draw_pile).items())),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=self._legal_actions(),
        )

    def _require_active(self) -> None:
        if not self._initialized:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("cannot step a completed fixture")


class FixtureBeliefSource:
    def __init__(self):
        self._root = StsEnv(StochasticCombatFixtureBackend())
        self._observation, _ = self._root.reset(seed=0)

    @property
    def observation(self) -> Observation:
        return self._observation

    def sample(self, search_seed: int) -> StsEnv:
        sampled = self._root.redeterminized_clone(search_seed)
        if sampled.observation != self._observation:
            raise RuntimeError("fixture belief sample changed the root observation")
        return sampled


def exact_fixture_action_values(gamma: float = 1.0) -> dict[Action, float]:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    worlds: list[tuple[StsEnv, float]] = []
    for order in (
        (FINISHER.card_id, GUARD.card_id),
        (GUARD.card_id, FINISHER.card_id),
    ):
        environment = StsEnv(StochasticCombatFixtureBackend(hidden_order=order))
        environment.reset(seed=0)
        worlds.append((environment, 0.5))
    return _exact_belief_values(worlds, gamma)


def _exact_belief_values(
    worlds: list[tuple[StsEnv, float]],
    gamma: float,
) -> dict[Action, float]:
    observation = worlds[0][0].observation
    if any(environment.observation != observation for environment, _ in worlds[1:]):
        raise ValueError("exact belief worlds must share one public observation")
    values: dict[Action, float] = {}
    for action in observation.legal_actions:
        grouped: dict[str, list[tuple[StsEnv, float]]] = {}
        immediate = 0.0
        terminal_bonus = 0.0
        for environment, weight in worlds:
            branch = environment.clone()
            next_observation, reward, terminated, truncated, _ = branch.step(action)
            immediate += weight * reward
            if terminated or truncated:
                terminal_bonus += weight * (
                    next_observation.player.hp / max(1, next_observation.player.max_hp)
                )
            else:
                grouped.setdefault(public_observation_key(next_observation), []).append(
                    (branch, weight)
                )
        continuation = 0.0
        for group in grouped.values():
            group_weight = sum(weight for _, weight in group)
            normalized = [(environment, weight / group_weight) for environment, weight in group]
            child_values = _exact_belief_values(normalized, gamma)
            continuation += group_weight * max(child_values.values())
        values[action] = immediate + terminal_bonus + gamma * continuation
    return values
