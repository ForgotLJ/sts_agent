from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass
from typing import Self

from sts_env.backend import Transition
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
class _CardSpec:
    card_id: str
    name: str
    cost: int
    damage: int = 0
    block: int = 0
    vulnerable: int = 0

    @property
    def requires_target(self) -> bool:
        return self.damage > 0 or self.vulnerable > 0


@dataclass(frozen=True, slots=True)
class _CardInstance:
    instance_id: int
    spec: _CardSpec


STRIKE = _CardSpec("strike", "Strike", cost=1, damage=6)
DEFEND = _CardSpec("defend", "Defend", cost=1, block=5)
BASH = _CardSpec("bash", "Bash", cost=2, damage=8, vulnerable=2)


class ToyCombatBackend:
    """A deterministic-by-seed reference backend for infrastructure tests.

    The mechanics are deliberately small and are not claimed to exactly match
    every Slay the Spire rule. Hidden draw order and RNG state stay private.
    """

    def __init__(self, max_turns: int = 50):
        self._max_turns = max_turns
        self._rng = random.Random()
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
        self._rng = random.Random(seed)
        self._initialized = True
        self._terminated = False
        self._truncated = False
        self._turn = 1
        self._player_hp = 80
        self._player_block = 0
        self._energy = 3
        self._enemy_hp = 46
        self._enemy_block = 0
        self._enemy_vulnerable = 0
        self._enemy_strength = 0
        self._draw_pile = self._make_starter_deck()
        self._discard_pile: list[_CardInstance] = []
        self._exhaust_pile: list[_CardInstance] = []
        self._hand: list[_CardInstance] = []
        self._rng.shuffle(self._draw_pile)
        self._draw_cards(5)
        return self._observe(), {"seed": seed if seed is not None else "random"}

    def step(self, action: Action) -> Transition:
        self._require_active()
        legal_actions = self._legal_actions()
        if action not in legal_actions:
            raise ValueError("backend received an illegal action")

        damage_dealt = 0
        damage_taken = 0
        terminal_reason = ""

        if action.kind is ActionKind.PLAY_CARD:
            damage_dealt = self._play_card(int(action.source_id))
        elif action.kind is ActionKind.END_TURN:
            damage_taken = self._end_turn()
        else:
            raise ValueError(f"unsupported toy action: {action.kind.value}")

        reward = 0.0
        if self._enemy_hp <= 0:
            self._enemy_hp = 0
            self._terminated = True
            reward = 1.0
            terminal_reason = "combat_won"
        elif self._player_hp <= 0:
            self._player_hp = 0
            self._terminated = True
            reward = -1.0
            terminal_reason = "player_died"
        elif self._turn > self._max_turns:
            self._truncated = True
            terminal_reason = "turn_limit"

        observation = self._observe()
        return Transition(
            observation=observation,
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info={
                "damage_dealt": damage_dealt,
                "damage_taken": damage_taken,
                "terminal_reason": terminal_reason,
            },
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
        cloned._rng = random.Random(search_seed)
        cloned._draw_pile = self._redeterminize_pile(
            cloned._draw_pile,
            cloned._rng,
            known_top,
            known_bottom,
        )
        return cloned

    @staticmethod
    def _redeterminize_pile(
        pile: list[_CardInstance],
        random_source: random.Random,
        known_top: tuple[str, ...],
        known_bottom: tuple[str, ...],
    ) -> list[_CardInstance]:
        remaining = list(pile)

        def take(card_id: str) -> _CardInstance:
            index = next(
                (index for index, card in enumerate(remaining) if card.spec.card_id == card_id),
                None,
            )
            if index is None:
                raise ValueError(f"known draw constraint is absent from pile: {card_id}")
            return remaining.pop(index)

        bottom_cards = [take(card_id) for card_id in known_bottom]
        top_cards = [take(card_id) for card_id in known_top]
        random_source.shuffle(remaining)
        return bottom_cards + remaining + list(reversed(top_cards))

    def _make_starter_deck(self) -> list[_CardInstance]:
        specs = [STRIKE] * 5 + [DEFEND] * 4 + [BASH]
        return [_CardInstance(instance_id=index, spec=spec) for index, spec in enumerate(specs)]

    def _play_card(self, instance_id: int) -> int:
        hand_index = next(
            index for index, card in enumerate(self._hand) if card.instance_id == instance_id
        )
        card = self._hand.pop(hand_index)
        self._energy -= card.spec.cost

        damage = card.spec.damage
        if damage and self._enemy_vulnerable > 0:
            damage = (damage * 3 + 1) // 2
        damage_after_block = max(0, damage - self._enemy_block)
        self._enemy_block = max(0, self._enemy_block - damage)
        self._enemy_hp -= damage_after_block
        self._player_block += card.spec.block
        self._enemy_vulnerable += card.spec.vulnerable
        self._discard_pile.append(card)
        return damage_after_block

    def _end_turn(self) -> int:
        for card in self._hand:
            self._discard_pile.append(card)
        self._hand.clear()

        attack = self._enemy_intent()
        damage_taken = max(0, attack - self._player_block)
        self._player_hp -= damage_taken
        self._player_block = 0

        if self._enemy_vulnerable > 0:
            self._enemy_vulnerable -= 1

        if self._turn == 1:
            self._enemy_strength += 3

        self._turn += 1
        if self._player_hp > 0:
            self._energy = 3
            self._draw_cards(5)
        return damage_taken

    def _draw_cards(self, count: int) -> None:
        for _ in range(count):
            if not self._draw_pile:
                if not self._discard_pile:
                    return
                self._draw_pile = self._discard_pile
                self._discard_pile = []
                self._rng.shuffle(self._draw_pile)
            self._hand.append(self._draw_pile.pop())

    def _enemy_intent(self) -> int:
        if self._turn == 1:
            return 0
        return 6 + self._enemy_strength

    def _legal_actions(self) -> tuple[Action, ...]:
        if self._terminated or self._truncated:
            return ()

        actions: list[Action] = []
        for card in sorted(self._hand, key=lambda item: item.instance_id):
            if card.spec.cost > self._energy:
                continue
            actions.append(
                Action(
                    kind=ActionKind.PLAY_CARD,
                    source_id=card.instance_id,
                    target_id=0 if card.spec.requires_target else None,
                    label=f"Play {card.spec.name}",
                )
            )
        actions.append(Action(kind=ActionKind.END_TURN, label="End turn"))
        return tuple(actions)

    def _observe(self) -> Observation:
        phase = Phase.TERMINAL if self._terminated or self._truncated else Phase.COMBAT
        hand = tuple(
            CardView(
                instance_id=card.instance_id,
                card_id=card.spec.card_id,
                name=card.spec.name,
                cost=card.spec.cost,
                upgraded=False,
                playable=card.spec.cost <= self._energy,
                requires_target=card.spec.requires_target,
            )
            for card in sorted(self._hand, key=lambda item: item.instance_id)
        )
        enemy_statuses = (
            (("vulnerable", self._enemy_vulnerable),) if self._enemy_vulnerable else ()
        )
        return Observation(
            phase=phase,
            turn=self._turn,
            player=PlayerView(
                hp=self._player_hp,
                max_hp=80,
                block=self._player_block,
                energy=self._energy,
            ),
            hand=hand,
            enemies=(
                EnemyView(
                    enemy_id=0,
                    name="Training Cultist",
                    hp=self._enemy_hp,
                    max_hp=46,
                    block=self._enemy_block,
                    intent_damage=self._enemy_intent(),
                    intent_hits=1 if self._enemy_intent() else 0,
                    statuses=enemy_statuses,
                ),
            ),
            draw_pile=self._pile_counts(self._draw_pile),
            discard_pile=self._pile_counts(self._discard_pile),
            exhaust_pile=self._pile_counts(self._exhaust_pile),
            legal_actions=self._legal_actions(),
        )

    @staticmethod
    def _pile_counts(pile: list[_CardInstance]) -> tuple[tuple[str, int], ...]:
        counts = Counter(card.spec.card_id for card in pile)
        return tuple(sorted(counts.items()))

    def _require_active(self) -> None:
        if not self._initialized:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("cannot step a completed episode")
