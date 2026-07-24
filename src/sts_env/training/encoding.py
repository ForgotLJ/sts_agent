from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Iterable

import torch

from sts_env.types import Action, ActionKind, CardView, EnemyView, Observation, Phase


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    card_buckets: int = 32
    enemy_buckets: int = 16
    relic_buckets: int = 16
    potion_buckets: int = 16
    label_buckets: int = 16

    def __post_init__(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("encoder bucket counts must be positive")

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


class ObjectFeatureEncoder:
    def __init__(self, config: EncoderConfig | None = None):
        self.config = config or EncoderConfig()
        self._phase_order = tuple(Phase)
        self._action_order = tuple(ActionKind)
        self.dimension = len(self._empty_features())

    def encode(self, observation: Observation, action: Action) -> torch.Tensor:
        features = self._features(observation, action)
        if len(features) != self.dimension:
            raise AssertionError("encoder feature dimension changed unexpectedly")
        tensor = torch.tensor(features, dtype=torch.float32)
        if not torch.isfinite(tensor).all():
            raise ValueError("encoder produced a non-finite feature")
        return tensor

    def encode_candidates(self, observation: Observation) -> torch.Tensor:
        if not observation.legal_actions:
            return torch.empty((0, self.dimension), dtype=torch.float32)
        return torch.stack(
            [self.encode(observation, action) for action in observation.legal_actions]
        )

    def _empty_features(self) -> list[float]:
        player = [0.0] * 8
        run = [0.0] * 7
        pile_totals = [0.0] * 5
        hand_summary = [0.0] * 7
        enemy_summary = [0.0] * 8
        action_numeric = [0.0] * 12
        return (
            [0.0] * len(self._phase_order)
            + player
            + run
            + pile_totals
            + hand_summary
            + enemy_summary
            + [0.0] * (self.config.card_buckets * 5)
            + [0.0] * self.config.enemy_buckets
            + [0.0] * self.config.relic_buckets
            + [0.0] * self.config.potion_buckets
            + [0.0] * len(self._action_order)
            + action_numeric
            + [0.0] * self.config.card_buckets
            + [0.0] * self.config.enemy_buckets
            + [0.0] * self.config.label_buckets
        )

    def _features(self, observation: Observation, action: Action) -> list[float]:
        phase = [float(observation.phase is candidate) for candidate in self._phase_order]
        hp_scale = max(1, observation.player.max_hp)
        player = [
            observation.player.hp / hp_scale,
            observation.player.max_hp / 100.0,
            observation.player.block / 50.0,
            observation.player.energy / 5.0,
            observation.player.gold / 500.0,
            len(observation.player.statuses) / 10.0,
            sum(max(0, value) for _, value in observation.player.statuses) / 20.0,
            float(observation.player.hp <= 0),
        ]
        run = [
            observation.turn / 20.0,
            observation.ascension / 20.0,
            observation.act / 4.0,
            observation.floor / 60.0,
            observation.map_x / 7.0,
            observation.map_y / 16.0,
            len(observation.legal_actions) / 32.0,
        ]
        pile_totals = [
            self._count_total(observation.deck) / 50.0,
            self._count_total(observation.draw_pile) / 50.0,
            self._count_total(observation.discard_pile) / 50.0,
            self._count_total(observation.exhaust_pile) / 50.0,
            len(observation.hand) / 10.0,
        ]
        hand_summary = self._hand_summary(observation.hand)
        enemy_summary = self._enemy_summary(observation.enemies)
        card_histograms = (
            self._card_histogram((card.card_id, 1) for card in observation.hand)
            + self._card_histogram(observation.deck)
            + self._card_histogram(observation.draw_pile)
            + self._card_histogram(observation.discard_pile)
            + self._card_histogram(observation.exhaust_pile)
        )
        enemy_histogram = self._histogram(
            (enemy.monster_id or enemy.name for enemy in observation.enemies),
            self.config.enemy_buckets,
            scale=max(1, len(observation.enemies)),
        )
        relic_histogram = self._histogram(
            (relic_id for relic_id, _ in observation.relics),
            self.config.relic_buckets,
            scale=max(1, len(observation.relics)),
        )
        potion_histogram = self._histogram(
            observation.potions,
            self.config.potion_buckets,
            scale=max(1, len(observation.potions)),
        )
        action_kind = [float(action.kind is candidate) for candidate in self._action_order]
        source_card = next(
            (card for card in observation.hand if card.instance_id == action.source_id),
            None,
        )
        target_enemy = next(
            (enemy for enemy in observation.enemies if enemy.enemy_id == action.target_id),
            None,
        )
        action_numeric = self._action_numeric(action, source_card, target_enemy)
        action_card_hash = self._single_hash(
            source_card.card_id if source_card is not None else action.source_id,
            self.config.card_buckets,
        )
        action_enemy_hash = self._single_hash(
            target_enemy.monster_id or target_enemy.name if target_enemy is not None else action.target_id,
            self.config.enemy_buckets,
        )
        action_label_hash = self._single_hash(action.label, self.config.label_buckets)
        return (
            phase
            + player
            + run
            + pile_totals
            + hand_summary
            + enemy_summary
            + card_histograms
            + enemy_histogram
            + relic_histogram
            + potion_histogram
            + action_kind
            + action_numeric
            + action_card_hash
            + action_enemy_hash
            + action_label_hash
        )

    def _hand_summary(self, hand: tuple[CardView, ...]) -> list[float]:
        if not hand:
            return [0.0] * 7
        costs = [max(0, card.cost) for card in hand]
        return [
            sum(costs) / (len(hand) * 5.0),
            min(costs) / 5.0,
            max(costs) / 5.0,
            sum(card.playable for card in hand) / len(hand),
            sum(card.requires_target for card in hand) / len(hand),
            sum(card.upgraded for card in hand) / len(hand),
            sum(card.cost == 0 for card in hand) / len(hand),
        ]

    @staticmethod
    def _enemy_summary(enemies: tuple[EnemyView, ...]) -> list[float]:
        living = [enemy for enemy in enemies if enemy.hp > 0]
        if not living:
            return [0.0] * 8
        hp_ratios = [enemy.hp / max(1, enemy.max_hp) for enemy in living]
        return [
            len(living) / 5.0,
            sum(hp_ratios) / len(living),
            min(hp_ratios),
            sum(enemy.block for enemy in living) / 100.0,
            sum(enemy.intent_damage * max(1, enemy.intent_hits) for enemy in living) / 100.0,
            max(enemy.intent_damage * max(1, enemy.intent_hits) for enemy in living) / 50.0,
            sum(len(enemy.statuses) for enemy in living) / 20.0,
            sum(max(0, value) for enemy in living for _, value in enemy.statuses) / 50.0,
        ]

    @staticmethod
    def _action_numeric(
        action: Action,
        source_card: CardView | None,
        target_enemy: EnemyView | None,
    ) -> list[float]:
        source_numeric = float(action.source_id) / 100.0 if isinstance(action.source_id, int) else 0.0
        target_numeric = float(action.target_id) / 10.0 if isinstance(action.target_id, int) else 0.0
        choice_numeric = (
            float(action.choice_index) / 20.0 if action.choice_index is not None else 0.0
        )
        return [
            source_numeric,
            target_numeric,
            choice_numeric,
            (source_card.cost / 5.0) if source_card is not None else 0.0,
            float(source_card.playable) if source_card is not None else 0.0,
            float(source_card.requires_target) if source_card is not None else 0.0,
            float(source_card.upgraded) if source_card is not None else 0.0,
            (target_enemy.hp / max(1, target_enemy.max_hp)) if target_enemy is not None else 0.0,
            (target_enemy.block / 50.0) if target_enemy is not None else 0.0,
            (target_enemy.intent_damage / 50.0) if target_enemy is not None else 0.0,
            (target_enemy.intent_hits / 10.0) if target_enemy is not None else 0.0,
            math.log1p(len(action.label)) / 5.0,
        ]

    def _card_histogram(self, values: Iterable[tuple[str, int]]) -> list[float]:
        histogram = [0.0] * self.config.card_buckets
        total = 0
        for card_id, count in values:
            if count <= 0:
                continue
            histogram[self._bucket(card_id, self.config.card_buckets)] += count
            total += count
        scale = max(1, total)
        return [value / scale for value in histogram]

    def _histogram(
        self,
        values: Iterable[str],
        buckets: int,
        scale: int,
    ) -> list[float]:
        histogram = [0.0] * buckets
        for value in values:
            histogram[self._bucket(value, buckets)] += 1.0
        return [value / scale for value in histogram]

    def _single_hash(self, value: object, buckets: int) -> list[float]:
        result = [0.0] * buckets
        if value is not None and str(value):
            result[self._bucket(str(value), buckets)] = 1.0
        return result

    @staticmethod
    def _bucket(value: str, buckets: int) -> int:
        digest = hashlib.blake2b(value.lower().encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % buckets

    @staticmethod
    def _count_total(counts: tuple[tuple[str, int], ...]) -> int:
        return sum(count for _, count in counts)
