from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Iterable

import torch

from sts_env.types import Action, ActionKind, CardView, EnemyView, Observation, Phase


_ROOM_SYMBOLS = ("M", "E", "?", "R", "$", "T", "B", "N")


@dataclass(frozen=True, slots=True)
class RunEncoderConfig:
    card_buckets: int = 64
    enemy_buckets: int = 32
    relic_buckets: int = 32
    potion_buckets: int = 16
    boss_buckets: int = 8
    option_buckets: int = 32
    text_buckets: int = 32

    def __post_init__(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("encoder bucket counts must be positive")

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


class RunFeatureEncoder:
    def __init__(self, config: RunEncoderConfig | None = None):
        self.config = config or RunEncoderConfig()
        self._phase_order = tuple(Phase)
        self._action_order = tuple(ActionKind)
        self.state_dimension = len(self._state_features(self._empty_observation()))
        self.action_dimension = len(
            self._action_features(self._empty_observation(), Action(ActionKind.LEAVE))
        )

    def encode_state(self, observation: Observation) -> torch.Tensor:
        tensor = torch.tensor(self._state_features(observation), dtype=torch.float32)
        self._validate(tensor, self.state_dimension, "state")
        return tensor

    def encode_actions(self, observation: Observation) -> torch.Tensor:
        if not observation.legal_actions:
            return torch.empty((0, self.action_dimension), dtype=torch.float32)
        tensor = torch.tensor(
            [self._action_features(observation, action) for action in observation.legal_actions],
            dtype=torch.float32,
        )
        if tensor.ndim != 2 or tensor.shape[1] != self.action_dimension:
            raise AssertionError("action encoder feature dimension changed unexpectedly")
        if not torch.isfinite(tensor).all():
            raise ValueError("action encoder produced a non-finite feature")
        return tensor

    def _state_features(self, observation: Observation) -> list[float]:
        phase = [float(observation.phase is candidate) for candidate in self._phase_order]
        hp_scale = max(1, observation.player.max_hp)
        player = [
            observation.player.hp / hp_scale,
            observation.player.max_hp / 100.0,
            observation.player.block / 100.0,
            observation.player.energy / 10.0,
            observation.player.gold / 1000.0,
            len(observation.player.statuses) / 16.0,
            sum(max(0, value) for _, value in observation.player.statuses) / 50.0,
            float(observation.player.hp <= 0),
        ]
        run = [
            observation.turn / 30.0,
            observation.ascension / 20.0,
            observation.act / 4.0,
            observation.floor / 60.0,
            observation.map_x / 7.0,
            observation.map_y / 16.0,
            len(observation.legal_actions) / 64.0,
            observation.potion_capacity / 5.0,
            float(observation.ruby_key),
            float(observation.emerald_key),
            float(observation.sapphire_key),
            len(observation.map_nodes) / 60.0,
        ]
        piles = [
            self._count_total(observation.deck) / 60.0,
            self._count_total(observation.draw_pile) / 60.0,
            self._count_total(observation.discard_pile) / 60.0,
            self._count_total(observation.exhaust_pile) / 60.0,
            len(observation.hand) / 10.0,
        ]
        history = observation.history
        history_numeric = [
            history.decisions / 1000.0,
            history.rooms_visited / 60.0,
            history.combats_won / 40.0,
            history.elites_won / 12.0,
            history.bosses_won / 3.0,
            history.acts_cleared / 3.0,
            history.cards_added / 40.0,
            history.cards_removed / 20.0,
            history.potions_used / 20.0,
            history.potions_discarded / 20.0,
            history.gold_spent / 1000.0,
            history.hp_lost / 500.0,
            history.hp_healed / 500.0,
        ]
        room_counts = self._room_counts(node.symbol for node in observation.map_nodes)
        next_rooms = self._room_counts(
            action.option_type
            for action in observation.legal_actions
            if action.kind is ActionKind.CHOOSE_MAP_NODE
        )
        recent_rooms = self._room_counts(history.recent_rooms)
        cards = (
            self._histogram(
                (card.card_id for card in observation.hand),
                self.config.card_buckets,
                max(1, len(observation.hand)),
            )
            + self._weighted_histogram(observation.deck, self.config.card_buckets)
            + self._weighted_histogram(observation.draw_pile, self.config.card_buckets)
            + self._weighted_histogram(observation.discard_pile, self.config.card_buckets)
            + self._weighted_histogram(observation.exhaust_pile, self.config.card_buckets)
        )
        enemies = self._histogram(
            (enemy.monster_id or enemy.name for enemy in observation.enemies),
            self.config.enemy_buckets,
            max(1, len(observation.enemies)),
        )
        relics = self._histogram(
            (relic_id for relic_id, _ in observation.relics),
            self.config.relic_buckets,
            max(1, len(observation.relics)),
        )
        potions = self._histogram(
            observation.potions,
            self.config.potion_buckets,
            max(1, observation.potion_capacity),
        )
        boss = self._single_hash(observation.act_boss, self.config.boss_buckets)
        recent_actions = self._histogram(
            history.recent_actions,
            self.config.option_buckets,
            max(1, len(history.recent_actions)),
        )
        return (
            phase
            + player
            + run
            + piles
            + self._hand_summary(observation.hand)
            + self._enemy_summary(observation.enemies)
            + history_numeric
            + room_counts
            + next_rooms
            + recent_rooms
            + cards
            + enemies
            + relics
            + potions
            + boss
            + recent_actions
        )

    def _action_features(self, observation: Observation, action: Action) -> list[float]:
        kind = [float(action.kind is candidate) for candidate in self._action_order]
        source_card = next(
            (card for card in observation.hand if card.instance_id == action.source_id),
            None,
        )
        target_enemy = next(
            (enemy for enemy in observation.enemies if enemy.enemy_id == action.target_id),
            None,
        )
        numeric = [
            float(action.source_id) / 128.0 if isinstance(action.source_id, int) else 0.0,
            float(action.target_id) / 16.0 if isinstance(action.target_id, int) else 0.0,
            float(action.choice_index) / 64.0 if action.choice_index is not None else 0.0,
            action.amount / 1000.0,
            action.gold_cost / 1000.0,
            action.hp_cost / 100.0,
            action.target_x / 7.0,
            action.target_y / 16.0,
            float(action.gold_cost <= observation.player.gold),
            float(action.hp_cost < observation.player.hp),
            math.log1p(len(action.label)) / 8.0,
            math.log1p(len(action.description)) / 8.0,
        ]
        card = self._card_action_summary(source_card)
        enemy = self._enemy_action_summary(target_enemy)
        destination = [float(action.option_type == symbol) for symbol in _ROOM_SYMBOLS]
        source_hash = self._single_hash(
            source_card.card_id if source_card is not None else action.source_id,
            self.config.card_buckets,
        )
        target_hash = self._single_hash(
            (target_enemy.monster_id or target_enemy.name)
            if target_enemy is not None
            else action.target_id,
            self.config.enemy_buckets,
        )
        option_hash = self._single_hash(action.option_type, self.config.option_buckets)
        label_hash = self._single_hash(action.label, self.config.text_buckets)
        description_hash = self._single_hash(action.description, self.config.text_buckets)
        return (
            kind
            + numeric
            + card
            + enemy
            + destination
            + source_hash
            + target_hash
            + option_hash
            + label_hash
            + description_hash
        )

    @staticmethod
    def _hand_summary(hand: tuple[CardView, ...]) -> list[float]:
        if not hand:
            return [0.0] * 7
        costs = [max(0, card.cost) for card in hand]
        return [
            sum(costs) / (len(costs) * 5.0),
            min(costs) / 5.0,
            max(costs) / 5.0,
            sum(card.playable for card in hand) / len(hand),
            sum(card.requires_target for card in hand) / len(hand),
            sum(card.upgraded for card in hand) / len(hand),
            sum(card.cost <= 0 for card in hand) / len(hand),
        ]

    @staticmethod
    def _enemy_summary(enemies: tuple[EnemyView, ...]) -> list[float]:
        living = [enemy for enemy in enemies if enemy.hp > 0]
        if not living:
            return [0.0] * 8
        hp_ratios = [enemy.hp / max(1, enemy.max_hp) for enemy in living]
        incoming = [enemy.intent_damage * max(1, enemy.intent_hits) for enemy in living]
        return [
            len(living) / 5.0,
            sum(hp_ratios) / len(living),
            min(hp_ratios),
            sum(enemy.block for enemy in living) / 100.0,
            sum(incoming) / 100.0,
            max(incoming) / 50.0,
            sum(len(enemy.statuses) for enemy in living) / 20.0,
            sum(max(0, value) for enemy in living for _, value in enemy.statuses) / 50.0,
        ]

    @staticmethod
    def _card_action_summary(card: CardView | None) -> list[float]:
        if card is None:
            return [0.0] * 5
        return [
            card.cost / 5.0,
            float(card.playable),
            float(card.requires_target),
            float(card.upgraded),
            float(card.cost <= 0),
        ]

    @staticmethod
    def _enemy_action_summary(enemy: EnemyView | None) -> list[float]:
        if enemy is None:
            return [0.0] * 5
        return [
            enemy.hp / max(1, enemy.max_hp),
            enemy.block / 100.0,
            enemy.intent_damage / 50.0,
            enemy.intent_hits / 10.0,
            len(enemy.statuses) / 10.0,
        ]

    @staticmethod
    def _room_counts(values: Iterable[str]) -> list[float]:
        counts = [0.0] * len(_ROOM_SYMBOLS)
        total = 0
        for value in values:
            symbol = str(value).upper()
            if symbol in _ROOM_SYMBOLS:
                counts[_ROOM_SYMBOLS.index(symbol)] += 1.0
                total += 1
        scale = max(1, total)
        return [count / scale for count in counts]

    def _weighted_histogram(
        self,
        values: Iterable[tuple[str, int]],
        buckets: int,
    ) -> list[float]:
        result = [0.0] * buckets
        total = 0
        for value, count in values:
            if count <= 0:
                continue
            result[self._bucket(value, buckets)] += count
            total += count
        scale = max(1, total)
        return [value / scale for value in result]

    def _histogram(self, values: Iterable[str], buckets: int, scale: int) -> list[float]:
        result = [0.0] * buckets
        for value in values:
            if str(value):
                result[self._bucket(str(value), buckets)] += 1.0
        return [value / scale for value in result]

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

    @staticmethod
    def _validate(tensor: torch.Tensor, dimension: int, name: str) -> None:
        if tensor.ndim != 1 or tensor.shape[0] != dimension:
            raise AssertionError(f"{name} encoder feature dimension changed unexpectedly")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} encoder produced a non-finite feature")

    @staticmethod
    def _empty_observation() -> Observation:
        from sts_env.types import PlayerView

        return Observation(
            phase=Phase.TERMINAL,
            turn=0,
            player=PlayerView(hp=0, max_hp=1, block=0, energy=0),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=(),
        )
