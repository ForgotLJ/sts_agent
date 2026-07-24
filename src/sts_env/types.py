from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    COMBAT = "combat"
    CARD_REWARD = "card_reward"
    MAP = "map"
    SHOP = "shop"
    REST_SITE = "rest_site"
    EVENT = "event"
    TERMINAL = "terminal"


class ActionKind(str, Enum):
    PLAY_CARD = "play_card"
    END_TURN = "end_turn"
    USE_POTION = "use_potion"
    DISCARD_POTION = "discard_potion"
    CHOOSE_CARD = "choose_card"
    CHOOSE_MAP_NODE = "choose_map_node"
    CHOOSE_OPTION = "choose_option"
    BUY = "buy"
    REMOVE_CARD = "remove_card"
    LEAVE = "leave"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    source_id: int | str | None = None
    target_id: int | str | None = None
    choice_index: int | None = None
    label: str = ""
    option_type: str = ""
    description: str = ""
    amount: int = 0
    gold_cost: int = 0
    hp_cost: int = 0
    target_x: int = -1
    target_y: int = -1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Action:
        return cls(
            kind=ActionKind(payload["kind"]),
            source_id=payload.get("source_id"),
            target_id=payload.get("target_id"),
            choice_index=payload.get("choice_index"),
            label=str(payload.get("label", "")),
            option_type=str(payload.get("option_type", "")),
            description=str(payload.get("description", "")),
            amount=int(payload.get("amount", 0)),
            gold_cost=int(payload.get("gold_cost", 0)),
            hp_cost=int(payload.get("hp_cost", 0)),
            target_x=int(payload.get("target_x", -1)),
            target_y=int(payload.get("target_y", -1)),
        )


@dataclass(frozen=True, slots=True)
class CardView:
    instance_id: int
    card_id: str
    name: str
    cost: int
    upgraded: bool
    playable: bool
    requires_target: bool


@dataclass(frozen=True, slots=True)
class PlayerView:
    hp: int
    max_hp: int
    block: int
    energy: int
    gold: int = 0
    statuses: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class EnemyView:
    enemy_id: int
    name: str
    hp: int
    max_hp: int
    block: int
    intent_damage: int
    intent_hits: int
    monster_id: str = ""
    statuses: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class MapNodeView:
    x: int
    y: int
    symbol: str
    children: tuple[tuple[int, int], ...] = ()
    burning_elite: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MapNodeView:
        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            symbol=str(payload.get("symbol", "")),
            children=tuple(
                (int(child[0]), int(child[1]))
                if isinstance(child, (list, tuple))
                else (int(child["x"]), int(child["y"]))
                for child in payload.get("children", ())
            ),
            burning_elite=bool(payload.get("burning_elite", False)),
        )


@dataclass(frozen=True, slots=True)
class RunHistoryView:
    decisions: int = 0
    rooms_visited: int = 0
    combats_won: int = 0
    elites_won: int = 0
    bosses_won: int = 0
    acts_cleared: int = 0
    cards_added: int = 0
    cards_removed: int = 0
    potions_used: int = 0
    potions_discarded: int = 0
    gold_spent: int = 0
    hp_lost: int = 0
    hp_healed: int = 0
    recent_rooms: tuple[str, ...] = ()
    recent_actions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunHistoryView:
        return cls(
            decisions=int(payload.get("decisions", 0)),
            rooms_visited=int(payload.get("rooms_visited", 0)),
            combats_won=int(payload.get("combats_won", 0)),
            elites_won=int(payload.get("elites_won", 0)),
            bosses_won=int(payload.get("bosses_won", 0)),
            acts_cleared=int(payload.get("acts_cleared", 0)),
            cards_added=int(payload.get("cards_added", 0)),
            cards_removed=int(payload.get("cards_removed", 0)),
            potions_used=int(payload.get("potions_used", 0)),
            potions_discarded=int(payload.get("potions_discarded", 0)),
            gold_spent=int(payload.get("gold_spent", 0)),
            hp_lost=int(payload.get("hp_lost", 0)),
            hp_healed=int(payload.get("hp_healed", 0)),
            recent_rooms=tuple(str(room) for room in payload.get("recent_rooms", ())),
            recent_actions=tuple(str(action) for action in payload.get("recent_actions", ())),
        )


@dataclass(frozen=True, slots=True)
class Observation:
    phase: Phase
    turn: int
    player: PlayerView
    hand: tuple[CardView, ...]
    enemies: tuple[EnemyView, ...]
    draw_pile: tuple[tuple[str, int], ...]
    discard_pile: tuple[tuple[str, int], ...]
    exhaust_pile: tuple[tuple[str, int], ...]
    legal_actions: tuple[Action, ...]
    ascension: int = 0
    act: int = 0
    floor: int = 0
    map_x: int = -1
    map_y: int = -1
    screen_state: str = ""
    deck: tuple[tuple[str, int], ...] = ()
    relics: tuple[tuple[str, int], ...] = ()
    potions: tuple[str, ...] = ()
    map_nodes: tuple[MapNodeView, ...] = ()
    act_boss: str = ""
    ruby_key: bool = False
    emerald_key: bool = False
    sapphire_key: bool = False
    potion_capacity: int = 0
    history: RunHistoryView = field(default_factory=RunHistoryView)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "turn": self.turn,
            "player": asdict(self.player),
            "hand": [asdict(card) for card in self.hand],
            "enemies": [asdict(enemy) for enemy in self.enemies],
            "draw_pile": list(self.draw_pile),
            "discard_pile": list(self.discard_pile),
            "exhaust_pile": list(self.exhaust_pile),
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "ascension": self.ascension,
            "act": self.act,
            "floor": self.floor,
            "map_x": self.map_x,
            "map_y": self.map_y,
            "screen_state": self.screen_state,
            "deck": list(self.deck),
            "relics": list(self.relics),
            "potions": list(self.potions),
            "map_nodes": [asdict(node) for node in self.map_nodes],
            "act_boss": self.act_boss,
            "ruby_key": self.ruby_key,
            "emerald_key": self.emerald_key,
            "sapphire_key": self.sapphire_key,
            "potion_capacity": self.potion_capacity,
            "history": asdict(self.history),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Observation:
        player_payload = payload["player"]
        player = PlayerView(
            hp=int(player_payload["hp"]),
            max_hp=int(player_payload["max_hp"]),
            block=int(player_payload["block"]),
            energy=int(player_payload["energy"]),
            gold=int(player_payload.get("gold", 0)),
            statuses=tuple(
                (str(name), int(value)) for name, value in player_payload.get("statuses", ())
            ),
        )
        hand = tuple(CardView(**card) for card in payload["hand"])
        enemies = tuple(
            EnemyView(
                enemy_id=int(enemy["enemy_id"]),
                name=str(enemy["name"]),
                hp=int(enemy["hp"]),
                max_hp=int(enemy["max_hp"]),
                block=int(enemy["block"]),
                intent_damage=int(enemy["intent_damage"]),
                intent_hits=int(enemy["intent_hits"]),
                monster_id=str(enemy.get("monster_id", "")),
                statuses=tuple(
                    (str(name), int(value)) for name, value in enemy.get("statuses", ())
                ),
            )
            for enemy in payload["enemies"]
        )
        return cls(
            phase=Phase(payload["phase"]),
            turn=int(payload["turn"]),
            player=player,
            hand=hand,
            enemies=enemies,
            draw_pile=tuple((str(card_id), int(count)) for card_id, count in payload["draw_pile"]),
            discard_pile=tuple(
                (str(card_id), int(count)) for card_id, count in payload["discard_pile"]
            ),
            exhaust_pile=tuple(
                (str(card_id), int(count)) for card_id, count in payload["exhaust_pile"]
            ),
            legal_actions=tuple(Action.from_dict(action) for action in payload["legal_actions"]),
            ascension=int(payload.get("ascension", 0)),
            act=int(payload.get("act", 0)),
            floor=int(payload.get("floor", 0)),
            map_x=int(payload.get("map_x", -1)),
            map_y=int(payload.get("map_y", -1)),
            screen_state=str(payload.get("screen_state", "")),
            deck=tuple((str(card_id), int(count)) for card_id, count in payload.get("deck", ())),
            relics=tuple(
                (str(relic_id), int(value)) for relic_id, value in payload.get("relics", ())
            ),
            potions=tuple(str(potion_id) for potion_id in payload.get("potions", ())),
            map_nodes=tuple(
                MapNodeView.from_dict(node) for node in payload.get("map_nodes", ())
            ),
            act_boss=str(payload.get("act_boss", "")),
            ruby_key=bool(payload.get("ruby_key", False)),
            emerald_key=bool(payload.get("emerald_key", False)),
            sapphire_key=bool(payload.get("sapphire_key", False)),
            potion_capacity=int(payload.get("potion_capacity", 0)),
            history=RunHistoryView.from_dict(dict(payload.get("history") or {})),
        )
