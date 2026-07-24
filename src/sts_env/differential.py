from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from sts_env.types import Observation, Phase


@dataclass(frozen=True, slots=True)
class Difference:
    path: str
    reference: Any
    candidate: Any
    allowed: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AllowlistEntry:
    path_prefix: str
    reason: str


class DifferentialAllowlist:
    def __init__(self, entries: tuple[AllowlistEntry, ...] = ()):
        self.entries = entries

    @classmethod
    def from_json(cls, path: str | Path) -> DifferentialAllowlist:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            tuple(
                AllowlistEntry(path_prefix=str(entry["path_prefix"]), reason=str(entry["reason"]))
                for entry in payload["entries"]
            )
        )

    def match(self, path: str) -> AllowlistEntry | None:
        return next(
            (entry for entry in self.entries if path.startswith(entry.path_prefix)),
            None,
        )


def _stable_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _normalize_potion(value: str) -> str:
    normalized = _stable_id(value)
    if normalized in {"emptypotionid", "emptypotionslot", "potionslot"}:
        return "empty"
    return normalized


def canonical_observation(observation: Observation) -> dict[str, Any]:
    combat = observation.phase is Phase.COMBAT
    return {
        "phase": observation.phase.value,
        "turn": observation.turn if combat else 0,
        "run": {
            "ascension": observation.ascension,
            "act": observation.act,
            "floor": observation.floor,
            "map_x": observation.map_x,
            "map_y": observation.map_y,
            "act_boss": _stable_id(observation.act_boss),
            "ruby_key": observation.ruby_key,
            "emerald_key": observation.emerald_key,
            "sapphire_key": observation.sapphire_key,
            "potion_capacity": observation.potion_capacity,
            "map": tuple(
                sorted(
                    (
                        node.x,
                        node.y,
                        node.symbol,
                        tuple(sorted(node.children)),
                        node.burning_elite,
                    )
                    for node in observation.map_nodes
                )
            ),
        },
        "player": {
            "hp": observation.player.hp,
            "max_hp": observation.player.max_hp,
            "block": observation.player.block if combat else 0,
            "energy": observation.player.energy if combat else 0,
            "gold": observation.player.gold,
            "statuses": (
                tuple(
                    sorted(
                        (_stable_id(name), value)
                        for name, value in observation.player.statuses
                    )
                )
                if combat
                else ()
            ),
        },
        "deck": tuple(sorted((_stable_id(card_id), count) for card_id, count in observation.deck)),
        "relics": tuple(sorted((_stable_id(relic_id), value) for relic_id, value in observation.relics)),
        "potions": tuple(_normalize_potion(potion_id) for potion_id in observation.potions),
        "combat": {
            "hand": tuple(sorted(_stable_id(card.card_id) for card in observation.hand)),
            "draw_pile": tuple(
                sorted((_stable_id(card_id), count) for card_id, count in observation.draw_pile)
            ),
            "discard_pile": tuple(
                sorted((_stable_id(card_id), count) for card_id, count in observation.discard_pile)
            ),
            "exhaust_pile": tuple(
                sorted((_stable_id(card_id), count) for card_id, count in observation.exhaust_pile)
            ),
            "enemies": tuple(
                (
                    _stable_id(enemy.monster_id or enemy.name),
                    enemy.hp,
                    enemy.max_hp,
                    enemy.block,
                    enemy.intent_damage,
                    enemy.intent_hits,
                    tuple(sorted((_stable_id(name), value) for name, value in enemy.statuses)),
                )
                for enemy in observation.enemies
            ),
        }
        if combat
        else None,
    }


def compare_observations(
    reference: Observation,
    candidate: Observation,
    allowlist: DifferentialAllowlist | None = None,
) -> tuple[Difference, ...]:
    active_allowlist = allowlist or DifferentialAllowlist()
    differences: list[Difference] = []

    def compare(path: str, expected: Any, actual: Any) -> None:
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in sorted(set(expected) | set(actual)):
                compare(f"{path}.{key}" if path else key, expected.get(key), actual.get(key))
            return
        if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
            if len(expected) != len(actual):
                record(path + ".length", len(expected), len(actual))
            for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
                compare(f"{path}[{index}]", expected_item, actual_item)
            return
        if expected != actual:
            record(path, expected, actual)

    def record(path: str, expected: Any, actual: Any) -> None:
        entry = active_allowlist.match(path)
        differences.append(
            Difference(
                path=path,
                reference=expected,
                candidate=actual,
                allowed=entry is not None,
                reason=entry.reason if entry is not None else "",
            )
        )

    compare("", canonical_observation(reference), canonical_observation(candidate))
    return tuple(differences)
