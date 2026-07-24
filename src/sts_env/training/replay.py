from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any

from sts_env.types import Action, Observation


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "reward": self.reward,
            "next_observation": self.next_observation.to_dict(),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReplayTransition:
        return cls(
            observation=Observation.from_dict(payload["observation"]),
            action=Action.from_dict(payload["action"]),
            reward=float(payload["reward"]),
            next_observation=Observation.from_dict(payload["next_observation"]),
            terminated=bool(payload["terminated"]),
            truncated=bool(payload["truncated"]),
            info=dict(payload.get("info") or {}),
        )


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._items: list[ReplayTransition] = []
        self._next_index = 0
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: ReplayTransition) -> None:
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[ReplayTransition, ...]:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if batch_size > len(self._items):
            raise ValueError("batch size exceeds replay size")
        return tuple(self._random.sample(self._items, batch_size))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "items": [item.to_dict() for item in self._items],
            "next_index": self._next_index,
            "random_state": self._random.getstate(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> ReplayBuffer:
        buffer = cls(capacity=int(payload["capacity"]))
        buffer._items = [ReplayTransition.from_dict(item) for item in payload["items"]]
        buffer._next_index = int(payload["next_index"])
        buffer._random.setstate(payload["random_state"])
        return buffer

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            header = {"type": "replay", "capacity": self.capacity}
            stream.write(json.dumps(header, sort_keys=True) + "\n")
            for item in self._items:
                stream.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path, seed: int = 0) -> ReplayBuffer:
        records = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records or records[0].get("type") != "replay":
            raise ValueError("replay JSONL must start with a replay header")
        buffer = cls(capacity=int(records[0]["capacity"]), seed=seed)
        for record in records[1:]:
            buffer.add(ReplayTransition.from_dict(record))
        return buffer
