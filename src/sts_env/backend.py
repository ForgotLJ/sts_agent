from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Self

from sts_env.types import Action, Observation


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class SimulatorBackend(Protocol):
    @property
    def supports_clone(self) -> bool:
        ...

    @property
    def supports_redeterminization(self) -> bool:
        ...

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        ...

    def step(self, action: Action) -> Transition:
        ...

    def clone(self) -> Self:
        ...

    def redeterminized_clone(
        self,
        search_seed: int,
        known_top: tuple[str, ...] = (),
        known_bottom: tuple[str, ...] = (),
    ) -> Self:
        ...
