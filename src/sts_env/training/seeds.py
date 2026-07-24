from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedSplit:
    train_start: int
    train_count: int
    evaluation_start: int
    evaluation_count: int

    def __post_init__(self) -> None:
        if self.train_start < 0 or self.evaluation_start < 0:
            raise ValueError("seed starts must be non-negative")
        if self.train_count <= 0 or self.evaluation_count <= 0:
            raise ValueError("seed counts must be positive")
        if set(self.training_seeds).intersection(self.evaluation_seeds):
            raise ValueError("training and evaluation seeds must be disjoint")

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.train_start, self.train_start + self.train_count))

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        return tuple(
            range(self.evaluation_start, self.evaluation_start + self.evaluation_count)
        )
