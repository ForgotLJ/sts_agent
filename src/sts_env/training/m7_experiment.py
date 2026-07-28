from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import torch

from sts_env.training.curriculum import CurriculumScheduler
from sts_env.training.recurrent_ppo import RecurrentPPOTrainer


M7_FINAL_SEED_START = 3_000_000
M7_FINAL_SEED_COUNT = 2_048
M7_FINAL_SEED_END = M7_FINAL_SEED_START + M7_FINAL_SEED_COUNT - 1


@dataclass(frozen=True, slots=True)
class M7TrainingConfig:
    run_seed: int
    num_environments: int = 16
    rollout_steps: int = 64
    max_curriculum_updates: int = 5_000
    full_run_updates: int = 1_000
    checkpoint_interval: int = 25
    validation_interval: int = 25
    selection_interval: int = 250
    training_seed_start: int = 0
    training_seed_count: int = 1_000_000
    promotion_seed_start: int = 1_200_000
    promotion_seed_count: int = 128
    screening_seed_start: int = 1_210_000
    screening_seed_count: int = 1_024
    screening_batch_size: int = 128
    selection_seed_start: int = 1_300_000
    selection_seed_count: int = 512
    pilot_gate_seed_start: int = 1_400_000
    pilot_gate_seed_count: int = 512
    selection_combat_policy: str = "heuristic"
    selection_search_budget: int = 64
    device: str = "cuda"

    def __post_init__(self) -> None:
        counts = (
            self.num_environments,
            self.rollout_steps,
            self.max_curriculum_updates,
            self.full_run_updates,
            self.checkpoint_interval,
            self.validation_interval,
            self.selection_interval,
            self.training_seed_count,
            self.promotion_seed_count,
            self.screening_seed_count,
            self.screening_batch_size,
            self.selection_seed_count,
            self.pilot_gate_seed_count,
            self.selection_search_budget,
        )
        if self.run_seed < 0 or min(counts) <= 0:
            raise ValueError("M7 training counts and run seed are invalid")
        if self.screening_batch_size > self.screening_seed_count:
            raise ValueError("M7 screening batch exceeds its seed range")
        if self.selection_combat_policy not in {"heuristic", "belief-search"}:
            raise ValueError("M7 selection combat policy is unsupported")
        ranges = self.seed_ranges()
        names = tuple(ranges)
        for index, name in enumerate(names):
            for other_name in names[index + 1 :]:
                if _ranges_overlap(ranges[name], ranges[other_name]):
                    raise ValueError(
                        f"M7 seed ranges overlap: {name} and {other_name}"
                    )

    @property
    def maximum_total_updates(self) -> int:
        return self.max_curriculum_updates + self.full_run_updates

    def seed_ranges(self) -> dict[str, range]:
        return {
            "training": range(
                self.training_seed_start,
                self.training_seed_start + self.training_seed_count,
            ),
            "promotion": range(
                self.promotion_seed_start,
                self.promotion_seed_start + self.promotion_seed_count,
            ),
            "screening": range(
                self.screening_seed_start,
                self.screening_seed_start + self.screening_seed_count,
            ),
            "selection": range(
                self.selection_seed_start,
                self.selection_seed_start + self.selection_seed_count,
            ),
            "pilot_gate": range(
                self.pilot_gate_seed_start,
                self.pilot_gate_seed_start + self.pilot_gate_seed_count,
            ),
            "final": range(M7_FINAL_SEED_START, M7_FINAL_SEED_END + 1),
        }

    def training_seeds(self) -> tuple[int, ...]:
        return tuple(self.seed_ranges()["training"])

    def promotion_seeds(self) -> tuple[int, ...]:
        return tuple(self.seed_ranges()["promotion"])

    def screening_seeds(self, batch_index: int) -> tuple[int, ...]:
        if batch_index < 0:
            raise ValueError("M7 screening batch index cannot be negative")
        start = (batch_index * self.screening_batch_size) % self.screening_seed_count
        return tuple(
            self.screening_seed_start
            + (start + offset) % self.screening_seed_count
            for offset in range(self.screening_batch_size)
        )

    def selection_seeds(self) -> tuple[int, ...]:
        return tuple(self.seed_ranges()["selection"])

    def pilot_gate_seeds(self) -> tuple[int, ...]:
        return tuple(self.seed_ranges()["pilot_gate"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M7TrainingProgress:
    full_run_entry_update: int | None = None
    full_run_updates_completed: int = 0
    screening_batches_completed: int = 0
    selection_evaluations_completed: int = 0

    def __post_init__(self) -> None:
        values = (
            self.full_run_updates_completed,
            self.screening_batches_completed,
            self.selection_evaluations_completed,
        )
        if min(values) < 0:
            raise ValueError("M7 training progress cannot be negative")
        if self.full_run_entry_update is not None and self.full_run_entry_update < 0:
            raise ValueError("M7 full-run entry update cannot be negative")
        if self.full_run_entry_update is None and self.full_run_updates_completed:
            raise ValueError("M7 full-run progress requires an entry update")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_m7_fixed_budget_progress(
    config: M7TrainingConfig,
    progress: M7TrainingProgress,
    update_index: int,
) -> None:
    if progress.full_run_entry_update is None:
        raise ValueError("M7 completion checkpoint lacks full-run entry progress")
    if progress.full_run_updates_completed != config.full_run_updates:
        raise ValueError("M7 completion checkpoint has not exhausted its fixed budget")
    expected_update = progress.full_run_entry_update + progress.full_run_updates_completed
    if update_index != expected_update:
        raise ValueError("M7 completion checkpoint update count disagrees with progress")


def validate_m7_evaluation_seed_range(
    seed_start: int,
    seed_count: int,
    *,
    final: bool,
) -> tuple[int, int]:
    if seed_count <= 0:
        raise ValueError("M7 evaluation seed count must be positive")
    seed_end = seed_start + seed_count - 1
    final_range = range(M7_FINAL_SEED_START, M7_FINAL_SEED_END + 1)
    requested = range(seed_start, seed_end + 1)
    intersects_final = _ranges_overlap(requested, final_range)
    if intersects_final and not final:
        raise ValueError("M7 final-test seeds require explicit final acknowledgement")
    if final and (seed_start, seed_count) != (
        M7_FINAL_SEED_START,
        M7_FINAL_SEED_COUNT,
    ):
        raise ValueError(
            "formal M7 evaluation must use exactly seeds "
            f"{M7_FINAL_SEED_START}-{M7_FINAL_SEED_END}"
        )
    return seed_start, seed_end


def m7_validation_selection_key(validation: dict[str, float]) -> tuple[float, ...]:
    return (
        float(validation["win_rate"]),
        float(validation["act3_clear_rate"]),
        float(validation["act2_clear_rate"]),
        float(validation["act1_clear_rate"]),
        float(validation["mean_floor"]),
        float(validation["mean_proxy_score"]),
    )


@dataclass(frozen=True, slots=True)
class LoadedM7Checkpoint:
    trainer: RecurrentPPOTrainer
    collector_state: dict[str, Any]
    scheduler: CurriculumScheduler
    config: M7TrainingConfig
    progress: M7TrainingProgress
    update_index: int
    metrics: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    parameter_ema_state: dict[str, Any] | None = None


def save_m7_checkpoint(
    path: str | Path,
    *,
    trainer: RecurrentPPOTrainer,
    collector_state: dict[str, Any],
    scheduler: CurriculumScheduler,
    config: M7TrainingConfig,
    progress: M7TrainingProgress,
    update_index: int,
    metrics: tuple[dict[str, Any], ...] = (),
    manifest: dict[str, Any] | None = None,
    parameter_ema_state: dict[str, Any] | None = None,
) -> None:
    if update_index < 0:
        raise ValueError("update index must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = {
        "protocol": "m7",
        "schema_version": 1,
        "trainer": trainer.checkpoint(),
        "collector": collector_state,
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "progress": progress.to_dict(),
        "update_index": update_index,
        "metrics": list(metrics),
        "manifest": dict(manifest or {}),
        "parameter_ema": parameter_ema_state,
        "global_python_rng_state": random.getstate(),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_m7_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedM7Checkpoint:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("protocol") != "m7" or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported M7 checkpoint schema")
    trainer = RecurrentPPOTrainer.from_checkpoint(payload["trainer"], device=device)
    random.setstate(payload["global_python_rng_state"])
    return LoadedM7Checkpoint(
        trainer=trainer,
        collector_state=dict(payload["collector"]),
        scheduler=CurriculumScheduler.from_state_dict(payload["scheduler"]),
        config=M7TrainingConfig(**payload["config"]),
        progress=M7TrainingProgress(**payload.get("progress", {})),
        update_index=int(payload["update_index"]),
        metrics=tuple(dict(metric) for metric in payload.get("metrics", ())),
        manifest=dict(payload.get("manifest") or {}),
        parameter_ema_state=(
            dict(payload["parameter_ema"])
            if payload.get("parameter_ema") is not None
            else None
        ),
    )


def _ranges_overlap(left: range, right: range) -> bool:
    return max(left.start, right.start) < min(left.stop, right.stop)
