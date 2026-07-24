from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import torch

from sts_env.training.curriculum import CurriculumScheduler
from sts_env.training.recurrent_ppo import RecurrentPPOTrainer


M6_FINAL_SEED_START = 2_000_000
M6_FINAL_SEED_COUNT = 1_024
M6_FINAL_SEED_END = M6_FINAL_SEED_START + M6_FINAL_SEED_COUNT - 1


@dataclass(frozen=True, slots=True)
class M6TrainingConfig:
    run_seed: int
    num_environments: int = 16
    rollout_steps: int = 128
    total_updates: int = 10000
    checkpoint_interval: int = 100
    validation_interval: int = 100
    training_seed_start: int = 0
    training_seed_count: int = 1000000
    validation_seed_start: int = 1100000
    validation_seed_count: int = 2048
    device: str = "cuda"

    def __post_init__(self) -> None:
        counts = (
            self.num_environments,
            self.rollout_steps,
            self.total_updates,
            self.checkpoint_interval,
            self.validation_interval,
            self.training_seed_count,
            self.validation_seed_count,
        )
        if self.run_seed < 0 or min(counts) <= 0:
            raise ValueError("M6 training counts and run seed are invalid")
        training = range(
            self.training_seed_start,
            self.training_seed_start + self.training_seed_count,
        )
        validation = range(
            self.validation_seed_start,
            self.validation_seed_start + self.validation_seed_count,
        )
        if max(training.start, validation.start) < min(training.stop, validation.stop):
            raise ValueError("M6 training and validation seed ranges overlap")
        final = range(M6_FINAL_SEED_START, M6_FINAL_SEED_END + 1)
        if any(
            max(candidate.start, final.start) < min(candidate.stop, final.stop)
            for candidate in (training, validation)
        ):
            raise ValueError("M6 training and validation seeds overlap final-test seeds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_m6_evaluation_seed_range(
    seed_start: int,
    seed_count: int,
    *,
    final: bool,
) -> tuple[int, int]:
    if seed_count <= 0:
        raise ValueError("M6 evaluation seed count must be positive")
    seed_end = seed_start + seed_count - 1
    intersects_final = max(seed_start, M6_FINAL_SEED_START) <= min(
        seed_end,
        M6_FINAL_SEED_END,
    )
    if intersects_final and not final:
        raise ValueError("final-test seeds require an explicit --final acknowledgement")
    if final and (seed_start, seed_count) != (
        M6_FINAL_SEED_START,
        M6_FINAL_SEED_COUNT,
    ):
        raise ValueError(
            "formal final evaluation must use exactly seeds "
            f"{M6_FINAL_SEED_START}–{M6_FINAL_SEED_END}"
        )
    return seed_start, seed_end


def m6_validation_selection_key(
    stage_name: str,
    validation: dict[str, float],
) -> tuple[float, float]:
    primary = (
        float(validation["win_rate"])
        if stage_name == "full_run"
        else float(validation["completion_rate"])
    )
    return primary, float(validation["mean_floor"])


@dataclass(frozen=True, slots=True)
class LoadedM6Checkpoint:
    trainer: RecurrentPPOTrainer
    collector_state: dict[str, Any]
    scheduler: CurriculumScheduler
    config: M6TrainingConfig
    update_index: int
    metrics: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    parameter_ema_state: dict[str, Any] | None = None


def save_m6_checkpoint(
    path: str | Path,
    *,
    trainer: RecurrentPPOTrainer,
    collector_state: dict[str, Any],
    scheduler: CurriculumScheduler,
    config: M6TrainingConfig,
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
        "schema_version": 1,
        "trainer": trainer.checkpoint(),
        "collector": collector_state,
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "update_index": update_index,
        "metrics": list(metrics),
        "manifest": dict(manifest or {}),
        "parameter_ema": parameter_ema_state,
        "global_python_rng_state": random.getstate(),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_m6_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedM6Checkpoint:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported M6 checkpoint schema")
    trainer = RecurrentPPOTrainer.from_checkpoint(payload["trainer"], device=device)
    random.setstate(payload["global_python_rng_state"])
    return LoadedM6Checkpoint(
        trainer=trainer,
        collector_state=dict(payload["collector"]),
        scheduler=CurriculumScheduler.from_state_dict(payload["scheduler"]),
        config=M6TrainingConfig(**payload["config"]),
        update_index=int(payload["update_index"]),
        metrics=tuple(dict(metric) for metric in payload.get("metrics", ())),
        manifest=dict(payload.get("manifest") or {}),
        parameter_ema_state=(
            dict(payload["parameter_ema"])
            if payload.get("parameter_ema") is not None
            else None
        ),
    )
