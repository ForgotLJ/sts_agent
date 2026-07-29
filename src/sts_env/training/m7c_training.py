from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import torch

from sts_env.training.recurrent_ppo import RecurrentPPOTrainer


@dataclass(frozen=True, slots=True)
class M7CDaggerTrainingConfig:
    run_seed: int
    round_index: int
    device: str = "cuda"
    max_epochs: int = 3
    early_stopping_patience: int = 2
    trace_batch_size: int = 64
    optimizer_batch_chunks: int = 16
    checkpoint_interval_batches: int = 1
    chunk_length: int = 64
    burn_in_steps: int = 16
    maximum_phase_multiplier: int = 4

    def __post_init__(self) -> None:
        counts = (
            self.max_epochs,
            self.early_stopping_patience,
            self.trace_batch_size,
            self.optimizer_batch_chunks,
            self.checkpoint_interval_batches,
            self.chunk_length,
            self.maximum_phase_multiplier,
        )
        if self.run_seed < 0 or self.round_index < 0 or min(counts) <= 0:
            raise ValueError("M7-C DAgger training configuration is invalid")
        if self.burn_in_steps < 0:
            raise ValueError("M7-C DAgger burn-in must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M7CDaggerTrainingProgress:
    next_epoch: int = 0
    next_trace_batch: int = 0
    total_trace_batches_completed: int = 0
    epochs_without_improvement: int = 0
    best_validation_key: tuple[float, ...] | None = None
    completed: bool = False
    completion_reason: str = ""

    def __post_init__(self) -> None:
        values = (
            self.next_epoch,
            self.next_trace_batch,
            self.total_trace_batches_completed,
            self.epochs_without_improvement,
        )
        if min(values) < 0:
            raise ValueError("M7-C DAgger progress cannot be negative")
        if self.completed and not self.completion_reason:
            raise ValueError("completed M7-C DAgger progress requires a reason")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.best_validation_key is not None:
            payload["best_validation_key"] = list(self.best_validation_key)
        return payload


@dataclass(frozen=True, slots=True)
class LoadedM7CDaggerCheckpoint:
    trainer: RecurrentPPOTrainer
    config: M7CDaggerTrainingConfig
    progress: M7CDaggerTrainingProgress
    metrics: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def m7c_validation_selection_key(
    teacher_anchor: dict[str, Any] | None,
    on_policy: dict[str, Any],
    *,
    require_all_phases: bool = True,
) -> tuple[float, ...]:
    if teacher_anchor is None:
        return _single_validation_key(on_policy, require_all_phases=require_all_phases)
    on_policy_key = _single_validation_key(
        on_policy,
        require_all_phases=require_all_phases,
    )
    anchor_key = _single_validation_key(
        teacher_anchor,
        require_all_phases=require_all_phases,
    )
    return (
        on_policy_key[0],
        anchor_key[0],
        on_policy_key[1],
        anchor_key[1],
        on_policy_key[2],
        anchor_key[2],
    )


def save_m7c_checkpoint(
    path: str | Path,
    *,
    trainer: RecurrentPPOTrainer,
    config: M7CDaggerTrainingConfig,
    progress: M7CDaggerTrainingProgress,
    metrics: tuple[dict[str, Any], ...],
    manifest: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(
        {
            "protocol": "m7c-dagger",
            "schema_version": 1,
            "trainer": trainer.checkpoint(),
            "config": config.to_dict(),
            "progress": progress.to_dict(),
            "metrics": list(metrics),
            "manifest": dict(manifest),
            "global_python_rng_state": random.getstate(),
        },
        temporary,
    )
    temporary.replace(destination)


def load_m7c_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedM7CDaggerCheckpoint:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if (
        payload.get("protocol") != "m7c-dagger"
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError("unsupported M7-C DAgger checkpoint schema")
    random.setstate(payload["global_python_rng_state"])
    progress_payload = dict(payload["progress"])
    key = progress_payload.get("best_validation_key")
    if key is not None:
        progress_payload["best_validation_key"] = tuple(float(value) for value in key)
    return LoadedM7CDaggerCheckpoint(
        trainer=RecurrentPPOTrainer.from_checkpoint(payload["trainer"], device=device),
        config=M7CDaggerTrainingConfig(**payload["config"]),
        progress=M7CDaggerTrainingProgress(**progress_payload),
        metrics=tuple(dict(metric) for metric in payload.get("metrics", ())),
        manifest=dict(payload.get("manifest") or {}),
    )


def _single_validation_key(
    validation: dict[str, Any],
    *,
    require_all_phases: bool,
) -> tuple[float, float, float]:
    phases = dict(validation["phases"])
    values = tuple(
        dict(entry)
        for entry in phases.values()
        if int(dict(entry).get("count", 0)) > 0
    )
    if not values or (
        require_all_phases
        and len(values) != len(phases)
    ):
        raise ValueError("M7-C validation must cover every supervised phase")
    accuracies = tuple(float(entry["accuracy"]) for entry in values)
    cross_entropies = tuple(float(entry["cross_entropy"]) for entry in values)
    return (
        min(accuracies),
        sum(accuracies) / len(accuracies),
        -sum(cross_entropies) / len(cross_entropies),
    )
