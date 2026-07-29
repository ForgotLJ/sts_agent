from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable

import torch
from torch.nn import functional

from sts_env.env import StsEnv
from sts_env.trace import EpisodeTrace, TraceStep, observation_digest
from sts_env.training.recurrent_ppo import RecurrentPPOConfig, RecurrentPPOTrainer
from sts_env.training.m7b_replay import load_m7b_replay_batch
from sts_env.training.self_imitation import ImitationChunk, build_imitation_chunks
from sts_env.types import Action, Observation, Phase


M7B_FINAL_SEED_START = 3_000_000
M7B_FINAL_SEED_COUNT = 2_048
M7B_FINAL_SEED_END = M7B_FINAL_SEED_START + M7B_FINAL_SEED_COUNT - 1
M7B_SUPERVISED_PHASES = (
    Phase.CARD_REWARD,
    Phase.EVENT,
    Phase.MAP,
    Phase.REST_SITE,
    Phase.SHOP,
)


@dataclass(frozen=True, slots=True)
class M7BDistillationConfig:
    run_seed: int
    device: str = "cuda"
    max_epochs: int = 20
    early_stopping_patience: int = 3
    trace_batch_size: int = 64
    optimizer_batch_chunks: int = 16
    checkpoint_interval_batches: int = 1
    chunk_length: int = 64
    burn_in_steps: int = 16
    maximum_phase_multiplier: int = 4
    training_seed_start: int = 400_000
    training_seed_count: int = 4_096
    validation_seed_start: int = 1_500_000
    validation_seed_count: int = 512
    gate_seed_start: int = 1_600_000
    gate_seed_count: int = 512

    def __post_init__(self) -> None:
        counts = (
            self.max_epochs,
            self.early_stopping_patience,
            self.trace_batch_size,
            self.optimizer_batch_chunks,
            self.checkpoint_interval_batches,
            self.chunk_length,
            self.maximum_phase_multiplier,
            self.training_seed_count,
            self.validation_seed_count,
            self.gate_seed_count,
        )
        if self.run_seed < 0 or min(counts) <= 0 or self.burn_in_steps < 0:
            raise ValueError("M7-B distillation configuration is invalid")
        starts = (
            self.training_seed_start,
            self.validation_seed_start,
            self.gate_seed_start,
        )
        if min(starts) < 0:
            raise ValueError("M7-B seeds cannot be negative")
        ranges = self.seed_ranges()
        names = tuple(ranges)
        for index, name in enumerate(names):
            for other_name in names[index + 1 :]:
                if _ranges_overlap(ranges[name], ranges[other_name]):
                    raise ValueError(
                        f"M7-B seed ranges overlap: {name} and {other_name}"
                    )

    def seed_ranges(self) -> dict[str, range]:
        return {
            "training": range(
                self.training_seed_start,
                self.training_seed_start + self.training_seed_count,
            ),
            "validation": range(
                self.validation_seed_start,
                self.validation_seed_start + self.validation_seed_count,
            ),
            "gate": range(
                self.gate_seed_start,
                self.gate_seed_start + self.gate_seed_count,
            ),
            "final": range(M7B_FINAL_SEED_START, M7B_FINAL_SEED_END + 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M7BDistillationProgress:
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
            raise ValueError("M7-B progress cannot be negative")
        if self.completed and not self.completion_reason:
            raise ValueError("completed M7-B progress requires a reason")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.best_validation_key is not None:
            payload["best_validation_key"] = list(self.best_validation_key)
        return payload


@dataclass(frozen=True, slots=True)
class LoadedM7BCheckpoint:
    trainer: RecurrentPPOTrainer
    config: M7BDistillationConfig
    progress: M7BDistillationProgress
    metrics: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def record_m7b_teacher_trace(
    environment: StsEnv,
    policy: Callable[[Observation], Action],
    *,
    seed: int,
    max_steps: int,
    truncate_on_horizon: bool = False,
) -> EpisodeTrace:
    if seed < 0 or max_steps <= 0:
        raise ValueError("M7-B teacher collection arguments are invalid")
    observation, reset_info = environment.reset(seed=seed)
    initial_digest = observation_digest(observation)
    phase_counts = {phase.value: 0 for phase in M7B_SUPERVISED_PHASES}
    steps: list[TraceStep] = []
    environment_return = 0.0

    def finish_trace(*, horizon_truncated: bool) -> EpisodeTrace:
        return EpisodeTrace(
            seed=seed,
            initial_observation_digest=initial_digest,
            steps=tuple(steps),
            backend=str(reset_info.get("backend", "unknown")),
            metadata={
                "protocol": "m7b-teacher",
                "collection_max_steps": max_steps,
                "horizon_truncated": horizon_truncated,
                "phase_supervision_counts": phase_counts,
                "final_act": observation.act,
                "final_floor": observation.floor,
                "won": environment_return > 0,
                "environment_return": environment_return,
            },
        )

    for _ in range(max_steps):
        action = policy(observation)
        if observation.phase in M7B_SUPERVISED_PHASES and len(
            observation.legal_actions
        ) > 1:
            phase_counts[observation.phase.value] += 1
        observation, reward, terminated, truncated, info = environment.step(action)
        environment_return += reward
        steps.append(
            TraceStep(
                action=action,
                observation_digest=observation_digest(observation),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        if terminated or truncated:
            return finish_trace(horizon_truncated=False)
    if not truncate_on_horizon:
        raise RuntimeError(
            f"M7-B teacher episode did not finish within {max_steps} steps"
        )
    last_step = steps[-1]
    steps[-1] = TraceStep(
        action=last_step.action,
        observation_digest=last_step.observation_digest,
        reward=last_step.reward,
        terminated=last_step.terminated,
        truncated=True,
        info={**last_step.info, "m7b_horizon_truncated": True},
    )
    return finish_trace(horizon_truncated=True)


def build_m7b_corpus_manifest(
    corpus_directory: str | Path,
    *,
    seed_start: int,
    seed_count: int,
    collection_max_steps: int | None = None,
    allow_horizon_truncation: bool | None = None,
) -> dict[str, Any]:
    root = Path(corpus_directory).resolve()
    traces_root = root / "traces"
    if (
        seed_start < 0
        or seed_count <= 0
        or (collection_max_steps is not None and collection_max_steps <= 0)
        or (
            allow_horizon_truncation is not None
            and not isinstance(allow_horizon_truncation, bool)
        )
        or not traces_root.is_dir()
    ):
        raise ValueError("M7-B corpus manifest arguments are invalid")
    aggregate = hashlib.sha256()
    entries = []
    phase_counts = {phase.value: 0 for phase in M7B_SUPERVISED_PHASES}
    wins = 0
    final_floors = []
    horizon_truncations = 0
    for seed in range(seed_start, seed_start + seed_count):
        path = traces_root / f"seed-{seed:08d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        trace = EpisodeTrace.read_jsonl(path)
        if trace.seed != seed or (trace.metadata or {}).get("protocol") != "m7b-teacher":
            raise ValueError(f"invalid M7-B teacher trace: {path}")
        metadata = dict(trace.metadata or {})
        if (
            collection_max_steps is not None
            and int(metadata.get("collection_max_steps", -1))
            != collection_max_steps
        ):
            raise ValueError(f"M7-B trace has the wrong collection horizon: {path}")
        horizon_truncated = metadata.get("horizon_truncated")
        if allow_horizon_truncation is not None:
            if not isinstance(horizon_truncated, bool):
                raise ValueError(f"M7-B trace lacks horizon provenance: {path}")
            if horizon_truncated and not allow_horizon_truncation:
                raise ValueError(f"M7-B trace has a forbidden horizon truncation: {path}")
            if horizon_truncated and (
                not trace.steps[-1].truncated
                or trace.steps[-1].info.get("m7b_horizon_truncated") is not True
            ):
                raise ValueError(f"M7-B trace has invalid horizon truncation: {path}")
        trace_phase_counts = {
            str(name): int(count)
            for name, count in dict(
                metadata.get("phase_supervision_counts") or {}
            ).items()
        }
        if set(trace_phase_counts) != set(phase_counts) or min(
            trace_phase_counts.values()
        ) < 0:
            raise ValueError(f"invalid M7-B phase counts: {path}")
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        aggregate.update(b"\0")
        entries.append(
            {
                "seed": seed,
                "path": relative,
                "sha256": digest,
                "size": path.stat().st_size,
                "steps": len(trace.steps),
                "phase_supervision_counts": trace_phase_counts,
            }
        )
        for phase, count in trace_phase_counts.items():
            phase_counts[phase] += count
        wins += int(bool(metadata.get("won")))
        final_floors.append(int(metadata.get("final_floor", 0)))
        horizon_truncations += int(bool(horizon_truncated))
    if any(count <= 0 for count in phase_counts.values()):
        raise ValueError("M7-B corpus lacks one or more supervised phases")
    manifest = {
        "protocol": "m7b-teacher-corpus",
        "schema_version": 1,
        "complete": True,
        "errors": 0,
        "root": str(root),
        "seed_range": [seed_start, seed_start + seed_count - 1],
        "trace_count": len(entries),
        "aggregate_sha256": aggregate.hexdigest(),
        "phase_supervision_counts": phase_counts,
        "wins": wins,
        "mean_final_floor": sum(final_floors) / len(final_floors),
        "files": entries,
    }
    if collection_max_steps is not None:
        manifest["collection_max_steps"] = collection_max_steps
    if allow_horizon_truncation is not None:
        manifest["allow_horizon_truncation"] = allow_horizon_truncation
        manifest["horizon_truncations"] = horizon_truncations
    return manifest


def verify_m7b_corpus_manifest(
    manifest_path: str | Path,
    *,
    expected_seed_start: int | None = None,
    expected_seed_count: int | None = None,
    expected_collection_max_steps: int | None = None,
    expected_allow_horizon_truncation: bool | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected_allow_horizon_truncation is not None and not isinstance(
        expected_allow_horizon_truncation, bool
    ):
        raise ValueError("M7-B expected horizon-truncation policy must be boolean")
    if (
        payload.get("protocol") != "m7b-teacher-corpus"
        or int(payload.get("schema_version", -1)) != 1
        or payload.get("complete") is not True
        or int(payload.get("errors", -1)) != 0
    ):
        raise ValueError("invalid M7-B corpus manifest")
    seed_range = tuple(int(value) for value in payload["seed_range"])
    if expected_seed_start is not None and seed_range[0] != expected_seed_start:
        raise ValueError("M7-B corpus starts at the wrong seed")
    if expected_seed_count is not None and seed_range[1] - seed_range[0] + 1 != expected_seed_count:
        raise ValueError("M7-B corpus has the wrong seed count")
    declared_collection_max_steps = payload.get("collection_max_steps")
    if declared_collection_max_steps is not None:
        declared_collection_max_steps = int(declared_collection_max_steps)
        if declared_collection_max_steps <= 0:
            raise ValueError("M7-B corpus has an invalid collection horizon")
    if expected_collection_max_steps is not None:
        if expected_collection_max_steps <= 0:
            raise ValueError("M7-B expected collection horizon must be positive")
        if declared_collection_max_steps != expected_collection_max_steps:
            raise ValueError("M7-B corpus has the wrong collection horizon")
    declared_allow_horizon_truncation = payload.get("allow_horizon_truncation")
    if declared_allow_horizon_truncation is not None and not isinstance(
        declared_allow_horizon_truncation, bool
    ):
        raise ValueError("M7-B corpus has invalid horizon-truncation provenance")
    if (
        expected_allow_horizon_truncation is not None
        and declared_allow_horizon_truncation
        is not expected_allow_horizon_truncation
    ):
        raise ValueError("M7-B corpus has the wrong horizon-truncation policy")
    declared_root = Path(str(payload["root"]))
    root = declared_root if declared_root.is_dir() else path.parent
    if not (root / "traces").is_dir():
        raise FileNotFoundError(
            "M7-B corpus root is unavailable and the manifest is not beside its traces"
        )
    rebuilt = build_m7b_corpus_manifest(
        root,
        seed_start=seed_range[0],
        seed_count=seed_range[1] - seed_range[0] + 1,
        collection_max_steps=declared_collection_max_steps,
        allow_horizon_truncation=declared_allow_horizon_truncation,
    )
    for key in (
        "aggregate_sha256",
        "trace_count",
        "phase_supervision_counts",
        "seed_range",
        "collection_max_steps",
        "allow_horizon_truncation",
        "horizon_truncations",
    ):
        if key in rebuilt and rebuilt[key] != payload.get(key):
            raise ValueError(f"M7-B corpus differs from its manifest: {key}")
    payload["root"] = str(root.resolve())
    return payload


def corpus_trace_paths(manifest: dict[str, Any]) -> tuple[Path, ...]:
    root = Path(str(manifest["root"]))
    paths = tuple(root / str(entry["path"]) for entry in manifest["files"])
    if not paths:
        raise ValueError("M7-B corpus contains no traces")
    return paths


def phase_stratified_imitation_chunks(
    chunks: tuple[ImitationChunk, ...],
    *,
    maximum_multiplier: int,
    seed: int,
) -> tuple[ImitationChunk, ...]:
    if not chunks or maximum_multiplier <= 0:
        raise ValueError("M7-B phase sampling arguments are invalid")
    phase_indices = {tuple(Phase).index(phase) for phase in M7B_SUPERVISED_PHASES}
    grouped: dict[int, list[ImitationChunk]] = {index: [] for index in phase_indices}
    for chunk in chunks:
        if chunk.supervision_phases is None:
            raise ValueError("M7-B phase sampling requires annotated chunks")
        for phase_index in phase_indices:
            mask = chunk.supervision_phases == phase_index
            weights = torch.where(
                mask,
                chunk.supervision_weights,
                torch.zeros_like(chunk.supervision_weights),
            )
            if weights.any():
                grouped[phase_index].append(
                    replace(chunk, supervision_weights=weights)
                )
    populated = {index: values for index, values in grouped.items() if values}
    if not populated:
        raise ValueError("M7-B trace batch contains no supervised phases")
    target = max(len(values) for values in populated.values())
    source = random.Random(seed)
    sampled = []
    for phase_index, values in sorted(populated.items()):
        order = list(range(len(values)))
        source.shuffle(order)
        sample_count = min(target, len(values) * maximum_multiplier)
        sampled.extend(values[order[index % len(order)]] for index in range(sample_count))
    source.shuffle(sampled)
    return tuple(sampled)


def validate_m7b_training_objective(config: RecurrentPPOConfig) -> None:
    auxiliary_weights = {
        "value_loss_weight": config.value_loss_weight,
        "entropy_weight": config.entropy_weight,
        "uniform_exploration_weight": config.uniform_exploration_weight,
    }
    enabled = {
        name: value for name, value in auxiliary_weights.items() if value != 0.0
    }
    if enabled:
        raise ValueError(
            "M7-B requires pure teacher cross-entropy; auxiliary losses are enabled: "
            f"{enabled}"
        )


def train_m7b_chunk_batch(
    trainer: RecurrentPPOTrainer,
    chunks: tuple[ImitationChunk, ...],
) -> dict[str, Any]:
    validate_m7b_training_objective(trainer.config)
    batch = _collate_chunks(chunks, trainer.device)
    logits, _ = trainer.network.forward_sequence(
        batch["states"],
        batch["actions"],
        batch["action_masks"],
        trainer.initial_hidden(len(chunks)),
        batch["episode_starts"],
    )
    weights = batch["weights"]
    denominator = weights.sum().clamp_min(1.0)
    cross_entropies = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["chosen_actions"].reshape(-1),
        reduction="none",
    ).reshape_as(weights)
    cross_entropy = (cross_entropies * weights).sum() / denominator
    loss = cross_entropy
    if not torch.isfinite(loss):
        raise FloatingPointError("M7-B distillation loss is non-finite")
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainer.network.parameters(),
        trainer.config.gradient_clip_norm,
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("M7-B gradient norm is non-finite")
    trainer.optimizer.step()
    trainer.gradient_steps += 1
    return _batch_metrics(
        logits.detach(),
        cross_entropies.detach(),
        batch,
        loss=float(loss.item()),
        gradient_norm=float(gradient_norm.item()),
    )


def evaluate_m7b_imitation(
    trainer: RecurrentPPOTrainer,
    environment_factory: Callable[[], StsEnv],
    trace_paths: tuple[Path, ...],
    *,
    trace_batch_size: int,
    optimizer_batch_chunks: int,
    chunk_length: int,
    burn_in_steps: int,
    replay_batch_paths: tuple[Path, ...] | None = None,
    target_action_resolver_factory: Callable[
        [], Callable[[EpisodeTrace, int, Observation], Action]
    ] | None = None,
    trace_validator: Callable[[EpisodeTrace, Callable[[], StsEnv]], None] | None = None,
) -> dict[str, Any]:
    if (
        (not trace_paths and not replay_batch_paths)
        or min(trace_batch_size, optimizer_batch_chunks, chunk_length) <= 0
    ):
        raise ValueError("M7-B imitation evaluation arguments are invalid")
    totals = _empty_metric_totals()
    trainer.network.eval()
    try:
        with torch.no_grad():
            if replay_batch_paths is None:
                def raw_chunk_batches() -> Iterable[tuple[ImitationChunk, ...]]:
                    for start in range(0, len(trace_paths), trace_batch_size):
                        traces = tuple(
                            EpisodeTrace.read_jsonl(path)
                            for path in trace_paths[start : start + trace_batch_size]
                        )
                        if trace_validator is not None:
                            for trace in traces:
                                trace_validator(trace, environment_factory)
                        yield build_imitation_chunks(
                            environment_factory,
                            trainer,
                            traces,
                            chunk_length=chunk_length,
                            burn_in_steps=burn_in_steps,
                            sparse_unsupervised_actions=True,
                            target_action_resolver=(
                                None
                                if target_action_resolver_factory is None
                                else target_action_resolver_factory()
                            ),
                        )

                chunk_batches = raw_chunk_batches()
            else:
                chunk_batches = (
                    load_m7b_replay_batch(path) for path in replay_batch_paths
                )
            for chunks in chunk_batches:
                for chunk_start in range(0, len(chunks), optimizer_batch_chunks):
                    selected = chunks[chunk_start : chunk_start + optimizer_batch_chunks]
                    batch = _collate_chunks(selected, trainer.device)
                    logits, _ = trainer.network.forward_sequence(
                        batch["states"],
                        batch["actions"],
                        batch["action_masks"],
                        trainer.initial_hidden(len(selected)),
                        batch["episode_starts"],
                    )
                    cross_entropies = functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        batch["chosen_actions"].reshape(-1),
                        reduction="none",
                    ).reshape_as(batch["weights"])
                    _accumulate_metric_totals(
                        totals,
                        logits,
                        cross_entropies,
                        batch,
                    )
    finally:
        trainer.network.train()
    return finalize_m7b_metric_totals(totals)


def m7b_validation_selection_key(validation: dict[str, Any]) -> tuple[float, ...]:
    phases = dict(validation["phases"])
    accuracies = [float(phases[phase.value]["accuracy"]) for phase in M7B_SUPERVISED_PHASES]
    cross_entropies = [
        float(phases[phase.value]["cross_entropy"])
        for phase in M7B_SUPERVISED_PHASES
    ]
    return (
        min(accuracies),
        sum(accuracies) / len(accuracies),
        -sum(cross_entropies) / len(cross_entropies),
    )


def save_m7b_checkpoint(
    path: str | Path,
    *,
    trainer: RecurrentPPOTrainer,
    config: M7BDistillationConfig,
    progress: M7BDistillationProgress,
    metrics: tuple[dict[str, Any], ...],
    manifest: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(
        {
            "protocol": "m7b",
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


def load_m7b_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedM7BCheckpoint:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("protocol") != "m7b" or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported M7-B checkpoint schema")
    random.setstate(payload["global_python_rng_state"])
    progress_payload = dict(payload["progress"])
    key = progress_payload.get("best_validation_key")
    if key is not None:
        progress_payload["best_validation_key"] = tuple(float(value) for value in key)
    return LoadedM7BCheckpoint(
        trainer=RecurrentPPOTrainer.from_checkpoint(payload["trainer"], device=device),
        config=M7BDistillationConfig(**payload["config"]),
        progress=M7BDistillationProgress(**progress_payload),
        metrics=tuple(dict(metric) for metric in payload.get("metrics", ())),
        manifest=dict(payload.get("manifest") or {}),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collate_chunks(
    chunks: tuple[ImitationChunk, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not chunks:
        raise ValueError("M7-B cannot collate an empty chunk batch")
    if any(chunk.supervision_phases is None for chunk in chunks):
        raise ValueError("M7-B chunks require phase annotations")
    time_steps = max(chunk.states.shape[0] for chunk in chunks)
    maximum_actions = max(chunk.actions.shape[1] for chunk in chunks)
    batch_size = len(chunks)
    state_dimension = chunks[0].states.shape[1]
    action_dimension = chunks[0].actions.shape[2]
    states = torch.zeros(time_steps, batch_size, state_dimension)
    actions = torch.zeros(time_steps, batch_size, maximum_actions, action_dimension)
    masks = torch.zeros(time_steps, batch_size, maximum_actions, dtype=torch.bool)
    chosen = torch.zeros(time_steps, batch_size, dtype=torch.long)
    weights = torch.zeros(time_steps, batch_size)
    phases = torch.full((time_steps, batch_size), -1, dtype=torch.long)
    starts = torch.zeros(time_steps, batch_size, dtype=torch.bool)
    starts[0] = True
    for batch_index, chunk in enumerate(chunks):
        length = chunk.states.shape[0]
        action_count = chunk.actions.shape[1]
        states[:length, batch_index] = chunk.states
        actions[:length, batch_index, :action_count] = chunk.actions
        masks[:length, batch_index, :action_count] = chunk.action_masks
        chosen[:length, batch_index] = chunk.chosen_actions
        weights[:length, batch_index] = chunk.supervision_weights
        assert chunk.supervision_phases is not None
        phases[:length, batch_index] = chunk.supervision_phases
        if length < time_steps:
            masks[length:, batch_index, 0] = True
    return {
        "states": states.to(device),
        "actions": actions.to(device),
        "action_masks": masks.to(device),
        "chosen_actions": chosen.to(device),
        "weights": weights.to(device),
        "phases": phases.to(device),
        "episode_starts": starts.to(device),
    }


def _batch_metrics(
    logits: torch.Tensor,
    cross_entropies: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    loss: float,
    gradient_norm: float,
) -> dict[str, Any]:
    totals = _empty_metric_totals()
    _accumulate_metric_totals(totals, logits, cross_entropies, batch)
    return {
        **finalize_m7b_metric_totals(totals),
        "loss": loss,
        "gradient_norm": gradient_norm,
    }


def _empty_metric_totals() -> dict[str, Any]:
    return {
        phase.value: {"count": 0, "correct": 0, "cross_entropy_sum": 0.0}
        for phase in M7B_SUPERVISED_PHASES
    }


def _accumulate_metric_totals(
    totals: dict[str, Any],
    logits: torch.Tensor,
    cross_entropies: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> None:
    predictions = torch.argmax(logits, dim=-1)
    supervised = batch["weights"] > 0
    for phase in M7B_SUPERVISED_PHASES:
        phase_index = tuple(Phase).index(phase)
        mask = supervised & (batch["phases"] == phase_index)
        count = int(torch.count_nonzero(mask).item())
        if not count:
            continue
        entry = totals[phase.value]
        entry["count"] += count
        entry["correct"] += int(
            torch.count_nonzero(
                (predictions == batch["chosen_actions"]) & mask
            ).item()
        )
        entry["cross_entropy_sum"] += float(cross_entropies[mask].sum().item())


def merge_m7b_metric_totals(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    totals = _empty_metric_totals()
    for metric in metrics:
        for phase in M7B_SUPERVISED_PHASES:
            source = dict(metric["phases"].get(phase.value) or {})
            count = int(source.get("count", 0))
            totals[phase.value]["count"] += count
            totals[phase.value]["correct"] += int(source.get("correct", 0))
            totals[phase.value]["cross_entropy_sum"] += float(
                source.get("cross_entropy_sum", 0.0)
            )
    return finalize_m7b_metric_totals(totals)


def finalize_m7b_metric_totals(totals: dict[str, Any]) -> dict[str, Any]:
    phases = {}
    for phase in M7B_SUPERVISED_PHASES:
        entry = totals[phase.value]
        count = int(entry["count"])
        phases[phase.value] = {
            "count": count,
            "correct": int(entry["correct"]),
            "cross_entropy_sum": float(entry["cross_entropy_sum"]),
            "accuracy": entry["correct"] / count if count else 0.0,
            "cross_entropy": entry["cross_entropy_sum"] / count if count else 0.0,
        }
    counts = sum(entry["count"] for entry in phases.values())
    correct = sum(entry["correct"] for entry in phases.values())
    cross_entropy_sum = sum(entry["cross_entropy_sum"] for entry in phases.values())
    return {
        "supervised_steps": counts,
        "accuracy": correct / counts if counts else 0.0,
        "cross_entropy": cross_entropy_sum / counts if counts else 0.0,
        "phases": phases,
    }


def _ranges_overlap(left: range, right: range) -> bool:
    return max(left.start, right.start) < min(left.stop, right.stop)
