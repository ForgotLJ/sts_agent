from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable

import torch

from sts_env.env import StsEnv
from sts_env.trace import EpisodeTrace, TraceStep, observation_digest
from sts_env.training.m7b_distillation import (
    M7B_SUPERVISED_PHASES,
    evaluate_m7b_imitation,
)
from sts_env.training.recurrent_ppo import RecurrentPPOTrainer
from sts_env.training.self_imitation import ImitationChunk, build_imitation_chunks
from sts_env.types import Action, Observation, Phase


M7C_DAGGER_TRACE_PROTOCOL = "m7c-dagger-trace"
M7C_DAGGER_CORPUS_PROTOCOL = "m7c-dagger-corpus"


@dataclass(frozen=True, slots=True)
class M7CDaggerLabel:
    step_index: int
    teacher_action: Action
    behavior_action_index: int
    student_action_index: int
    phase: Phase
    teacher_mixed: bool
    floor: int
    act: int
    legal_action_count: int
    policy_entropy: float
    policy_margin: float

    def __post_init__(self) -> None:
        if (
            self.step_index < 0
            or self.behavior_action_index < 0
            or self.student_action_index < 0
            or self.floor < 0
            or self.act < 0
            or self.legal_action_count <= 0
            or not math.isfinite(self.policy_entropy)
            or not math.isfinite(self.policy_margin)
        ):
            raise ValueError("invalid M7-C DAgger label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "teacher_action": self.teacher_action.to_dict(),
            "behavior_action_index": self.behavior_action_index,
            "student_action_index": self.student_action_index,
            "phase": self.phase.value,
            "teacher_mixed": self.teacher_mixed,
            "floor": self.floor,
            "act": self.act,
            "legal_action_count": self.legal_action_count,
            "policy_entropy": self.policy_entropy,
            "policy_margin": self.policy_margin,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> M7CDaggerLabel:
        return cls(
            step_index=int(payload["step_index"]),
            teacher_action=Action.from_dict(dict(payload["teacher_action"])),
            behavior_action_index=int(payload["behavior_action_index"]),
            student_action_index=int(payload["student_action_index"]),
            phase=Phase(str(payload["phase"])),
            teacher_mixed=bool(payload["teacher_mixed"]),
            floor=int(payload["floor"]),
            act=int(payload["act"]),
            legal_action_count=int(payload["legal_action_count"]),
            policy_entropy=float(payload["policy_entropy"]),
            policy_margin=float(payload["policy_margin"]),
        )


class M7CTeacherActionResolver:
    def __init__(self) -> None:
        self._labels_by_trace: dict[int, tuple[M7CDaggerLabel, ...]] = {}

    def __call__(
        self,
        trace: EpisodeTrace,
        step_index: int,
        observation: Observation,
    ) -> Action:
        labels = self._labels_by_trace.get(id(trace))
        if labels is None:
            labels = m7c_dagger_labels(trace)
            self._labels_by_trace[id(trace)] = labels
        if step_index >= len(labels):
            raise ValueError("M7-C trace label index is missing")
        label = labels[step_index]
        if label.step_index != step_index:
            raise ValueError("M7-C trace labels are not sequential")
        if label.phase is not observation.phase:
            raise ValueError("M7-C trace label phase differs from replay state")
        if label.legal_action_count != len(observation.legal_actions):
            raise ValueError("M7-C trace label legal-action count differs from replay state")
        try:
            behavior_index = observation.legal_actions.index(trace.steps[step_index].action)
        except ValueError as error:
            raise ValueError("M7-C trace behavior action is stale") from error
        if label.behavior_action_index != behavior_index:
            raise ValueError("M7-C trace behavior-action index differs from replay state")
        if label.teacher_action not in observation.legal_actions:
            raise ValueError("M7-C trace teacher action is stale")
        return label.teacher_action


def record_m7c_dagger_trace(
    environment: StsEnv,
    trainer: RecurrentPPOTrainer,
    teacher: Callable[[Observation], Action],
    *,
    seed: int,
    max_steps: int,
    teacher_mix_probability: float,
    mixing_seed: int,
    behavior_policy: dict[str, Any],
    teacher_identity: str,
    round_index: int,
) -> EpisodeTrace:
    if (
        seed < 0
        or max_steps <= 0
        or not 0.0 <= teacher_mix_probability <= 1.0
        or mixing_seed < 0
        or round_index < 0
        or not teacher_identity
        or not behavior_policy
    ):
        raise ValueError("M7-C DAgger trace collection arguments are invalid")
    observation, reset_info = environment.reset(seed=seed)
    initial_digest = observation_digest(observation)
    source = random.Random(mixing_seed)
    hidden = trainer.initial_hidden(1)
    labels: list[M7CDaggerLabel] = []
    steps: list[TraceStep] = []
    phase_counts = {phase.value: 0 for phase in M7B_SUPERVISED_PHASES}
    student_noncombat_steps = 0
    mixed_noncombat_steps = 0
    environment_return = 0.0

    def finish_trace(*, horizon_truncated: bool) -> EpisodeTrace:
        return EpisodeTrace(
            seed=seed,
            initial_observation_digest=initial_digest,
            steps=tuple(steps),
            backend=str(reset_info.get("backend", "unknown")),
            metadata={
                "protocol": M7C_DAGGER_TRACE_PROTOCOL,
                "round_index": round_index,
                "teacher_mix_probability": teacher_mix_probability,
                "mixing_seed": mixing_seed,
                "behavior_policy": dict(behavior_policy),
                "teacher_identity": teacher_identity,
                "phase_supervision_counts": phase_counts,
                "student_noncombat_steps": student_noncombat_steps,
                "mixed_noncombat_steps": mixed_noncombat_steps,
                "horizon_truncated": horizon_truncated,
                "final_act": observation.act,
                "final_floor": observation.floor,
                "won": environment_return > 0,
                "environment_return": environment_return,
                "dagger_labels": [label.to_dict() for label in labels],
            },
        )

    for step_index in range(max_steps):
        if not observation.legal_actions:
            raise RuntimeError("M7-C DAgger encountered a non-terminal empty action set")
        policy = trainer.sample_actions((observation,), hidden, deterministic=True)
        student_action_index = int(policy.action_indices[0])
        student_action = observation.legal_actions[student_action_index]
        teacher_action = teacher(observation)
        try:
            observation.legal_actions.index(teacher_action)
        except ValueError as error:
            raise ValueError("M7-C DAgger teacher returned an illegal action") from error
        teacher_mixed = (
            observation.phase is not Phase.COMBAT
            and source.random() < teacher_mix_probability
        )
        behavior_action = (
            teacher_action
            if observation.phase is Phase.COMBAT or teacher_mixed
            else student_action
        )
        behavior_action_index = observation.legal_actions.index(behavior_action)
        logits = policy.logits[0, : len(observation.legal_actions)].detach().cpu()
        probabilities = torch.softmax(logits, dim=0)
        entropy = float(
            -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item()
        )
        ordered_logits = torch.topk(logits, k=min(2, len(logits))).values
        margin = float(
            (ordered_logits[0] - ordered_logits[1]).item()
            if len(ordered_logits) == 2
            else 0.0
        )
        labels.append(
            M7CDaggerLabel(
                step_index=step_index,
                teacher_action=teacher_action,
                behavior_action_index=behavior_action_index,
                student_action_index=student_action_index,
                phase=observation.phase,
                teacher_mixed=teacher_mixed,
                floor=observation.floor,
                act=observation.act,
                legal_action_count=len(observation.legal_actions),
                policy_entropy=entropy,
                policy_margin=margin,
            )
        )
        if observation.phase in M7B_SUPERVISED_PHASES and len(observation.legal_actions) > 1:
            phase_counts[observation.phase.value] += 1
        if observation.phase is not Phase.COMBAT:
            if teacher_mixed:
                mixed_noncombat_steps += 1
            else:
                student_noncombat_steps += 1
        observation, reward, terminated, truncated, info = environment.step(behavior_action)
        environment_return += reward
        steps.append(
            TraceStep(
                action=behavior_action,
                observation_digest=observation_digest(observation),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        hidden = policy.next_hidden.detach()
        if terminated or truncated:
            return finish_trace(horizon_truncated=False)
    last_step = steps[-1]
    steps[-1] = TraceStep(
        action=last_step.action,
        observation_digest=last_step.observation_digest,
        reward=last_step.reward,
        terminated=last_step.terminated,
        truncated=True,
        info={**last_step.info, "m7c_horizon_truncated": True},
    )
    return finish_trace(horizon_truncated=True)


def m7c_dagger_labels(trace: EpisodeTrace) -> tuple[M7CDaggerLabel, ...]:
    metadata = dict(trace.metadata or {})
    if metadata.get("protocol") != M7C_DAGGER_TRACE_PROTOCOL:
        raise ValueError("trace is not an M7-C DAgger trace")
    labels = tuple(
        M7CDaggerLabel.from_dict(dict(payload))
        for payload in metadata.get("dagger_labels", ())
    )
    if len(labels) != len(trace.steps):
        raise ValueError("M7-C trace has the wrong teacher-label count")
    if not labels:
        raise ValueError("M7-C trace has no teacher labels")
    for index, label in enumerate(labels):
        if label.step_index != index:
            raise ValueError("M7-C trace teacher labels are not sequential")
    return labels


def validate_m7c_dagger_trace(
    trace: EpisodeTrace,
    environment_factory: Callable[[], StsEnv],
) -> None:
    labels = m7c_dagger_labels(trace)
    environment = environment_factory()
    observation, _ = environment.reset(seed=trace.seed)
    if observation_digest(observation) != trace.initial_observation_digest:
        raise ValueError("M7-C trace initial observation differs from replay")
    resolver = M7CTeacherActionResolver()
    for index, step in enumerate(trace.steps):
        resolver(trace, index, observation)
        if step.action not in observation.legal_actions:
            raise ValueError("M7-C trace behavior action is stale")
        observation, _, terminated, truncated, _ = environment.step(step.action)
        if observation_digest(observation) != step.observation_digest:
            raise ValueError("M7-C trace next observation differs from replay")
        horizon_truncated = bool(step.info.get("m7c_horizon_truncated"))
        if horizon_truncated and index != len(trace.steps) - 1:
            raise ValueError("M7-C horizon truncation is not the final trace step")
        if not horizon_truncated and (terminated or truncated) != (
            step.terminated or step.truncated
        ):
            raise ValueError("M7-C trace termination differs from replay")
        if terminated or truncated:
            if index != len(trace.steps) - 1:
                raise ValueError("M7-C trace continues after termination")
            return
        if horizon_truncated:
            return
    if labels:
        raise ValueError("M7-C trace did not terminate")


def build_m7c_imitation_chunks(
    environment_factory: Callable[[], StsEnv],
    trainer: RecurrentPPOTrainer,
    traces: tuple[EpisodeTrace, ...],
    *,
    chunk_length: int = 64,
    burn_in_steps: int = 16,
    validate_replay: bool = True,
) -> tuple[ImitationChunk, ...]:
    if validate_replay:
        for trace in traces:
            validate_m7c_dagger_trace(trace, environment_factory)
    return build_imitation_chunks(
        environment_factory,
        trainer,
        traces,
        chunk_length=chunk_length,
        burn_in_steps=burn_in_steps,
        sparse_unsupervised_actions=True,
        target_action_resolver=M7CTeacherActionResolver(),
        loop_erase=False,
    )


def evaluate_m7c_imitation(
    trainer: RecurrentPPOTrainer,
    environment_factory: Callable[[], StsEnv],
    trace_paths: tuple[Path, ...],
    *,
    trace_batch_size: int,
    optimizer_batch_chunks: int,
    chunk_length: int,
    burn_in_steps: int,
) -> dict[str, Any]:
    return evaluate_m7b_imitation(
        trainer,
        environment_factory,
        trace_paths,
        trace_batch_size=trace_batch_size,
        optimizer_batch_chunks=optimizer_batch_chunks,
        chunk_length=chunk_length,
        burn_in_steps=burn_in_steps,
        target_action_resolver_factory=M7CTeacherActionResolver,
        trace_validator=validate_m7c_dagger_trace,
    )


def build_m7c_corpus_manifest(
    corpus_directory: str | Path,
    *,
    seed_start: int,
    seed_count: int,
    round_index: int,
) -> dict[str, Any]:
    root = Path(corpus_directory).resolve()
    traces_root = root / "traces"
    if seed_start < 0 or seed_count <= 0 or round_index < 0 or not traces_root.is_dir():
        raise ValueError("M7-C corpus manifest arguments are invalid")
    aggregate = hashlib.sha256()
    entries = []
    phase_counts = {phase.value: 0 for phase in M7B_SUPERVISED_PHASES}
    behavior_policy: dict[str, Any] | None = None
    teacher_identity: str | None = None
    teacher_mix_probability: float | None = None
    wins = 0
    final_floors: list[int] = []
    horizon_truncations = 0
    for seed in range(seed_start, seed_start + seed_count):
        path = traces_root / f"seed-{seed:08d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        trace = EpisodeTrace.read_jsonl(path)
        labels = m7c_dagger_labels(trace)
        metadata = dict(trace.metadata or {})
        if trace.seed != seed or int(metadata.get("round_index", -1)) != round_index:
            raise ValueError(f"invalid M7-C trace: {path}")
        current_behavior_policy = dict(metadata.get("behavior_policy") or {})
        current_teacher_identity = str(metadata.get("teacher_identity") or "")
        current_mix_probability = float(metadata.get("teacher_mix_probability", -1.0))
        if not current_behavior_policy or not current_teacher_identity or not 0.0 <= current_mix_probability <= 1.0:
            raise ValueError(f"M7-C trace lacks collection provenance: {path}")
        if behavior_policy is None:
            behavior_policy = current_behavior_policy
            teacher_identity = current_teacher_identity
            teacher_mix_probability = current_mix_probability
        elif (
            behavior_policy != current_behavior_policy
            or teacher_identity != current_teacher_identity
            or teacher_mix_probability != current_mix_probability
        ):
            raise ValueError(f"M7-C corpus mixes incompatible collection provenance: {path}")
        for label in labels:
            if label.phase in M7B_SUPERVISED_PHASES and label.legal_action_count > 1:
                phase_counts[label.phase.value] += 1
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
                "teacher_labels": len(labels),
            }
        )
        wins += int(bool(metadata.get("won")))
        final_floors.append(int(metadata.get("final_floor", -1)))
        horizon_truncations += int(bool(metadata.get("horizon_truncated")))
    return {
        "protocol": M7C_DAGGER_CORPUS_PROTOCOL,
        "schema_version": 1,
        "complete": True,
        "errors": 0,
        "root": str(root),
        "round_index": round_index,
        "seed_range": [seed_start, seed_start + seed_count - 1],
        "trace_count": seed_count,
        "behavior_policy": behavior_policy,
        "teacher_identity": teacher_identity,
        "teacher_mix_probability": teacher_mix_probability,
        "phase_supervision_counts": phase_counts,
        "wins": wins,
        "horizon_truncations": horizon_truncations,
        "mean_final_floor": sum(final_floors) / len(final_floors),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
    }


def verify_m7c_corpus_manifest(
    manifest_path: str | Path,
    *,
    expected_seed_start: int | None = None,
    expected_seed_count: int | None = None,
    expected_round_index: int | None = None,
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != M7C_DAGGER_CORPUS_PROTOCOL
        or int(manifest.get("schema_version", -1)) != 1
        or manifest.get("complete") is not True
        or int(manifest.get("errors", -1)) != 0
    ):
        raise ValueError("invalid M7-C corpus manifest")
    seed_range = list(manifest.get("seed_range") or ())
    if len(seed_range) != 2:
        raise ValueError("M7-C corpus manifest lacks its seed range")
    observed_start, observed_end = (int(value) for value in seed_range)
    observed_count = observed_end - observed_start + 1
    if observed_start < 0 or observed_count <= 0:
        raise ValueError("M7-C corpus manifest has an invalid seed range")
    if expected_seed_start is not None and observed_start != expected_seed_start:
        raise ValueError("M7-C corpus manifest has the wrong seed start")
    if expected_seed_count is not None and observed_count != expected_seed_count:
        raise ValueError("M7-C corpus manifest has the wrong seed count")
    if expected_round_index is not None and int(manifest.get("round_index", -1)) != expected_round_index:
        raise ValueError("M7-C corpus manifest has the wrong round")
    root = Path(str(manifest.get("root") or ""))
    entries = tuple(dict(entry) for entry in manifest.get("files", ()))
    expected_seeds = list(range(observed_start, observed_end + 1))
    if (
        len(entries) != observed_count
        or int(manifest.get("trace_count", -1)) != observed_count
        or [int(entry.get("seed", -1)) for entry in entries] != expected_seeds
    ):
        raise ValueError("M7-C corpus manifest has incomplete trace entries")
    aggregate = hashlib.sha256()
    phase_counts = {phase.value: 0 for phase in M7B_SUPERVISED_PHASES}
    horizon_truncations = 0
    expected_behavior_policy = dict(manifest.get("behavior_policy") or {})
    expected_teacher_identity = str(manifest.get("teacher_identity") or "")
    expected_mix_probability = float(manifest.get("teacher_mix_probability", -1.0))
    if (
        not expected_behavior_policy
        or not expected_teacher_identity
        or not 0.0 <= expected_mix_probability <= 1.0
    ):
        raise ValueError("M7-C corpus manifest lacks collection provenance")
    for entry in entries:
        relative = str(entry.get("path") or "")
        trace_path = root / relative
        if not relative or not trace_path.is_file() or trace_path.stat().st_size != int(entry.get("size", -1)):
            raise ValueError(f"M7-C corpus trace is missing or resized: {trace_path}")
        digest = sha256_file(trace_path) if verify_file_hashes else str(entry.get("sha256") or "")
        if digest != str(entry.get("sha256") or ""):
            raise ValueError(f"M7-C corpus trace differs: {trace_path}")
        trace = EpisodeTrace.read_jsonl(trace_path)
        labels = m7c_dagger_labels(trace)
        metadata = dict(trace.metadata or {})
        if (
            trace.seed != int(entry["seed"])
            or int(metadata.get("round_index", -1)) != int(manifest.get("round_index", -1))
            or dict(metadata.get("behavior_policy") or {}) != expected_behavior_policy
            or str(metadata.get("teacher_identity") or "") != expected_teacher_identity
            or float(metadata.get("teacher_mix_probability", -1.0)) != expected_mix_probability
        ):
            raise ValueError(f"M7-C corpus trace provenance differs: {trace_path}")
        for label in labels:
            if label.phase in M7B_SUPERVISED_PHASES and label.legal_action_count > 1:
                phase_counts[label.phase.value] += 1
        horizon_truncations += int(bool(metadata.get("horizon_truncated")))
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        aggregate.update(b"\0")
    if aggregate.hexdigest() != manifest.get("aggregate_sha256"):
        raise ValueError("M7-C corpus aggregate hash differs")
    if phase_counts != dict(manifest.get("phase_supervision_counts") or {}):
        raise ValueError("M7-C corpus phase counts differ")
    if horizon_truncations != int(manifest.get("horizon_truncations", -1)):
        raise ValueError("M7-C corpus horizon-truncation count differs")
    return manifest


def m7c_corpus_trace_paths(manifest: dict[str, Any]) -> tuple[Path, ...]:
    root = Path(str(manifest["root"]))
    paths = tuple(root / str(entry["path"]) for entry in manifest["files"])
    if not paths:
        raise ValueError("M7-C corpus contains no traces")
    return paths


def summarize_m7c_on_policy_labels(traces: Iterable[EpisodeTrace]) -> dict[str, Any]:
    totals = _empty_diagnostic_totals()
    by_floor: dict[str, dict[str, int]] = {}
    for trace in traces:
        for label in m7c_dagger_labels(trace):
            if label.phase not in M7B_SUPERVISED_PHASES or label.legal_action_count <= 1:
                continue
            floor_entry = by_floor.setdefault(str(label.floor), {"count": 0, "correct": 0})
            correct = trace.steps[label.step_index].action == label.teacher_action
            _accumulate_diagnostic(totals["all_noncombat"], label, correct)
            _accumulate_diagnostic(floor_entry, label, correct)
            if not label.teacher_mixed:
                _accumulate_diagnostic(totals["student_behavior"], label, correct)
                _accumulate_diagnostic(totals["phases"][label.phase.value], label, correct)
    return {
        "protocol": "m7c-on-policy-diagnostic",
        "schema_version": 1,
        "all_noncombat": _finalize_diagnostic(totals["all_noncombat"]),
        "student_behavior": _finalize_diagnostic(totals["student_behavior"]),
        "phases": {
            phase.value: _finalize_diagnostic(totals["phases"][phase.value])
            for phase in M7B_SUPERVISED_PHASES
        },
        "floors": {
            floor: _finalize_diagnostic(entry)
            for floor, entry in sorted(by_floor.items(), key=lambda item: int(item[0]))
        },
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_diagnostic_totals() -> dict[str, Any]:
    return {
        "all_noncombat": {"count": 0, "correct": 0, "entropy_sum": 0.0, "margin_sum": 0.0},
        "student_behavior": {"count": 0, "correct": 0, "entropy_sum": 0.0, "margin_sum": 0.0},
        "phases": {
            phase.value: {"count": 0, "correct": 0, "entropy_sum": 0.0, "margin_sum": 0.0}
            for phase in M7B_SUPERVISED_PHASES
        },
    }


def _accumulate_diagnostic(
    entry: dict[str, float | int],
    label: M7CDaggerLabel,
    correct: bool,
) -> None:
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["correct"] = int(entry.get("correct", 0)) + int(correct)
    entry["entropy_sum"] = float(entry.get("entropy_sum", 0.0)) + label.policy_entropy
    entry["margin_sum"] = float(entry.get("margin_sum", 0.0)) + label.policy_margin


def _finalize_diagnostic(entry: dict[str, float | int]) -> dict[str, float | int | None]:
    count = int(entry.get("count", 0))
    return {
        "count": count,
        "correct": int(entry.get("correct", 0)),
        "agreement": None if count == 0 else int(entry.get("correct", 0)) / count,
        "mean_entropy": None if count == 0 else float(entry.get("entropy_sum", 0.0)) / count,
        "mean_margin": None if count == 0 else float(entry.get("margin_sum", 0.0)) / count,
    }
