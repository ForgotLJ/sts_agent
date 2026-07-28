from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import random
from typing import Callable

import torch
from torch.nn import functional as functional

from sts_env.env import StsEnv
from sts_env.differential import canonical_observation
from sts_env.trace import EpisodeTrace
from sts_env.training.curriculum import materialize_recovery_trace
from sts_env.training.recurrent_ppo import RecurrentPPOTrainer
from sts_env.types import Action, Observation, Phase


@dataclass(frozen=True, slots=True)
class SelfImitationConfig:
    enabled: bool = True
    minimum_traces: int = 4
    maximum_traces: int = 8
    maximum_candidate_traces: int = 8
    frontier_trace_repeats: int = 1
    chunk_length: int = 64
    burn_in_steps: int = 16
    epochs: int = 1
    corpus_capacity: int = 64
    interval_updates: int = 5

    def __post_init__(self) -> None:
        if (
            self.minimum_traces <= 0
            or self.maximum_traces < self.minimum_traces
            or self.maximum_candidate_traces <= 0
            or self.maximum_candidate_traces > self.maximum_traces
            or self.frontier_trace_repeats <= 0
            or self.chunk_length <= 0
            or self.burn_in_steps < 0
            or self.epochs <= 0
            or self.corpus_capacity < self.maximum_traces
            or self.interval_updates <= 0
        ):
            raise ValueError("self-imitation configuration is invalid")

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DaggerConfig:
    enabled: bool = True
    interval_updates: int = 25
    rounds: int = 2
    stage_rounds: dict[str, int] = field(default_factory=dict)
    episodes: int = 64
    max_steps: int = 1000
    chunk_length: int = 64
    burn_in_steps: int = 16
    epochs: int = 3
    training_seed_offset: int = 30000
    round_seed_stride: int = 100000
    card_reward_weight: float = 1.0
    event_weight: float = 1.0
    map_weight: float = 3.0
    rest_site_weight: float = 2.0
    shop_weight: float = 4.0

    def __post_init__(self) -> None:
        if (
            self.interval_updates <= 0
            or self.rounds <= 0
            or self.episodes <= 0
            or self.max_steps <= 0
            or self.chunk_length <= 0
            or self.burn_in_steps < 0
            or self.epochs <= 0
            or self.training_seed_offset < 0
            or self.round_seed_stride <= 0
            or any(
                not stage or rounds <= 0 or rounds > self.rounds
                for stage, rounds in self.stage_rounds.items()
            )
            or min(self.phase_weights().values()) <= 0
        ):
            raise ValueError("DAgger configuration is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def rounds_for_stage(self, stage: str) -> int:
        return self.stage_rounds.get(stage, self.rounds)

    def phase_weights(self) -> dict[Phase, float]:
        return {
            Phase.CARD_REWARD: self.card_reward_weight,
            Phase.EVENT: self.event_weight,
            Phase.MAP: self.map_weight,
            Phase.REST_SITE: self.rest_site_weight,
            Phase.SHOP: self.shop_weight,
        }


@dataclass(frozen=True, slots=True)
class ImitationChunk:
    states: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    chosen_actions: torch.Tensor
    supervision_weights: torch.Tensor
    supervision_phases: torch.Tensor | None = None


def build_imitation_chunks(
    environment_factory: Callable[[], StsEnv],
    trainer: RecurrentPPOTrainer,
    traces: tuple[EpisodeTrace, ...],
    *,
    chunk_length: int = 64,
    burn_in_steps: int = 16,
    recovery_environment_factory: Callable[[], StsEnv] | None = None,
) -> tuple[ImitationChunk, ...]:
    if not traces or chunk_length <= 0 or burn_in_steps < 0:
        raise ValueError("self-imitation requires traces and valid sequence lengths")
    chunks: list[ImitationChunk] = []
    for supplied_trace in traces:
        has_recovery_prefix = bool(
            (supplied_trace.metadata or {}).get("curriculum_source_trace")
        )
        if has_recovery_prefix:
            if recovery_environment_factory is None:
                raise ValueError(
                    "curriculum self-imitation requires a recovery environment factory"
                )
            trace = supplied_trace
            environment = recovery_environment_factory()
            if not hasattr(environment, "replay_recovery_trace"):
                raise TypeError("recovery environment cannot replay curriculum traces")
            observation = environment.replay_recovery_trace(trace.prefix(0))
        else:
            trace = materialize_recovery_trace(supplied_trace)
            environment = environment_factory()
            observation, _ = environment.reset(seed=trace.seed)
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        chosen: list[int] = []
        phases = []
        state_keys = [_state_key(observation)]
        for step in trace.steps:
            try:
                action_index = observation.legal_actions.index(step.action)
            except ValueError as error:
                raise ValueError("self-imitation trace contains a stale action") from error
            states.append(trainer.encoder.encode_state(observation))
            actions.append(trainer.encoder.encode_actions(observation))
            chosen.append(action_index)
            phases.append(observation.phase)
            observation, _, terminated, truncated, _ = environment.step(step.action)
            state_keys.append(_state_key(observation))
            if terminated or truncated:
                break
        retained = set(_loop_erased_action_indices(tuple(state_keys)))
        for start in range(0, len(states), chunk_length):
            stop = min(len(states), start + chunk_length)
            context_start = max(0, start - burn_in_steps)
            supervised = [index for index in range(start, stop) if index in retained]
            if not supervised:
                continue
            chunk_actions = actions[context_start:stop]
            maximum_actions = max(features.shape[0] for features in chunk_actions)
            padded = torch.zeros(
                stop - context_start,
                maximum_actions,
                trainer.encoder.action_dimension,
            )
            mask = torch.zeros(stop - context_start, maximum_actions, dtype=torch.bool)
            for time_index, features in enumerate(chunk_actions):
                padded[time_index, : features.shape[0]] = features
                mask[time_index, : features.shape[0]] = True
            weights = torch.zeros(stop - context_start)
            for action_index in supervised:
                if (
                    phases[action_index] is not Phase.COMBAT
                    and actions[action_index].shape[0] > 1
                ):
                    weights[action_index - context_start] = 1.0
            if not weights.any():
                continue
            chunks.append(
                ImitationChunk(
                    states=torch.stack(states[context_start:stop]),
                    actions=padded,
                    action_masks=mask,
                    chosen_actions=torch.tensor(chosen[context_start:stop], dtype=torch.long),
                    supervision_weights=weights,
                    supervision_phases=torch.tensor(
                        [tuple(Phase).index(phase) for phase in phases[context_start:stop]],
                        dtype=torch.long,
                    ),
                )
            )
    return tuple(chunks)


def is_self_imitation_candidate(
    stage_name: str,
    *,
    target_act: int | None,
    final_act: int,
    won: bool,
) -> bool:
    if target_act is not None:
        return final_act >= target_act
    if stage_name == "act3_clear":
        return won
    if stage_name == "full_run":
        return won or final_act >= 2
    return False


def imitation_trace_progress(trace: EpisodeTrace) -> tuple[int, int, int]:
    won = any(step.reward > 0 for step in trace.steps)
    maximum_floor = max(
        (int(step.info.get("floor", -1)) for step in trace.steps),
        default=-1,
    )
    return int(won), maximum_floor, -len(trace.steps)


def rank_imitation_traces(
    traces: tuple[EpisodeTrace, ...],
    limit: int,
) -> tuple[EpisodeTrace, ...]:
    if limit < 0:
        raise ValueError("imitation trace limit cannot be negative")
    return tuple(
        sorted(traces, key=imitation_trace_progress, reverse=True)[:limit]
    )


def select_weighted_frontier_traces(
    traces: tuple[EpisodeTrace, ...],
    limit: int,
    frontier_trace_repeats: int,
) -> tuple[EpisodeTrace, ...]:
    if frontier_trace_repeats <= 0:
        raise ValueError("frontier trace repeats must be positive")
    ranked = rank_imitation_traces(traces, limit)
    if not ranked:
        return ()
    return (ranked[0],) * frontier_trace_repeats + ranked[1:]


def collect_dagger_chunks(
    environment_factory: Callable[[], StsEnv],
    trainer: RecurrentPPOTrainer,
    teacher: Callable[[Observation], Action],
    seeds: tuple[int, ...],
    *,
    max_steps: int = 1000,
    chunk_length: int = 64,
    burn_in_steps: int = 16,
    phase_weights: dict[Phase, float] | None = None,
) -> tuple[ImitationChunk, ...]:
    if not seeds or max_steps <= 0 or chunk_length <= 0 or burn_in_steps < 0:
        raise ValueError("DAgger collection configuration is invalid")
    chunks: list[ImitationChunk] = []
    resolved_phase_weights = phase_weights or {}
    if any(weight <= 0 for weight in resolved_phase_weights.values()):
        raise ValueError("DAgger phase weights must be positive")
    for seed in seeds:
        environment = environment_factory()
        observation, _ = environment.reset(seed=seed)
        hidden = trainer.initial_hidden(1)
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        chosen: list[int] = []
        weights: list[float] = []
        phases: list[Phase] = []
        for _ in range(max_steps):
            policy = trainer.sample_actions((observation,), hidden, deterministic=True)
            student_index = int(policy.action_indices[0])
            student_action = observation.legal_actions[student_index]
            teacher_action = teacher(observation)
            try:
                teacher_index = observation.legal_actions.index(teacher_action)
            except ValueError as error:
                raise ValueError("DAgger teacher returned an illegal action") from error
            states.append(policy.state_features[0])
            actions.append(
                policy.action_features[0, : len(observation.legal_actions)].detach().cpu()
            )
            chosen.append(teacher_index)
            phases.append(observation.phase)
            if observation.phase is Phase.COMBAT or len(observation.legal_actions) <= 1:
                weights.append(0.0)
            else:
                weights.append(float(resolved_phase_weights.get(observation.phase, 1.0)))
            executed_action = teacher_action if observation.phase is Phase.COMBAT else student_action
            observation, _, terminated, truncated, _ = environment.step(executed_action)
            hidden = policy.next_hidden.detach()
            if terminated or truncated:
                break
        for start in range(0, len(states), chunk_length):
            stop = min(len(states), start + chunk_length)
            context_start = max(0, start - burn_in_steps)
            chunk_weights = torch.tensor(weights[context_start:stop])
            if not chunk_weights.any():
                continue
            chunk_actions = actions[context_start:stop]
            maximum_actions = max(features.shape[0] for features in chunk_actions)
            padded = torch.zeros(
                stop - context_start,
                maximum_actions,
                trainer.encoder.action_dimension,
            )
            mask = torch.zeros(stop - context_start, maximum_actions, dtype=torch.bool)
            for time_index, features in enumerate(chunk_actions):
                padded[time_index, : features.shape[0]] = features
                mask[time_index, : features.shape[0]] = True
            chunks.append(
                ImitationChunk(
                    states=torch.stack(states[context_start:stop]),
                    actions=padded,
                    action_masks=mask,
                    chosen_actions=torch.tensor(
                        chosen[context_start:stop],
                        dtype=torch.long,
                    ),
                    supervision_weights=chunk_weights,
                    supervision_phases=torch.tensor(
                        [
                            tuple(Phase).index(phase)
                            for phase in phases[context_start:stop]
                        ],
                        dtype=torch.long,
                    ),
                )
            )
    return tuple(chunks)


def balance_imitation_phase_weights(
    chunks: tuple[ImitationChunk, ...],
    *,
    maximum_multiplier: float = 4.0,
) -> tuple[ImitationChunk, ...]:
    if not chunks or maximum_multiplier < 1:
        raise ValueError("phase balancing requires chunks and a multiplier of at least one")
    phase_totals: dict[int, float] = {}
    for chunk in chunks:
        if chunk.supervision_phases is None:
            raise ValueError("phase balancing requires phase-annotated chunks")
        for phase_index in torch.unique(chunk.supervision_phases):
            index = int(phase_index)
            mask = chunk.supervision_phases == index
            weight = float(chunk.supervision_weights[mask].sum().item())
            if weight > 0:
                phase_totals[index] = phase_totals.get(index, 0.0) + weight
    if not phase_totals:
        raise ValueError("phase balancing found no supervised steps")
    target = sum(phase_totals.values()) / len(phase_totals)
    minimum_multiplier = 1.0 / maximum_multiplier
    multipliers = {
        phase_index: min(
            maximum_multiplier,
            max(minimum_multiplier, target / total),
        )
        for phase_index, total in phase_totals.items()
    }
    balanced = []
    for chunk in chunks:
        assert chunk.supervision_phases is not None
        weights = chunk.supervision_weights.clone()
        for phase_index, multiplier in multipliers.items():
            weights[chunk.supervision_phases == phase_index] *= multiplier
        balanced.append(replace(chunk, supervision_weights=weights))
    return tuple(balanced)


def imitation_phase_coverage(
    chunks: tuple[ImitationChunk, ...],
) -> dict[str, dict[str, float]]:
    coverage: dict[str, dict[str, float]] = {}
    phases = tuple(Phase)
    for chunk in chunks:
        if chunk.supervision_phases is None:
            raise ValueError("phase coverage requires phase-annotated chunks")
        for phase_index in torch.unique(chunk.supervision_phases):
            index = int(phase_index)
            mask = chunk.supervision_phases == index
            supervised = mask & chunk.supervision_weights.bool()
            if not supervised.any():
                continue
            name = phases[index].value
            entry = coverage.setdefault(name, {"steps": 0.0, "weight": 0.0})
            entry["steps"] += float(torch.count_nonzero(supervised).item())
            entry["weight"] += float(chunk.supervision_weights[supervised].sum().item())
    return dict(sorted(coverage.items()))


def dagger_training_seeds(
    config: DaggerConfig,
    *,
    training_seed_start: int,
    training_seed_count: int,
    update_index: int,
    round_index: int = 0,
) -> tuple[int, ...]:
    if training_seed_count <= 0 or config.episodes > training_seed_count:
        raise ValueError("DAgger seed batch exceeds the training seed range")
    if update_index <= 0 or update_index % config.interval_updates != 0:
        raise ValueError("DAgger seeds require a positive scheduled update")
    if round_index < 0 or round_index >= config.rounds:
        raise ValueError("DAgger round index is out of range")
    cycle_index = update_index // config.interval_updates - 1
    cycle_offset = (
        config.training_seed_offset
        + cycle_index * config.episodes
        + round_index * config.round_seed_stride
    ) % training_seed_count
    return tuple(
        training_seed_start + (cycle_offset + episode_index) % training_seed_count
        for episode_index in range(config.episodes)
    )


def train_self_imitation(
    trainer: RecurrentPPOTrainer,
    chunks: tuple[ImitationChunk, ...],
    *,
    epochs: int = 1,
    seed: int = 0,
) -> dict[str, float]:
    if not chunks or epochs <= 0:
        raise ValueError("self-imitation requires chunks and positive epochs")
    source = random.Random(seed)
    losses: list[float] = []
    cross_entropies: list[float] = []
    accuracies: list[float] = []
    gradient_norms: list[float] = []
    uniform_kls: list[float] = []
    supervised_steps = 0.0
    supervision_weight = 0.0
    for _ in range(epochs):
        order = list(range(len(chunks)))
        source.shuffle(order)
        for chunk_index in order:
            chunk = chunks[chunk_index]
            states = chunk.states[:, None].to(trainer.device)
            actions = chunk.actions[:, None].to(trainer.device)
            masks = chunk.action_masks[:, None].to(trainer.device)
            starts = torch.zeros(states.shape[0], 1, dtype=torch.bool, device=trainer.device)
            starts[0] = True
            logits, _ = trainer.network.forward_sequence(
                states,
                actions,
                masks,
                trainer.initial_hidden(1),
                starts,
            )
            chosen = chunk.chosen_actions.to(trainer.device)
            weights = chunk.supervision_weights.to(trainer.device)
            denominator = weights.sum().clamp_min(1.0)
            per_step_cross_entropy = functional.cross_entropy(
                logits[:, 0],
                chosen,
                reduction="none",
            )
            cross_entropy = (per_step_cross_entropy * weights).sum() / denominator
            masks = chunk.action_masks.to(trainer.device)
            log_probabilities = torch.log_softmax(logits[:, 0], dim=-1)
            valid_action_count = masks.sum(dim=-1).clamp_min(1)
            uniform_cross_entropy = -(
                log_probabilities.masked_fill(~masks, 0.0).sum(dim=-1)
                / valid_action_count
            )
            per_step_uniform_kl = uniform_cross_entropy - torch.log(
                valid_action_count.float()
            )
            uniform_kl = (per_step_uniform_kl * weights).sum() / denominator
            loss = (
                cross_entropy
                + trainer.config.uniform_exploration_weight * uniform_kl
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("self-imitation loss is non-finite")
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainer.network.parameters(),
                trainer.config.gradient_clip_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("self-imitation gradient norm is non-finite")
            trainer.optimizer.step()
            trainer.gradient_steps += 1
            losses.append(float(loss.item()))
            cross_entropies.append(float(cross_entropy.item()))
            correct = (torch.argmax(logits[:, 0], dim=-1) == chosen).float()
            accuracies.append(float((correct * weights).sum().item() / denominator.item()))
            gradient_norms.append(float(gradient_norm.item()))
            uniform_kls.append(float(uniform_kl.item()))
            supervised_steps += float(torch.count_nonzero(weights).item())
            supervision_weight += float(denominator.item())
    return {
        "loss": sum(losses) / len(losses),
        "cross_entropy": sum(cross_entropies) / len(cross_entropies),
        "accuracy": sum(accuracies) / len(accuracies),
        "gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "uniform_kl": sum(uniform_kls) / len(uniform_kls),
        "chunks": float(len(chunks)),
        "supervised_steps": supervised_steps / epochs,
        "supervision_weight": supervision_weight / epochs,
    }


def _state_key(observation: Observation) -> str:
    return json.dumps(
        canonical_observation(observation),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _loop_erased_action_indices(state_keys: tuple[str, ...]) -> tuple[int, ...]:
    if len(state_keys) < 2:
        return ()
    path_states = [state_keys[0]]
    path_actions: list[int] = []
    positions = {state_keys[0]: 0}
    for action_index, next_state in enumerate(state_keys[1:]):
        repeated_position = positions.get(next_state)
        if repeated_position is None:
            path_actions.append(action_index)
            path_states.append(next_state)
            positions[next_state] = len(path_states) - 1
            continue
        for removed_state in path_states[repeated_position + 1 :]:
            positions.pop(removed_state, None)
        del path_states[repeated_position + 1 :]
        del path_actions[repeated_position:]
    return tuple(path_actions)
