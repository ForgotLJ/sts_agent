from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as functional

from sts_env.env import StsEnv
from sts_env.trace import EpisodeTrace, TraceStep, observation_digest, replay_trace
from sts_env.training.run_encoding import RunEncoderConfig, RunFeatureEncoder
from sts_env.types import Action, Observation, Phase


@dataclass(frozen=True, slots=True)
class RecurrentPPOConfig:
    recurrent_size: int = 128
    state_embedding_size: int = 128
    action_embedding_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    uniform_exploration_weight: float = 0.0
    gradient_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_environments: int = 8

    def __post_init__(self) -> None:
        sizes = (
            self.recurrent_size,
            self.state_embedding_size,
            self.action_embedding_size,
            self.update_epochs,
            self.minibatch_environments,
        )
        if min(sizes) <= 0:
            raise ValueError("network sizes, epochs, and minibatch size must be positive")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("learning rate, gamma, or GAE lambda is invalid")
        if self.clip_ratio <= 0 or self.value_clip < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("PPO clipping and gradient clipping values are invalid")
        if (
            self.value_loss_weight < 0
            or self.entropy_weight < 0
            or self.uniform_exploration_weight < 0
        ):
            raise ValueError("loss weights must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecurrentDynamicPolicyValue(nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        config: RecurrentPPOConfig,
    ):
        super().__init__()
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dimension, config.state_embedding_size),
            nn.LayerNorm(config.state_embedding_size),
            nn.Tanh(),
        )
        self.recurrent = nn.GRUCell(config.state_embedding_size, config.recurrent_size)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dimension, config.action_embedding_size),
            nn.LayerNorm(config.action_embedding_size),
            nn.Tanh(),
        )
        self.policy_context = nn.Linear(config.recurrent_size, config.action_embedding_size)
        self.policy_bias = nn.Linear(config.action_embedding_size, 1)
        self.value_head = nn.Sequential(
            nn.Linear(config.recurrent_size, config.recurrent_size),
            nn.Tanh(),
            nn.Linear(config.recurrent_size, 1),
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.recurrent_size, device=device)

    def forward_step(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not action_mask.any(dim=-1).all():
            raise ValueError("every non-terminal policy state must contain a legal action")
        state_embedding = self.state_encoder(state_features)
        next_hidden = self.recurrent(state_embedding, hidden)
        action_embedding = self.action_encoder(action_features)
        context = self.policy_context(next_hidden).unsqueeze(1)
        logits = (
            (action_embedding * context).sum(dim=-1)
            / math.sqrt(self.config.action_embedding_size)
            + self.policy_bias(action_embedding).squeeze(-1)
        )
        logits = logits.masked_fill(~action_mask, -1e9)
        value = self.value_head(next_hidden).squeeze(-1)
        return logits, value, next_hidden

    def forward_sequence(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = initial_hidden
        logits: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for time_index in range(states.shape[0]):
            hidden = hidden * (~episode_starts[time_index]).unsqueeze(-1)
            step_logits, step_values, hidden = self.forward_step(
                states[time_index],
                actions[time_index],
                action_masks[time_index],
                hidden,
            )
            logits.append(step_logits)
            values.append(step_values)
        return torch.stack(logits), torch.stack(values)


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    state_features: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor
    next_hidden: torch.Tensor
    action_indices: torch.Tensor
    log_probabilities: torch.Tensor


@dataclass(frozen=True, slots=True)
class RunEpisode:
    environment_index: int
    seed: int
    length: int
    environment_return: float
    final_hp: int
    final_floor: int
    final_act: int
    won: bool


def _recovery_trace_step(
    action: Action,
    observation: Observation,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> TraceStep:
    curriculum_completed = bool(info.get("curriculum_completed", False))
    curriculum_timeout = bool(info.get("curriculum_timeout", False))
    curriculum_loop_detected = bool(info.get("curriculum_loop_detected", False))
    recovery_info = {
        key: value
        for key, value in info.items()
        if key
        not in {
            "raw_reward",
            "curriculum_stage",
            "curriculum_completed",
            "curriculum_timeout",
            "curriculum_loop_detected",
            "curriculum_repeat_count",
            "curriculum_progress_reward",
        }
    }
    return TraceStep(
        action=action,
        observation_digest=observation_digest(observation),
        reward=float(info.get("raw_reward", reward)),
        terminated=terminated,
        truncated=(
            truncated
            and not curriculum_completed
            and not curriculum_timeout
            and not curriculum_loop_detected
        ),
        info=recovery_info,
    )


@dataclass(frozen=True, slots=True)
class RecurrentRolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    chosen_actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    episode_starts: torch.Tensor
    policy_weights: torch.Tensor
    initial_hidden: torch.Tensor
    bootstrap_values: torch.Tensor
    completed_episodes: tuple[RunEpisode, ...]
    completed_traces: tuple[EpisodeTrace, ...]

    @property
    def time_steps(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_environments(self) -> int:
        return int(self.states.shape[1])


class RecurrentPPOTrainer:
    def __init__(
        self,
        config: RecurrentPPOConfig | None = None,
        encoder: RunFeatureEncoder | None = None,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.config = config or RecurrentPPOConfig()
        self.encoder = encoder or RunFeatureEncoder()
        self.seed = seed
        self.device = torch.device(device)
        self._random = random.Random(seed)
        torch.manual_seed(seed)
        self.network = RecurrentDynamicPolicyValue(
            self.encoder.state_dimension,
            self.encoder.action_dimension,
            self.config,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
        )
        self.environment_steps = 0
        self.gradient_steps = 0

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        return self.network.initial_hidden(batch_size, self.device)

    def sample_actions(
        self,
        observations: tuple[Observation, ...],
        hidden: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> PolicyBatch:
        states, actions, action_mask = self._encode_batch(observations)
        with torch.no_grad():
            logits, values, next_hidden = self.network.forward_step(
                states.to(self.device),
                actions.to(self.device),
                action_mask.to(self.device),
                hidden.to(self.device),
            )
            distribution = torch.distributions.Categorical(logits=logits)
            action_indices = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
            log_probabilities = distribution.log_prob(action_indices)
        return PolicyBatch(
            state_features=states,
            action_features=actions,
            action_mask=action_mask,
            logits=logits,
            values=values,
            next_hidden=next_hidden,
            action_indices=action_indices,
            log_probabilities=log_probabilities,
        )

    def bootstrap_values(
        self,
        observations: tuple[Observation, ...],
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        policy = self.sample_actions(observations, hidden, deterministic=True)
        return policy.values.detach().cpu()

    def update(self, rollout: RecurrentRolloutBatch) -> dict[str, float]:
        advantages, returns = self._advantages(rollout)
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        metrics: list[dict[str, float]] = []
        environment_count = rollout.num_environments
        for _ in range(self.config.update_epochs):
            order = torch.randperm(environment_count)
            for start in range(0, environment_count, self.config.minibatch_environments):
                indices = order[start : start + self.config.minibatch_environments]
                metrics.append(
                    self._update_minibatch(
                        rollout,
                        normalized_advantages,
                        returns,
                        indices,
                    )
                )
        self.environment_steps += rollout.time_steps * rollout.num_environments
        return {
            key: sum(metric[key] for metric in metrics) / len(metrics)
            for key in metrics[0]
        }

    def _update_minibatch(
        self,
        rollout: RecurrentRolloutBatch,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        environment_indices: torch.Tensor,
    ) -> dict[str, float]:
        states = rollout.states[:, environment_indices].to(self.device)
        actions = rollout.actions[:, environment_indices].to(self.device)
        masks = rollout.action_masks[:, environment_indices].to(self.device)
        starts = rollout.episode_starts[:, environment_indices].to(self.device)
        initial_hidden = rollout.initial_hidden[environment_indices].to(self.device)
        logits, values = self.network.forward_sequence(
            states,
            actions,
            masks,
            initial_hidden,
            starts,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        chosen = rollout.chosen_actions[:, environment_indices].to(self.device)
        new_log_probabilities = distribution.log_prob(chosen)
        old_log_probabilities = rollout.old_log_probabilities[:, environment_indices].to(
            self.device
        )
        ratio = torch.exp(new_log_probabilities - old_log_probabilities)
        minibatch_advantages = advantages[:, environment_indices].to(self.device)
        policy_weights = rollout.policy_weights[:, environment_indices].to(self.device)
        unclipped = ratio * minibatch_advantages
        clipped = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio,
        ) * minibatch_advantages
        policy_denominator = policy_weights.sum().clamp_min(1.0)
        policy_loss = -(
            torch.minimum(unclipped, clipped) * policy_weights
        ).sum() / policy_denominator
        entropy = (distribution.entropy() * policy_weights).sum() / policy_denominator
        log_probabilities = torch.log_softmax(logits, dim=-1)
        valid_action_count = masks.sum(dim=-1).clamp_min(1)
        uniform_cross_entropy = -(
            log_probabilities.masked_fill(~masks, 0.0).sum(dim=-1)
            / valid_action_count
        )
        uniform_kl = (
            (uniform_cross_entropy - torch.log(valid_action_count.float()))
            * policy_weights
        ).sum() / policy_denominator

        old_values = rollout.old_values[:, environment_indices].to(self.device)
        expected_returns = returns[:, environment_indices].to(self.device)
        clipped_values = old_values + torch.clamp(
            values - old_values,
            -self.config.value_clip,
            self.config.value_clip,
        )
        value_loss = 0.5 * torch.maximum(
            (values - expected_returns).square(),
            (clipped_values - expected_returns).square(),
        ).mean()
        total_loss = (
            policy_loss
            + self.config.value_loss_weight * value_loss
            - self.config.entropy_weight * entropy
            + self.config.uniform_exploration_weight * uniform_kl
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("recurrent PPO loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("recurrent PPO gradient norm is non-finite")
        self.optimizer.step()
        self.gradient_steps += 1
        approximate_kl = (old_log_probabilities - new_log_probabilities).mean()
        clip_fraction = (
            (torch.abs(ratio - 1.0) > self.config.clip_ratio).float() * policy_weights
        ).sum() / policy_denominator
        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "uniform_kl": float(uniform_kl.item()),
            "gradient_norm": float(gradient_norm.item()),
            "approximate_kl": float(approximate_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
        }

    def _advantages(
        self,
        rollout: RecurrentRolloutBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rollout.rewards)
        accumulator = torch.zeros(rollout.num_environments)
        for time_index in range(rollout.time_steps - 1, -1, -1):
            next_values = (
                rollout.bootstrap_values
                if time_index == rollout.time_steps - 1
                else rollout.old_values[time_index + 1]
            )
            nonterminal = 1.0 - rollout.dones[time_index].float()
            delta = (
                rollout.rewards[time_index]
                + self.config.gamma * next_values * nonterminal
                - rollout.old_values[time_index]
            )
            accumulator = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * nonterminal
                * accumulator
            )
            advantages[time_index] = accumulator
        return advantages, advantages + rollout.old_values

    def _encode_batch(
        self,
        observations: tuple[Observation, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not observations or any(not observation.legal_actions for observation in observations):
            raise ValueError("policy batch must contain non-terminal observations")
        states = torch.stack([self.encoder.encode_state(observation) for observation in observations])
        encoded_actions = [self.encoder.encode_actions(observation) for observation in observations]
        maximum_actions = max(features.shape[0] for features in encoded_actions)
        actions = torch.zeros(
            len(observations),
            maximum_actions,
            self.encoder.action_dimension,
        )
        mask = torch.zeros(len(observations), maximum_actions, dtype=torch.bool)
        for environment_index, features in enumerate(encoded_actions):
            action_count = features.shape[0]
            actions[environment_index, :action_count] = features
            mask[environment_index, :action_count] = True
        return states, actions, mask

    def checkpoint(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "encoder_config": self.encoder.config.to_dict(),
            "seed": self.seed,
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "python_rng_state": self._random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> RecurrentPPOTrainer:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        return cls.from_checkpoint(payload, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict[str, Any],
        *,
        device: str | torch.device = "cpu",
    ) -> RecurrentPPOTrainer:
        trainer = cls(
            config=RecurrentPPOConfig(**payload["config"]),
            encoder=RunFeatureEncoder(RunEncoderConfig(**payload["encoder_config"])),
            seed=int(payload["seed"]),
            device=device,
        )
        trainer.network.load_state_dict(payload["network"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.environment_steps = int(payload["environment_steps"])
        trainer.gradient_steps = int(payload["gradient_steps"])
        trainer._random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("torch_cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["torch_cuda_rng_state"]]
            )
        return trainer


class RecurrentRolloutCollector:
    def __init__(
        self,
        environment_factory: Callable[[], StsEnv],
        trainer: RecurrentPPOTrainer,
        num_environments: int,
        seeds: tuple[int, ...],
        combat_selector: Callable[[StsEnv], Action] | None = None,
    ):
        if num_environments <= 0 or len(seeds) < num_environments:
            raise ValueError("collector requires a positive environment count and enough seeds")
        self.trainer = trainer
        self._environments = [environment_factory() for _ in range(num_environments)]
        self._seeds = seeds
        self._next_seed_index = 0
        self._combat_selector = combat_selector
        self._observations: list[Observation] = []
        self._active_seeds: list[int] = []
        self._episode_lengths = [0] * num_environments
        self._episode_returns = [0.0] * num_environments
        self._episode_starts = torch.ones(num_environments, dtype=torch.bool)
        self._hidden = trainer.initial_hidden(num_environments)
        self._traces: list[EpisodeTrace] = []
        for environment in self._environments:
            seed = self._next_seed()
            observation, info = environment.reset(seed=seed)
            self._observations.append(observation)
            self._active_seeds.append(seed)
            self._traces.append(
                EpisodeTrace(
                    seed=int(info.get("seed", seed)),
                    initial_observation_digest=observation_digest(observation),
                    steps=(),
                    backend=str(info.get("backend", "unknown")),
                    metadata=dict(info),
                )
            )

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def collect(self, steps_per_environment: int) -> RecurrentRolloutBatch:
        if steps_per_environment <= 0:
            raise ValueError("steps_per_environment must be positive")
        initial_hidden = self._hidden.detach().cpu().clone()
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        chosen: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        starts: list[torch.Tensor] = []
        policy_weights: list[torch.Tensor] = []
        completed: list[RunEpisode] = []
        completed_traces: list[EpisodeTrace] = []

        for rollout_step in range(steps_per_environment):
            observations = tuple(self._observations)
            invalid_observations = [
                environment_index
                for environment_index, observation in enumerate(observations)
                if observation.phase is Phase.TERMINAL or not observation.legal_actions
            ]
            if invalid_observations:
                details = []
                for environment_index in invalid_observations:
                    trace = self._traces[environment_index]
                    last = trace.steps[-1] if trace.steps else None
                    details.append(
                        {
                            "environment_index": environment_index,
                            "active_seed": self._active_seeds[environment_index],
                            "trace_seed": trace.seed,
                            "trace_steps": len(trace.steps),
                            "phase": observations[environment_index].phase.value,
                            "floor": observations[environment_index].floor,
                            "last_action": None if last is None else last.action.to_dict(),
                            "last_terminated": None if last is None else last.terminated,
                            "last_truncated": None if last is None else last.truncated,
                            "last_info": None if last is None else last.info,
                        }
                    )
                raise RuntimeError(
                    f"collector has terminal observations before rollout step {rollout_step}: "
                    f"{details}"
                )
            policy = self.trainer.sample_actions(observations, self._hidden)
            selected_indices = policy.action_indices.detach().cpu().clone()
            selected_log_probabilities = policy.log_probabilities.detach().cpu().clone()
            step_policy_weights = torch.ones(len(self._environments))
            if self._combat_selector is not None:
                for environment_index, (environment, observation) in enumerate(
                    zip(self._environments, observations, strict=True)
                ):
                    if observation.phase is not Phase.COMBAT:
                        continue
                    delegated = self._combat_selector(environment)
                    try:
                        delegated_index = observation.legal_actions.index(delegated)
                    except ValueError as error:
                        raise ValueError("combat selector returned an illegal action") from error
                    selected_indices[environment_index] = delegated_index
                    selected_log_probabilities[environment_index] = torch.log_softmax(
                        policy.logits[environment_index], dim=-1
                    )[delegated_index].detach().cpu()
                    step_policy_weights[environment_index] = 0.0

            states.append(policy.state_features)
            actions.append(policy.action_features)
            masks.append(policy.action_mask)
            chosen.append(selected_indices)
            log_probabilities.append(selected_log_probabilities)
            values.append(policy.values.detach().cpu())
            starts.append(self._episode_starts.clone())
            policy_weights.append(step_policy_weights)
            self._hidden = policy.next_hidden.detach()
            step_rewards = torch.zeros(len(self._environments))
            step_dones = torch.zeros(len(self._environments), dtype=torch.bool)
            next_episode_starts = torch.zeros(len(self._environments), dtype=torch.bool)

            for environment_index, environment in enumerate(self._environments):
                observation = observations[environment_index]
                action = observation.legal_actions[int(selected_indices[environment_index])]
                next_observation, reward, terminated, truncated, info = environment.step(action)
                trace = self._traces[environment_index]
                trace = EpisodeTrace(
                    seed=trace.seed,
                    initial_observation_digest=trace.initial_observation_digest,
                    steps=(
                        *trace.steps,
                        _recovery_trace_step(
                            action,
                            next_observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ),
                    ),
                    backend=trace.backend,
                    metadata=trace.metadata,
                )
                self._traces[environment_index] = trace
                self._episode_lengths[environment_index] += 1
                self._episode_returns[environment_index] += reward
                step_rewards[environment_index] = reward
                done = terminated or truncated
                step_dones[environment_index] = done
                self._observations[environment_index] = next_observation
                if not done:
                    continue
                completed.append(
                    RunEpisode(
                        environment_index=environment_index,
                        seed=self._active_seeds[environment_index],
                        length=self._episode_lengths[environment_index],
                        environment_return=self._episode_returns[environment_index],
                        final_hp=next_observation.player.hp,
                        final_floor=next_observation.floor,
                        final_act=next_observation.act,
                        won=reward > 0,
                    )
                )
                completed_traces.append(trace)
                seed = self._next_seed()
                reset_observation, reset_info = environment.reset(seed=seed)
                self._observations[environment_index] = reset_observation
                self._active_seeds[environment_index] = seed
                self._episode_lengths[environment_index] = 0
                self._episode_returns[environment_index] = 0.0
                self._hidden[environment_index].zero_()
                next_episode_starts[environment_index] = True
                self._traces[environment_index] = EpisodeTrace(
                    seed=int(reset_info.get("seed", seed)),
                    initial_observation_digest=observation_digest(reset_observation),
                    steps=(),
                    backend=str(reset_info.get("backend", "unknown")),
                    metadata=dict(reset_info),
                )

            rewards.append(step_rewards)
            dones.append(step_dones)
            self._episode_starts = next_episode_starts

        maximum_actions = max(tensor.shape[1] for tensor in actions)
        padded_actions = [
            functional.pad(tensor, (0, 0, 0, maximum_actions - tensor.shape[1]))
            for tensor in actions
        ]
        padded_masks = [
            functional.pad(tensor, (0, maximum_actions - tensor.shape[1]), value=False)
            for tensor in masks
        ]
        bootstrap_values = self.trainer.bootstrap_values(
            tuple(self._observations),
            self._hidden,
        )
        return RecurrentRolloutBatch(
            states=torch.stack(states),
            actions=torch.stack(padded_actions),
            action_masks=torch.stack(padded_masks),
            chosen_actions=torch.stack(chosen),
            old_log_probabilities=torch.stack(log_probabilities),
            old_values=torch.stack(values),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            episode_starts=torch.stack(starts),
            policy_weights=torch.stack(policy_weights),
            initial_hidden=initial_hidden,
            bootstrap_values=bootstrap_values,
            completed_episodes=tuple(completed),
            completed_traces=tuple(completed_traces),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "seeds": self._seeds,
            "next_seed_index": self._next_seed_index,
            "active_seeds": tuple(self._active_seeds),
            "episode_lengths": tuple(self._episode_lengths),
            "episode_returns": tuple(self._episode_returns),
            "episode_starts": self._episode_starts,
            "hidden": self._hidden.detach().cpu(),
            "traces": [trace.to_dict() for trace in self._traces],
        }

    @classmethod
    def from_state_dict(
        cls,
        environment_factory: Callable[[], StsEnv],
        trainer: RecurrentPPOTrainer,
        state: dict[str, Any],
        combat_selector: Callable[[StsEnv], Action] | None = None,
    ) -> RecurrentRolloutCollector:
        traces = [EpisodeTrace.from_dict(payload) for payload in state["traces"]]
        collector = object.__new__(cls)
        collector.trainer = trainer
        collector._environments = [environment_factory() for _ in traces]
        collector._seeds = tuple(int(seed) for seed in state["seeds"])
        collector._next_seed_index = int(state["next_seed_index"])
        collector._combat_selector = combat_selector
        collector._observations = [
            (
                environment.replay_recovery_trace(trace)
                if hasattr(environment, "replay_recovery_trace")
                else replay_trace(environment, trace)
            )
            for environment, trace in zip(collector._environments, traces, strict=True)
        ]
        collector._active_seeds = [int(seed) for seed in state["active_seeds"]]
        collector._episode_lengths = [int(value) for value in state["episode_lengths"]]
        collector._episode_returns = [float(value) for value in state["episode_returns"]]
        collector._episode_starts = state["episode_starts"].detach().cpu().clone().bool()
        collector._hidden = state["hidden"].to(trainer.device)
        collector._traces = traces
        return collector

    def _next_seed(self) -> int:
        if self._next_seed_index >= len(self._seeds):
            raise RuntimeError("collector exhausted its training seed stream")
        seed = self._seeds[self._next_seed_index]
        self._next_seed_index += 1
        return seed


class MultiprocessRecurrentRolloutCollector:
    def __init__(
        self,
        environment_pool: Any,
        trainer: RecurrentPPOTrainer,
        seeds: tuple[int, ...],
        combat_selector: Callable[[Observation], Action] | None = None,
    ):
        if len(seeds) < environment_pool.num_environments:
            raise ValueError("collector seed stream cannot initialize every worker")
        self.pool = environment_pool
        self.trainer = trainer
        self._combat_selector = combat_selector
        self._seeds = seeds
        self._next_seed_index = 0
        initial_seeds = tuple(self._next_seed() for _ in range(self.pool.num_environments))
        reset_results = self.pool.reset(initial_seeds)
        self._observations = [result[0] for result in reset_results]
        self._traces = [
            EpisodeTrace(
                seed=int(result[1].get("seed", seed)),
                initial_observation_digest=observation_digest(result[0]),
                steps=(),
                backend=str(result[1].get("backend", "unknown")),
                metadata=dict(result[1]),
            )
            for seed, result in zip(initial_seeds, reset_results, strict=True)
        ]
        self._active_seeds = list(initial_seeds)
        self._episode_lengths = [0] * self.pool.num_environments
        self._episode_returns = [0.0] * self.pool.num_environments
        self._episode_starts = torch.ones(self.pool.num_environments, dtype=torch.bool)
        self._hidden = trainer.initial_hidden(self.pool.num_environments)

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    def collect(self, steps_per_environment: int) -> RecurrentRolloutBatch:
        if steps_per_environment <= 0:
            raise ValueError("steps_per_environment must be positive")
        initial_hidden = self._hidden.detach().cpu().clone()
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        chosen: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        starts: list[torch.Tensor] = []
        policy_weights: list[torch.Tensor] = []
        completed: list[RunEpisode] = []
        completed_traces: list[EpisodeTrace] = []

        for rollout_step in range(steps_per_environment):
            observations = tuple(self._observations)
            invalid_observations = [
                environment_index
                for environment_index, observation in enumerate(observations)
                if observation.phase is Phase.TERMINAL or not observation.legal_actions
            ]
            if invalid_observations:
                details = []
                for environment_index in invalid_observations:
                    trace = self._traces[environment_index]
                    last = trace.steps[-1] if trace.steps else None
                    details.append(
                        {
                            "environment_index": environment_index,
                            "active_seed": self._active_seeds[environment_index],
                            "trace_seed": trace.seed,
                            "trace_steps": len(trace.steps),
                            "phase": observations[environment_index].phase.value,
                            "floor": observations[environment_index].floor,
                            "last_action": None if last is None else last.action.to_dict(),
                            "last_terminated": None if last is None else last.terminated,
                            "last_truncated": None if last is None else last.truncated,
                            "last_info": None if last is None else last.info,
                        }
                    )
                raise RuntimeError(
                    f"collector has terminal observations before rollout step {rollout_step}: "
                    f"{details}"
                )
            policy = self.trainer.sample_actions(observations, self._hidden)
            selected_indices = policy.action_indices.detach().cpu().clone()
            selected_log_probabilities = policy.log_probabilities.detach().cpu().clone()
            selected_actions = tuple(
                observation.legal_actions[int(selected_indices[index])]
                for index, observation in enumerate(observations)
            )
            step_policy_weights = torch.ones(self.pool.num_environments)
            if self._combat_selector is not None:
                delegated_actions = list(selected_actions)
                for environment_index, observation in enumerate(observations):
                    if observation.phase is not Phase.COMBAT:
                        continue
                    delegated = self._combat_selector(observation)
                    try:
                        delegated_index = observation.legal_actions.index(delegated)
                    except ValueError as error:
                        raise ValueError("combat selector returned an illegal action") from error
                    selected_indices[environment_index] = delegated_index
                    delegated_actions[environment_index] = delegated
                    selected_log_probabilities[environment_index] = torch.log_softmax(
                        policy.logits[environment_index], dim=-1
                    )[delegated_index].detach().cpu()
                    step_policy_weights[environment_index] = 0.0
                selected_actions = tuple(delegated_actions)
            results = self.pool.step(selected_actions)
            states.append(policy.state_features)
            actions.append(policy.action_features)
            masks.append(policy.action_mask)
            chosen.append(selected_indices)
            log_probabilities.append(selected_log_probabilities)
            values.append(policy.values.detach().cpu())
            starts.append(self._episode_starts.clone())
            policy_weights.append(step_policy_weights)
            self._hidden = policy.next_hidden.detach()
            step_rewards = torch.zeros(self.pool.num_environments)
            step_dones = torch.zeros(self.pool.num_environments, dtype=torch.bool)
            next_episode_starts = torch.zeros(self.pool.num_environments, dtype=torch.bool)

            for environment_index, result in enumerate(results):
                observation, reward, terminated, truncated, info = result
                trace = self._traces[environment_index]
                trace = EpisodeTrace(
                    seed=trace.seed,
                    initial_observation_digest=trace.initial_observation_digest,
                    steps=(
                        *trace.steps,
                        _recovery_trace_step(
                            selected_actions[environment_index],
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ),
                    ),
                    backend=trace.backend,
                    metadata=trace.metadata,
                )
                self._traces[environment_index] = trace
                self._episode_lengths[environment_index] += 1
                self._episode_returns[environment_index] += reward
                step_rewards[environment_index] = reward
                done = terminated or truncated
                step_dones[environment_index] = done
                self._observations[environment_index] = observation
                if not done:
                    continue
                completed.append(
                    RunEpisode(
                        environment_index=environment_index,
                        seed=self._active_seeds[environment_index],
                        length=self._episode_lengths[environment_index],
                        environment_return=self._episode_returns[environment_index],
                        final_hp=observation.player.hp,
                        final_floor=observation.floor,
                        final_act=observation.act,
                        won=reward > 0,
                    )
                )
                completed_traces.append(trace)
                seed = self._next_seed()
                reset_observation, reset_info = self.pool.reset_at(environment_index, seed)
                self._observations[environment_index] = reset_observation
                self._active_seeds[environment_index] = seed
                self._episode_lengths[environment_index] = 0
                self._episode_returns[environment_index] = 0.0
                self._hidden[environment_index].zero_()
                next_episode_starts[environment_index] = True
                self._traces[environment_index] = EpisodeTrace(
                    seed=int(reset_info.get("seed", seed)),
                    initial_observation_digest=observation_digest(reset_observation),
                    steps=(),
                    backend=str(reset_info.get("backend", "unknown")),
                    metadata=dict(reset_info),
                )

            rewards.append(step_rewards)
            dones.append(step_dones)
            self._episode_starts = next_episode_starts

        maximum_actions = max(tensor.shape[1] for tensor in actions)
        padded_actions = [
            functional.pad(tensor, (0, 0, 0, maximum_actions - tensor.shape[1]))
            for tensor in actions
        ]
        padded_masks = [
            functional.pad(tensor, (0, maximum_actions - tensor.shape[1]), value=False)
            for tensor in masks
        ]
        return RecurrentRolloutBatch(
            states=torch.stack(states),
            actions=torch.stack(padded_actions),
            action_masks=torch.stack(padded_masks),
            chosen_actions=torch.stack(chosen),
            old_log_probabilities=torch.stack(log_probabilities),
            old_values=torch.stack(values),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            episode_starts=torch.stack(starts),
            policy_weights=torch.stack(policy_weights),
            initial_hidden=initial_hidden,
            bootstrap_values=self.trainer.bootstrap_values(
                tuple(self._observations),
                self._hidden,
            ),
            completed_episodes=tuple(completed),
            completed_traces=tuple(completed_traces),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "seeds": self._seeds,
            "next_seed_index": self._next_seed_index,
            "active_seeds": tuple(self._active_seeds),
            "episode_lengths": tuple(self._episode_lengths),
            "episode_returns": tuple(self._episode_returns),
            "episode_starts": self._episode_starts,
            "hidden": self._hidden.detach().cpu(),
            "traces": [trace.to_dict() for trace in self._traces],
        }

    @classmethod
    def from_state_dict(
        cls,
        environment_pool: Any,
        trainer: RecurrentPPOTrainer,
        state: dict[str, Any],
        combat_selector: Callable[[Observation], Action] | None = None,
    ) -> MultiprocessRecurrentRolloutCollector:
        traces = tuple(EpisodeTrace.from_dict(payload) for payload in state["traces"])
        if len(traces) != environment_pool.num_environments:
            raise ValueError("checkpoint worker count differs from the environment pool")
        collector = object.__new__(cls)
        collector.pool = environment_pool
        collector.trainer = trainer
        collector._combat_selector = combat_selector
        collector._seeds = tuple(int(seed) for seed in state["seeds"])
        collector._next_seed_index = int(state["next_seed_index"])
        collector._observations = list(environment_pool.replay(traces))
        collector._active_seeds = [int(seed) for seed in state["active_seeds"]]
        collector._episode_lengths = [int(value) for value in state["episode_lengths"]]
        collector._episode_returns = [float(value) for value in state["episode_returns"]]
        collector._episode_starts = state["episode_starts"].detach().cpu().clone().bool()
        collector._hidden = state["hidden"].to(trainer.device)
        collector._traces = list(traces)
        return collector

    def close(self) -> None:
        self.pool.close()

    def _next_seed(self) -> int:
        if self._next_seed_index >= len(self._seeds):
            raise RuntimeError("collector exhausted its training seed stream")
        seed = self._seeds[self._next_seed_index]
        self._next_seed_index += 1
        return seed


class HierarchicalRecurrentPolicy:
    def __init__(
        self,
        trainer: RecurrentPPOTrainer,
        combat_selector: Callable[[StsEnv], Action] | None = None,
        *,
        deterministic: bool = True,
    ):
        self.trainer = trainer
        self.combat_selector = combat_selector
        self.deterministic = deterministic
        self._hidden = trainer.initial_hidden(1)

    def reset(self) -> None:
        self._hidden = self.trainer.initial_hidden(1)

    def select(self, environment: StsEnv) -> Action:
        observation = environment.observation
        if not observation.legal_actions:
            raise ValueError("cannot act in a terminal observation")
        policy = self.trainer.sample_actions(
            (observation,),
            self._hidden,
            deterministic=self.deterministic,
        )
        self._hidden = policy.next_hidden.detach()
        if self.combat_selector is not None and observation.phase is Phase.COMBAT:
            action = self.combat_selector(environment)
            if action not in observation.legal_actions:
                raise ValueError("combat selector returned an illegal action")
            return action
        return observation.legal_actions[int(policy.action_indices[0])]
