from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

from sts_env.training.encoding import EncoderConfig, ObjectFeatureEncoder
from sts_env.training.replay import ReplayBuffer, ReplayTransition
from sts_env.types import Action, Observation


@dataclass(frozen=True, slots=True)
class CandidateQConfig:
    hidden_sizes: tuple[int, ...] = (256, 128)
    learning_rate: float = 3e-4
    gamma: float = 0.97
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 512
    target_update_interval: int = 250
    gradient_clip_norm: float = 5.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 15_000
    damage_dealt_scale: float = 0.03
    damage_taken_scale: float = -0.04
    step_penalty: float = -0.002

    def __post_init__(self) -> None:
        if not self.hidden_sizes or min(self.hidden_sizes) <= 0:
            raise ValueError("hidden_sizes must contain positive widths")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must cover a positive batch")
        if self.warmup_steps < self.batch_size:
            raise ValueError("warmup_steps must be at least batch_size")
        if self.target_update_interval <= 0 or self.epsilon_decay_steps <= 0:
            raise ValueError("update and decay intervals must be positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateQConfig:
        values = dict(payload)
        values["hidden_sizes"] = tuple(int(value) for value in values["hidden_sizes"])
        return cls(**values)


class CandidateQNetwork(nn.Module):
    def __init__(self, input_dimension: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive")
        layers: list[nn.Module] = []
        previous = input_dimension
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class CandidateQTrainer:
    def __init__(
        self,
        config: CandidateQConfig | None = None,
        encoder: ObjectFeatureEncoder | None = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.config = config or CandidateQConfig()
        self.encoder = encoder or ObjectFeatureEncoder()
        self.device = torch.device(device)
        self.seed = seed
        self._random = random.Random(seed)
        torch.manual_seed(seed)
        self.network = CandidateQNetwork(
            self.encoder.dimension,
            self.config.hidden_sizes,
        ).to(self.device)
        self.target_network = CandidateQNetwork(
            self.encoder.dimension,
            self.config.hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
        )
        self.replay = ReplayBuffer(self.config.replay_capacity, seed=seed ^ 0xC0FFEE)
        self.environment_steps = 0
        self.gradient_steps = 0
        self.last_loss: float | None = None
        self.last_gradient_norm: float | None = None

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def q_values(self, observation: Observation) -> torch.Tensor:
        features = self.encoder.encode_candidates(observation).to(self.device)
        if features.shape[0] == 0:
            return torch.empty(0, device=self.device)
        with torch.no_grad():
            values = self.network(features)
        if not torch.isfinite(values).all():
            raise FloatingPointError("candidate Q network produced non-finite values")
        return values

    def greedy_action(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            raise ValueError("cannot act in a terminal observation")
        values = self.q_values(observation)
        action_index = int(torch.argmax(values).item())
        return observation.legal_actions[action_index]

    def select_action(self, observation: Observation, explore: bool = True) -> Action:
        if not observation.legal_actions:
            raise ValueError("cannot act in a terminal observation")
        if explore and self._random.random() < self.epsilon:
            return self._random.choice(observation.legal_actions)
        return self.greedy_action(observation)

    def observe(self, transition: ReplayTransition) -> float:
        shaped_reward = self.training_reward(transition)
        self.replay.add(replace(transition, reward=shaped_reward))
        self.environment_steps += 1
        return shaped_reward

    def training_reward(self, transition: ReplayTransition) -> float:
        damage_dealt = float(transition.info.get("damage_dealt", 0.0))
        damage_taken = float(transition.info.get("damage_taken", 0.0))
        reward = (
            transition.reward
            + self.config.damage_dealt_scale * damage_dealt
            + self.config.damage_taken_scale * damage_taken
            + self.config.step_penalty
        )
        if not math.isfinite(reward):
            raise FloatingPointError("training reward is non-finite")
        return reward

    def train_step(self, updates: int = 1) -> dict[str, float] | None:
        if updates <= 0:
            raise ValueError("updates must be positive")
        if len(self.replay) < self.config.warmup_steps:
            return None
        losses: list[float] = []
        gradient_norms: list[float] = []
        for _ in range(updates):
            batch = self.replay.sample(self.config.batch_size)
            features = torch.stack(
                [self.encoder.encode(item.observation, item.action) for item in batch]
            ).to(self.device)
            predictions = self.network(features)
            targets = self._targets(batch)
            loss = functional.smooth_l1_loss(predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("candidate Q loss is non-finite")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.gradient_clip_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("candidate Q gradient norm is non-finite")
            self.optimizer.step()
            self.gradient_steps += 1
            if self.gradient_steps % self.config.target_update_interval == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            losses.append(float(loss.item()))
            gradient_norms.append(float(gradient_norm.item()))
        self.last_loss = sum(losses) / len(losses)
        self.last_gradient_norm = sum(gradient_norms) / len(gradient_norms)
        return {
            "loss": self.last_loss,
            "gradient_norm": self.last_gradient_norm,
            "epsilon": self.epsilon,
            "replay_size": float(len(self.replay)),
        }

    def _targets(self, batch: tuple[ReplayTransition, ...]) -> torch.Tensor:
        values: list[float] = []
        with torch.no_grad():
            for item in batch:
                terminal = item.terminated or item.truncated
                if terminal or not item.next_observation.legal_actions:
                    bootstrap = 0.0
                else:
                    next_features = self.encoder.encode_candidates(
                        item.next_observation
                    ).to(self.device)
                    online_values = self.network(next_features)
                    best_index = int(torch.argmax(online_values).item())
                    bootstrap = float(self.target_network(next_features)[best_index].item())
                values.append(item.reward + self.config.gamma * bootstrap)
        targets = torch.tensor(values, dtype=torch.float32, device=self.device)
        if not torch.isfinite(targets).all():
            raise FloatingPointError("candidate Q targets are non-finite")
        return targets

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "config": self.config.to_dict(),
            "encoder_config": self.encoder.config.to_dict(),
            "seed": self.seed,
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "random_state": self._random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "metadata": metadata or {},
        }
        torch.save(payload, destination)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> tuple[CandidateQTrainer, dict[str, Any]]:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError("unsupported candidate Q checkpoint version")
        trainer = cls(
            config=CandidateQConfig.from_dict(payload["config"]),
            encoder=ObjectFeatureEncoder(EncoderConfig(**payload["encoder_config"])),
            seed=int(payload["seed"]),
            device=device,
        )
        trainer.network.load_state_dict(payload["network"])
        trainer.target_network.load_state_dict(payload["target_network"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.replay = ReplayBuffer.from_state_dict(payload["replay"])
        trainer.environment_steps = int(payload["environment_steps"])
        trainer.gradient_steps = int(payload["gradient_steps"])
        trainer._random.setstate(payload["random_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        return trainer, dict(payload.get("metadata") or {})
