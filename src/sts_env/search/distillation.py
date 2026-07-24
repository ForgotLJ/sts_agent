from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

from sts_env.search.mcts import BeliefSearchResult, default_leaf_value
from sts_env.training.candidate_q import CandidateQNetwork
from sts_env.training.encoding import EncoderConfig, ObjectFeatureEncoder
from sts_env.types import Action, Observation, Phase


@dataclass(frozen=True, slots=True)
class SearchTarget:
    observation: Observation
    policy: tuple[float, ...]
    value: float

    def __post_init__(self) -> None:
        if len(self.policy) != len(self.observation.legal_actions):
            raise ValueError("search target policy must align with legal actions")
        if any(not math.isfinite(probability) or probability < 0 for probability in self.policy):
            raise ValueError("search target probabilities must be finite and non-negative")
        if self.policy and not math.isclose(sum(self.policy), 1.0, abs_tol=1e-6):
            raise ValueError("search target policy must sum to one")
        if not math.isfinite(self.value):
            raise ValueError("search target value must be finite")

    @classmethod
    def from_search_result(
        cls,
        observation: Observation,
        result: BeliefSearchResult,
        *,
        selected_action_only: bool = False,
    ) -> SearchTarget:
        probabilities = result.policy
        if selected_action_only:
            probabilities = {
                action: float(action == result.selected_action)
                for action in observation.legal_actions
            }
        return cls(
            observation=observation,
            policy=tuple(probabilities[action] for action in observation.legal_actions),
            value=result.root_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "policy": list(self.policy),
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SearchTarget:
        return cls(
            observation=Observation.from_dict(dict(payload["observation"])),
            policy=tuple(float(value) for value in payload["policy"]),
            value=float(payload["value"]),
        )


class SearchTargetBuffer:
    def __init__(self, capacity: int = 100_000, seed: int = 0):
        if capacity <= 0:
            raise ValueError("target buffer capacity must be positive")
        self.capacity = capacity
        self._targets: list[SearchTarget] = []
        self._next_index = 0
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._targets)

    def add(self, target: SearchTarget) -> None:
        if len(self._targets) < self.capacity:
            self._targets.append(target)
        else:
            self._targets[self._next_index] = target
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, count: int) -> tuple[SearchTarget, ...]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        if count > len(self._targets):
            raise ValueError("sample count exceeds target buffer size")
        return tuple(self._random.sample(self._targets, count))

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            header = {"capacity": self.capacity, "next_index": self._next_index}
            stream.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
            for target in self._targets:
                stream.write(json.dumps(target.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path, seed: int = 0) -> SearchTargetBuffer:
        with Path(path).open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if not records:
            raise ValueError("target buffer JSONL is empty")
        buffer = cls(capacity=int(records[0]["capacity"]), seed=seed)
        for payload in records[1:]:
            buffer._targets.append(SearchTarget.from_dict(payload))
        buffer._next_index = int(records[0]["next_index"])
        if buffer._next_index < 0 or buffer._next_index >= buffer.capacity:
            raise ValueError("target buffer next index is invalid")
        return buffer


@dataclass(frozen=True, slots=True)
class PolicyValueConfig:
    policy_hidden_sizes: tuple[int, ...] = (256, 128)
    value_hidden_sizes: tuple[int, ...] = (256, 128)
    learning_rate: float = 3e-4
    batch_size: int = 64
    gradient_clip_norm: float = 5.0
    value_loss_weight: float = 1.0
    entropy_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.policy_hidden_sizes or min(self.policy_hidden_sizes) <= 0:
            raise ValueError("policy hidden sizes must be positive")
        if not self.value_hidden_sizes or min(self.value_hidden_sizes) <= 0:
            raise ValueError("value hidden sizes must be positive")
        if self.learning_rate <= 0 or self.batch_size <= 0:
            raise ValueError("learning rate and batch size must be positive")
        if self.gradient_clip_norm <= 0 or self.value_loss_weight < 0:
            raise ValueError("gradient clip must be positive and value weight non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_hidden_sizes"] = list(self.policy_hidden_sizes)
        payload["value_hidden_sizes"] = list(self.value_hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyValueConfig:
        values = dict(payload)
        values["policy_hidden_sizes"] = tuple(int(value) for value in values["policy_hidden_sizes"])
        values["value_hidden_sizes"] = tuple(int(value) for value in values["value_hidden_sizes"])
        return cls(**values)


class PolicyValueTrainer:
    def __init__(
        self,
        config: PolicyValueConfig | None = None,
        encoder: ObjectFeatureEncoder | None = None,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.config = config or PolicyValueConfig()
        self.encoder = encoder or ObjectFeatureEncoder()
        self.device = torch.device(device)
        self.seed = seed
        self._random = random.Random(seed)
        torch.manual_seed(seed)
        self.policy_network = CandidateQNetwork(
            self.encoder.dimension,
            self.config.policy_hidden_sizes,
        ).to(self.device)
        self.value_network = CandidateQNetwork(
            self.encoder.dimension,
            self.config.value_hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=self.config.learning_rate,
        )
        self.gradient_steps = 0

    def policy_logits(self, observation: Observation) -> torch.Tensor:
        features = self.encoder.encode_candidates(observation).to(self.device)
        if features.shape[0] == 0:
            return torch.empty(0, device=self.device)
        logits = self.policy_network(features)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("policy network produced non-finite logits")
        return logits

    def policy(self, observation: Observation, temperature: float = 1.0) -> tuple[float, ...]:
        if temperature <= 0:
            raise ValueError("policy temperature must be positive")
        with torch.no_grad():
            logits = self.policy_logits(observation) / temperature
            if logits.numel() == 0:
                return ()
            probabilities = torch.softmax(logits, dim=0)
        return tuple(float(value) for value in probabilities.cpu())

    def value(self, observation: Observation) -> float:
        if observation.phase is Phase.TERMINAL or not observation.legal_actions:
            return default_leaf_value(observation)
        features = self.encoder.encode_candidates(observation).mean(dim=0).to(self.device)
        with torch.no_grad():
            value = self.value_network(features.unsqueeze(0))[0]
        if not torch.isfinite(value):
            raise FloatingPointError("value network produced a non-finite value")
        return float(value.item())

    def greedy_action(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            raise ValueError("cannot select an action in a terminal observation")
        logits = self.policy_logits(observation)
        return observation.legal_actions[int(torch.argmax(logits).item())]

    def train_batch(self, targets: tuple[SearchTarget, ...]) -> dict[str, float]:
        if not targets:
            raise ValueError("distillation batch must not be empty")
        policy_losses: list[torch.Tensor] = []
        value_predictions: list[torch.Tensor] = []
        value_targets: list[float] = []
        entropies: list[torch.Tensor] = []
        for target in targets:
            candidate_features = self.encoder.encode_candidates(target.observation).to(self.device)
            if candidate_features.shape[0] == 0:
                raise ValueError("distillation target must contain legal actions")
            logits = self.policy_network(candidate_features)
            log_probabilities = torch.log_softmax(logits, dim=0)
            probabilities = torch.softmax(logits, dim=0)
            target_policy = torch.tensor(target.policy, dtype=torch.float32, device=self.device)
            policy_losses.append(-(target_policy * log_probabilities).sum())
            entropies.append(-(probabilities * log_probabilities).sum())
            state_features = candidate_features.mean(dim=0, keepdim=True)
            value_predictions.append(self.value_network(state_features)[0])
            value_targets.append(target.value)

        policy_loss = torch.stack(policy_losses).mean()
        predicted_values = torch.stack(value_predictions)
        expected_values = torch.tensor(value_targets, dtype=torch.float32, device=self.device)
        value_loss = functional.smooth_l1_loss(predicted_values, expected_values)
        entropy = torch.stack(entropies).mean()
        total_loss = (
            policy_loss
            + self.config.value_loss_weight * value_loss
            - self.config.entropy_weight * entropy
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("policy-value loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            self.config.gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("policy-value gradient norm is non-finite")
        self.optimizer.step()
        self.gradient_steps += 1
        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "gradient_norm": float(gradient_norm.item()),
        }

    def train_from_buffer(
        self,
        buffer: SearchTargetBuffer,
        updates: int = 1,
    ) -> dict[str, float]:
        if updates <= 0:
            raise ValueError("updates must be positive")
        if len(buffer) < self.config.batch_size:
            raise ValueError("target buffer is smaller than the configured batch size")
        metrics: list[dict[str, float]] = []
        for _ in range(updates):
            metrics.append(self.train_batch(buffer.sample(self.config.batch_size)))
        return {
            key: sum(metric[key] for metric in metrics) / len(metrics)
            for key in metrics[0]
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "encoder_config": self.encoder.config.to_dict(),
            "seed": self.seed,
            "gradient_steps": self.gradient_steps,
            "policy_network": self.policy_network.state_dict(),
            "value_network": self.value_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "python_rng_state": self._random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
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
    ) -> PolicyValueTrainer:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        trainer = cls(
            config=PolicyValueConfig.from_dict(payload["config"]),
            encoder=ObjectFeatureEncoder(EncoderConfig(**payload["encoder_config"])),
            seed=int(payload["seed"]),
            device=device,
        )
        trainer.policy_network.load_state_dict(payload["policy_network"])
        trainer.value_network.load_state_dict(payload["value_network"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.gradient_steps = int(payload["gradient_steps"])
        trainer._random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        return trainer
