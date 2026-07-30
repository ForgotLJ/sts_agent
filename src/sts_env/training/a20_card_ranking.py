from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from sts_env.training.a20_coldstart import A20ColdStartConfig, _base_card, _bucket, encode_prefix


SKIP_CARD = "SKIP"


@dataclass(frozen=True, slots=True)
class A20CardRankingConfig:
    state_config: A20ColdStartConfig = A20ColdStartConfig()
    candidate_buckets: int = 256
    hidden_dimension: int = 128

    def __post_init__(self) -> None:
        if self.candidate_buckets <= 0 or self.hidden_dimension <= 0:
            raise ValueError("A20 card-ranking dimensions must be positive")


@dataclass(frozen=True, slots=True)
class CardChoiceExample:
    run_id: str
    floor: int
    state_features: tuple[float, ...]
    candidates: tuple[str, ...]
    selected_index: int

    def __post_init__(self) -> None:
        if not self.run_id or self.floor <= 0 or len(self.candidates) < 2:
            raise ValueError("invalid card-choice example")
        if self.selected_index < 0 or self.selected_index >= len(self.candidates):
            raise ValueError("card-choice target is outside its candidate set")


class A20CardRanker(nn.Module):
    def __init__(self, state_dimension: int, config: A20CardRankingConfig | None = None):
        super().__init__()
        self.config = config or A20CardRankingConfig()
        self.state_body = nn.Sequential(
            nn.Linear(state_dimension, self.config.hidden_dimension),
            nn.LayerNorm(self.config.hidden_dimension),
            nn.GELU(),
            nn.Linear(self.config.hidden_dimension, self.config.hidden_dimension),
            nn.GELU(),
        )
        self.candidate_embedding = nn.Embedding(
            self.config.candidate_buckets,
            self.config.hidden_dimension,
        )
        self.candidate_bias = nn.Embedding(self.config.candidate_buckets, 1)

    def forward(self, state_features: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        context = self.state_body(state_features).unsqueeze(1)
        candidates = self.candidate_embedding(candidate_ids)
        return (context * candidates).sum(dim=-1) + self.candidate_bias(candidate_ids).squeeze(-1)


def _choice_candidates(choice: dict[str, Any]) -> tuple[tuple[str, ...], int] | None:
    selected = choice.get("picked")
    if selected is None or str(selected) == "":
        return None
    selected = SKIP_CARD if selected == SKIP_CARD else _base_card(selected)
    candidates: list[str] = []
    for candidate in choice.get("not_picked") or []:
        candidate = _base_card(candidate)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    if selected != SKIP_CARD and selected not in candidates:
        candidates.append(selected)
    if SKIP_CARD not in candidates:
        candidates.append(SKIP_CARD)
    if selected not in candidates:
        return None
    return tuple(candidates), candidates.index(selected)


def build_card_choice_examples(
    record: dict[str, Any],
    config: A20CardRankingConfig | None = None,
) -> tuple[CardChoiceExample, ...]:
    if not record.get("run_id"):
        raise ValueError("card-ranking record requires run_id")
    config = config or A20CardRankingConfig()
    result: list[CardChoiceExample] = []
    seen: set[tuple[int, tuple[str, ...], int]] = set()
    for choice in record.get("card_choices") or []:
        if not isinstance(choice, dict):
            continue
        try:
            floor = int(float(choice.get("floor")))
        except (TypeError, ValueError):
            continue
        if floor <= 0 or floor > int(record.get("floor_reached") or 0):
            continue
        candidate_spec = _choice_candidates(choice)
        if candidate_spec is None:
            continue
        candidates, selected_index = candidate_spec
        identity = (floor, candidates, selected_index)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(
            CardChoiceExample(
                run_id=str(record["run_id"]),
                floor=floor,
                state_features=encode_prefix(
                    record,
                    floor,
                    config.state_config,
                    decision_floor=floor - 1,
                ),
                candidates=candidates,
                selected_index=selected_index,
            )
        )
    return tuple(result)


def _split_examples(examples: list[CardChoiceExample]) -> dict[str, list[CardChoiceExample]]:
    splits = {"train": [], "validation": [], "test": []}
    for example in examples:
        value = int(hashlib.blake2b(example.run_id.encode("utf-8"), digest_size=2).hexdigest(), 16) % 100
        split = "train" if value < 80 else "validation" if value < 90 else "test"
        splits[split].append(example)
    return splits


def _batch_tensors(
    examples: list[CardChoiceExample],
    indices: torch.Tensor,
    config: A20CardRankingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [examples[int(index)] for index in indices.tolist()]
    maximum_candidates = max(len(example.candidates) for example in selected)
    state_features = torch.tensor(
        [example.state_features for example in selected],
        dtype=torch.float32,
        device=device,
    )
    candidate_ids = torch.zeros(
        (len(selected), maximum_candidates),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros((len(selected), maximum_candidates), dtype=torch.bool, device=device)
    targets = torch.tensor([example.selected_index for example in selected], dtype=torch.long, device=device)
    for row, example in enumerate(selected):
        values = [_bucket(candidate, config.candidate_buckets) for candidate in example.candidates]
        candidate_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, : len(values)] = True
    return state_features, candidate_ids, mask, targets


def evaluate_card_ranker(
    model: A20CardRanker,
    examples: list[CardChoiceExample],
    device: torch.device,
) -> dict[str, float]:
    if not examples:
        raise ValueError("cannot evaluate an empty card-choice split")
    model.eval()
    correct = 0
    negative_log_likelihood = 0.0
    selected_probability = 0.0
    chance = 0.0
    with torch.no_grad():
        for start in range(0, len(examples), 512):
            indices = torch.arange(start, min(start + 512, len(examples)))
            states, candidate_ids, mask, targets = _batch_tensors(examples, indices, model.config, device)
            scores = model(states, candidate_ids).masked_fill(~mask, float("-inf"))
            probabilities = torch.softmax(scores, dim=1)
            correct += int((scores.argmax(dim=1) == targets).sum().item())
            selected_probability += float(probabilities.gather(1, targets.unsqueeze(1)).sum().item())
            negative_log_likelihood += float(functional.cross_entropy(scores, targets, reduction="sum").item())
            chance += sum(1.0 / len(example.candidates) for example in examples[start : start + 512])
    total = len(examples)
    return {
        "examples": float(total),
        "top1_accuracy": correct / total,
        "mean_selected_probability": selected_probability / total,
        "mean_uniform_chance": chance / total,
        "negative_log_likelihood": negative_log_likelihood / total,
    }


def train_card_ranker(
    examples: list[CardChoiceExample],
    *,
    config: A20CardRankingConfig | None = None,
    epochs: int = 8,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    seed: int = 17,
    device: str = "cpu",
) -> tuple[A20CardRanker, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train a card ranker without examples")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("invalid card-ranker training parameters")
    config = config or A20CardRankingConfig()
    torch.manual_seed(seed)
    target_device = torch.device(device)
    splits = _split_examples(examples)
    if any(not values for values in splits.values()):
        raise ValueError("card-ranker split is empty")
    model = A20CardRanker(len(examples[0].state_features), config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    metrics: dict[str, Any] = {"splits": {name: len(values) for name, values in splits.items()}, "epochs": []}
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(splits["train"]), generator=generator)
        losses = []
        for start in range(0, len(splits["train"]), batch_size):
            indices = permutation[start : start + batch_size]
            states, candidate_ids, mask, targets = _batch_tensors(
                splits["train"], indices, config, target_device
            )
            scores = model(states, candidate_ids).masked_fill(~mask, float("-inf"))
            loss = functional.cross_entropy(scores, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(losses) / len(losses),
                **evaluate_card_ranker(model, splits["validation"], target_device),
            }
        )
    metrics["test"] = evaluate_card_ranker(model, splits["test"], target_device)
    return model, metrics
