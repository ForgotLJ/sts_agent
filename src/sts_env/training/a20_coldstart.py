from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional


CHARACTERS = ("DEFECT", "IRONCLAD", "THE_SILENT", "WATCHER")
PREFIX_FLOORS = (10, 20, 30, 40, 50)
ROOM_SYMBOLS = ("M", "E", "?", "R", "$", "T", "B", "N")


@dataclass(frozen=True, slots=True)
class A20ColdStartConfig:
    card_buckets: int = 96
    relic_buckets: int = 48
    event_buckets: int = 48
    item_buckets: int = 48

    @property
    def feature_dimension(self) -> int:
        return 17 + len(ROOM_SYMBOLS) + self.card_buckets + self.relic_buckets + self.event_buckets + self.item_buckets


@dataclass(frozen=True, slots=True)
class PrefixExample:
    run_id: str
    prefix_floor: int
    features: tuple[float, ...]
    heart_win: float
    normal_win: float
    final_floor: float


def _bucket(value: object, buckets: int) -> int:
    digest = hashlib.blake2b(str(value).lower().encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def _base_card(card: object) -> str:
    text = str(card)
    while text.endswith(tuple(f"+{level}" for level in range(1, 10))):
        text = text.rsplit("+", 1)[0]
    return text


def _prefix_items(items: object, floor: int, key: str) -> list[str]:
    if not isinstance(items, list):
        return []
    return [
        str(item.get(key))
        for item in items
        if isinstance(item, dict) and float(item.get("floor", 10**9)) <= floor
    ]


def _prefix_card_picks(record: dict[str, Any], floor: int) -> tuple[list[str], list[str]]:
    picked: list[str] = []
    offered: list[str] = []
    choices = record.get("card_choices") or []
    if not isinstance(choices, list):
        return picked, offered
    for choice in choices:
        if not isinstance(choice, dict) or float(choice.get("floor", 10**9)) > floor:
            continue
        selected = choice.get("picked")
        if selected and selected != "SKIP":
            picked.append(_base_card(selected))
        for card in choice.get("not_picked") or []:
            offered.append(_base_card(card))
        if selected and selected != "SKIP":
            offered.append(_base_card(selected))
    return picked, offered


def _room_counts(path: object, floor: int) -> list[float]:
    counts = [0.0] * len(ROOM_SYMBOLS)
    if not isinstance(path, list):
        return counts
    for symbol in path[:floor]:
        text = str(symbol).upper()
        if text in ROOM_SYMBOLS:
            counts[ROOM_SYMBOLS.index(text)] += 1.0
    total = max(1.0, sum(counts))
    return [count / total for count in counts]


def _histogram(values: Iterable[object], buckets: int) -> list[float]:
    result = [0.0] * buckets
    count = 0
    for value in values:
        if value is None or str(value) == "":
            continue
        result[_bucket(value, buckets)] += 1.0
        count += 1
    scale = max(1.0, float(count))
    return [value / scale for value in result]


def _prefix_scalar(values: object, floor: int, default: float = 0.0) -> float:
    if not isinstance(values, list) or not values:
        return default
    index = min(max(0, floor - 1), len(values) - 1)
    try:
        return float(values[index])
    except (TypeError, ValueError):
        return default


def encode_prefix(
    record: dict[str, Any],
    floor: int,
    config: A20ColdStartConfig | None = None,
    decision_floor: int | None = None,
) -> tuple[float, ...]:
    config = config or A20ColdStartConfig()
    if floor <= 0:
        raise ValueError("prefix floor must be positive")
    decision_floor = floor if decision_floor is None else decision_floor
    if decision_floor < 0 or decision_floor > floor:
        raise ValueError("decision floor must be between zero and the prefix floor")
    picked, offered = _prefix_card_picks(record, decision_floor)
    relics = _prefix_items(record.get("relics_obtained"), decision_floor, "key")
    potions = _prefix_items(record.get("potions_obtained"), decision_floor, "key")
    events = _prefix_items(record.get("event_choices"), decision_floor, "event_name")
    shops = [
        str(item)
        for item, item_floor in zip(
            record.get("items_purchased") or [], record.get("item_purchase_floors") or []
        )
        if float(item_floor) <= decision_floor
    ]
    max_hp = max(1.0, _prefix_scalar(record.get("max_hp_per_floor"), floor, 1.0))
    hp = _prefix_scalar(record.get("current_hp_per_floor"), floor, max_hp)
    gold = _prefix_scalar(record.get("gold_per_floor"), floor)
    damage = sum(
        float(item.get("damage", 0.0))
        for item in record.get("damage_taken") or []
        if isinstance(item, dict) and float(item.get("floor", 10**9)) <= floor
    )
    initial_deck = {
        "IRONCLAD": ("Strike_R", "Defend_R", "Bash"),
        "THE_SILENT": ("Strike_R", "Defend_R", "Survivor", "Neutralize"),
        "DEFECT": ("Strike_G", "Defend_G", "Zap", "Dualcast"),
        "WATCHER": ("Strike_P", "Defend_P", "Eruption", "Vigilance"),
    }.get(str(record.get("character")), ())
    deck_count = len(initial_deck) + len(picked)
    feature_values = [
        floor / 60.0,
        hp / max_hp,
        max_hp / 100.0,
        gold / 500.0,
        damage / 500.0,
        deck_count / 60.0,
        len(set(picked)) / 40.0,
        len(relics) / 20.0,
        len(potions) / 5.0,
        len(events) / 20.0,
        len(shops) / 20.0,
        len(offered) / 80.0,
        sum(1 for symbol in (record.get("path_per_floor") or [])[:floor] if symbol == "E") / 12.0,
        sum(1 for symbol in (record.get("path_per_floor") or [])[:floor] if symbol == "R") / 12.0,
        sum(1 for symbol in (record.get("path_per_floor") or [])[:floor] if symbol == "$") / 12.0,
        float(record.get("ascension_level") or 0) / 20.0,
        float(bool(record.get("is_ascension_mode"))),
    ]
    features = (
        feature_values
        + _room_counts(record.get("path_per_floor"), floor)
        + _histogram(initial_deck + tuple(picked) + tuple(offered), config.card_buckets)
        + _histogram(relics, config.relic_buckets)
        + _histogram(events, config.event_buckets)
        + _histogram(tuple(potions) + tuple(shops), config.item_buckets)
    )
    if len(features) != config.feature_dimension:
        raise AssertionError("A20 cold-start feature dimension changed unexpectedly")
    return tuple(features)


def build_prefix_examples(record: dict[str, Any], config: A20ColdStartConfig | None = None) -> tuple[PrefixExample, ...]:
    if not record.get("run_id"):
        raise ValueError("A20 run record requires run_id")
    config = config or A20ColdStartConfig()
    final_floor_reached = int(record.get("floor_reached") or 0)
    final_floor = float(final_floor_reached) / 60.0
    return tuple(
        PrefixExample(
            run_id=str(record["run_id"]),
            prefix_floor=floor,
            features=encode_prefix(record, floor, config),
            heart_win=float(bool(record.get("heart_victory"))),
            normal_win=float(bool(record.get("victory"))),
            final_floor=final_floor,
        )
        for floor in PREFIX_FLOORS
        if floor <= final_floor_reached
    )


class A20ValueNetwork(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.heart = nn.Linear(hidden_dimension, 1)
        self.normal = nn.Linear(hidden_dimension, 1)
        self.floor = nn.Linear(hidden_dimension, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.body(features)
        return self.heart(hidden).squeeze(-1), self.normal(hidden).squeeze(-1), self.floor(hidden).squeeze(-1)


def _split_examples(examples: list[PrefixExample]) -> dict[str, list[PrefixExample]]:
    splits = {"train": [], "validation": [], "test": []}
    for example in examples:
        digest = int(hashlib.blake2b(example.run_id.encode("utf-8"), digest_size=2).hexdigest(), 16) % 100
        split = "train" if digest < 80 else "validation" if digest < 90 else "test"
        splits[split].append(example)
    return splits


def _auc(scores: list[float], labels: list[float]) -> float:
    positives = sum(label > 0.5 for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = sum(index + 1 for index, source in enumerate(order) if labels[source] > 0.5)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _prediction_metrics(
    heart_scores: list[float],
    normal_scores: list[float],
    floor_values: list[float],
    examples: list[PrefixExample],
) -> dict[str, float]:
    heart_labels = [example.heart_win for example in examples]
    normal_labels = [example.normal_win for example in examples]
    floors = [example.final_floor for example in examples]
    return {
        "examples": float(len(examples)),
        "heart_auc": _auc(heart_scores, heart_labels),
        "normal_auc": _auc(normal_scores, normal_labels),
        "heart_brier": sum((score - label) ** 2 for score, label in zip(heart_scores, heart_labels)) / len(examples),
        "normal_brier": sum((score - label) ** 2 for score, label in zip(normal_scores, normal_labels)) / len(examples),
        "floor_mae": sum(abs(value - target) for value, target in zip(floor_values, floors)) / len(examples),
        "mean_floor_target": sum(floors) / len(floors),
    }


def evaluate_value_model(model: A20ValueNetwork, examples: list[PrefixExample], device: torch.device) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot evaluate A20 cold-start model without examples")
    model.eval()
    features = torch.tensor([example.features for example in examples], dtype=torch.float32, device=device)
    with torch.no_grad():
        heart_logits, normal_logits, floor_predictions = model(features)
    heart_scores = torch.sigmoid(heart_logits).cpu().tolist()
    normal_scores = torch.sigmoid(normal_logits).cpu().tolist()
    floor_values = floor_predictions.cpu().tolist()
    metrics: dict[str, Any] = _prediction_metrics(
        heart_scores,
        normal_scores,
        floor_values,
        examples,
    )
    metrics["by_prefix_floor"] = {}
    for prefix_floor in PREFIX_FLOORS:
        indices = [
            index for index, example in enumerate(examples) if example.prefix_floor == prefix_floor
        ]
        if indices:
            metrics["by_prefix_floor"][str(prefix_floor)] = _prediction_metrics(
                [heart_scores[index] for index in indices],
                [normal_scores[index] for index in indices],
                [floor_values[index] for index in indices],
                [examples[index] for index in indices],
            )
    return metrics


def train_value_model(
    examples: list[PrefixExample],
    *,
    epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    seed: int = 17,
    device: str = "cpu",
) -> tuple[A20ValueNetwork, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train A20 cold-start model without examples")
    torch.manual_seed(seed)
    target_device = torch.device(device)
    splits = _split_examples(examples)
    if any(not split for split in splits.values()):
        raise ValueError("A20 cold-start split is empty")
    model = A20ValueNetwork(len(examples[0].features)).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_features = torch.tensor([example.features for example in splits["train"]], dtype=torch.float32, device=target_device)
    train_heart = torch.tensor([example.heart_win for example in splits["train"]], dtype=torch.float32, device=target_device)
    train_normal = torch.tensor([example.normal_win for example in splits["train"]], dtype=torch.float32, device=target_device)
    train_floor = torch.tensor([example.final_floor for example in splits["train"]], dtype=torch.float32, device=target_device)
    heart_positive = max(1.0, float(train_heart.sum()))
    heart_negative = max(1.0, float(train_heart.numel()) - heart_positive)
    heart_weight = torch.tensor(heart_negative / heart_positive, device=target_device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    metrics: dict[str, Any] = {"epochs": [], "splits": {name: len(values) for name, values in splits.items()}}
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_features.shape[0], generator=generator)
        losses = []
        for start in range(0, train_features.shape[0], batch_size):
            indices = permutation[start : start + batch_size].to(target_device)
            heart_logits, normal_logits, floor_predictions = model(train_features[indices])
            loss = (
                functional.binary_cross_entropy_with_logits(heart_logits, train_heart[indices], pos_weight=heart_weight)
                + functional.binary_cross_entropy_with_logits(normal_logits, train_normal[indices])
                + functional.smooth_l1_loss(floor_predictions, train_floor[indices])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_value_model(model, splits["validation"], target_device)
        metrics["epochs"].append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), **validation})
    metrics["test"] = evaluate_value_model(model, splits["test"], target_device)
    return model, metrics


def read_a20_records(path: str | Path, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
                if max_records is not None and len(records) >= max_records:
                    break
    return records
