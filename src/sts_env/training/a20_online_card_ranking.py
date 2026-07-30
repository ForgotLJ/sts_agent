from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.nn import functional

from sts_env.training.policies import HeuristicPolicy
from sts_env.types import Action, ActionKind, Observation, Phase


SKIP_CARD = "skip"
IRONCLAD_STARTING_DECK = ("Strike_R",) * 5 + ("Defend_R",) * 4 + ("Bash",)
IRONCLAD_STARTING_RELICS = ("Burning Blood",)


@dataclass(frozen=True, slots=True)
class A20OnlineCardRankingConfig:
    card_buckets: int = 192
    relic_buckets: int = 64
    potion_buckets: int = 32
    candidate_buckets: int = 256
    hidden_dimension: int = 128

    def __post_init__(self) -> None:
        if min(
            self.card_buckets,
            self.relic_buckets,
            self.potion_buckets,
            self.candidate_buckets,
            self.hidden_dimension,
        ) <= 0:
            raise ValueError("A20 online card-ranking dimensions must be positive")

    @property
    def feature_dimension(self) -> int:
        return 10 + self.card_buckets + self.relic_buckets + self.potion_buckets


@dataclass(frozen=True, slots=True)
class OnlineCardChoiceExample:
    run_id: str
    floor: int
    state_features: tuple[float, ...]
    candidates: tuple[str, ...]
    selected_index: int

    def __post_init__(self) -> None:
        if not self.run_id or self.floor <= 0 or len(self.candidates) < 2:
            raise ValueError("invalid online card-choice example")
        if self.selected_index < 0 or self.selected_index >= len(self.candidates):
            raise ValueError("card-choice target is outside its candidate set")


def canonical_card_id(value: object) -> str:
    text = re.sub(r"\+\d+$", "", str(value).strip())
    return "".join(character for character in text.lower() if character.isalnum())


def _bucket(value: object, buckets: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def _expand_inventory(entries: Iterable[tuple[object, object]]) -> list[str]:
    result: list[str] = []
    for item, count in entries:
        try:
            copies = max(0, int(count))
        except (TypeError, ValueError):
            copies = 0
        result.extend([canonical_card_id(item)] * copies)
    return [item for item in result if item]


def _histogram(values: Iterable[object], buckets: int) -> list[float]:
    result = [0.0] * buckets
    count = 0
    for value in values:
        normalized = canonical_card_id(value)
        if not normalized:
            continue
        result[_bucket(normalized, buckets)] += 1.0
        count += 1
    scale = max(1.0, float(count))
    return [value / scale for value in result]


def _act_for_floor(floor: int) -> int:
    if floor <= 16:
        return 1
    if floor <= 33:
        return 2
    if floor <= 50:
        return 3
    return 4


def _encode_inventory_state(
    *,
    floor: int,
    act: int,
    hp: float,
    max_hp: float,
    gold: float,
    ascension: int,
    deck: Iterable[object],
    relics: Iterable[object],
    potions: Iterable[object],
    config: A20OnlineCardRankingConfig,
) -> tuple[float, ...]:
    if floor <= 0:
        raise ValueError("card-reward floor must be positive")
    normalized_deck = [canonical_card_id(card) for card in deck]
    normalized_deck = [card for card in normalized_deck if card]
    normalized_relics = [canonical_card_id(relic) for relic in relics]
    normalized_relics = [relic for relic in normalized_relics if relic]
    normalized_potions = [canonical_card_id(potion) for potion in potions]
    normalized_potions = [potion for potion in normalized_potions if potion]
    safe_max_hp = max(1.0, max_hp)
    features = (
        [
            floor / 60.0,
            max(0, act) / 4.0,
            hp / safe_max_hp,
            safe_max_hp / 150.0,
            gold / 500.0,
            len(normalized_deck) / 60.0,
            len(set(normalized_deck)) / 40.0,
            len(normalized_relics) / 20.0,
            len(normalized_potions) / 5.0,
            max(0, ascension) / 20.0,
        ]
        + _histogram(normalized_deck, config.card_buckets)
        + _histogram(normalized_relics, config.relic_buckets)
        + _histogram(normalized_potions, config.potion_buckets)
    )
    if len(features) != config.feature_dimension:
        raise AssertionError("A20 online card-ranking feature dimension changed unexpectedly")
    return tuple(features)


def encode_online_observation(
    observation: Observation,
    config: A20OnlineCardRankingConfig | None = None,
) -> tuple[float, ...]:
    config = config or A20OnlineCardRankingConfig()
    floor = max(1, observation.floor)
    return _encode_inventory_state(
        floor=floor,
        act=observation.act or _act_for_floor(floor),
        hp=float(observation.player.hp),
        max_hp=float(observation.player.max_hp),
        gold=float(observation.player.gold),
        ascension=observation.ascension,
        deck=_expand_inventory(observation.deck),
        relics=_expand_inventory(observation.relics),
        potions=observation.potions,
        config=config,
    )


def _floor_value(values: object, floor: int, default: float) -> float:
    if not isinstance(values, list) or not values:
        return default
    index = min(max(0, floor - 1), len(values) - 1)
    try:
        return float(values[index])
    except (TypeError, ValueError):
        return default


def _entry_floor(entry: object) -> int | None:
    if not isinstance(entry, dict):
        return None
    try:
        floor = int(float(entry.get("floor")))
    except (TypeError, ValueError):
        return None
    return floor if floor > 0 else None


def _remove_card(deck: list[str], card: object) -> None:
    normalized = canonical_card_id(card)
    try:
        deck.remove(normalized)
    except ValueError:
        pass


def _reconstructed_deck_before_choice(record: dict[str, Any], floor: int) -> list[str]:
    deck = [canonical_card_id(card) for card in IRONCLAD_STARTING_DECK]
    timeline: list[tuple[int, int, dict[str, Any], str]] = []
    for index, event in enumerate(record.get("event_choices") or []):
        event_floor = _entry_floor(event)
        if event_floor is not None and event_floor < floor:
            timeline.append((event_floor, index, event, "event"))
    for index, choice in enumerate(record.get("card_choices") or []):
        choice_floor = _entry_floor(choice)
        if choice_floor is not None and choice_floor < floor:
            timeline.append((choice_floor, index, choice, "card_choice"))
    for _, _, entry, entry_type in sorted(timeline, key=lambda item: (item[0], item[3], item[1])):
        if entry_type == "event":
            for card in entry.get("cards_removed") or []:
                _remove_card(deck, card)
            for card in entry.get("cards_obtained") or []:
                normalized = canonical_card_id(card)
                if normalized:
                    deck.append(normalized)
            continue
        selected = canonical_card_id(entry.get("picked") or "")
        if selected and selected != canonical_card_id("SKIP"):
            deck.append(selected)
    return deck


def _prefix_inventory(record: dict[str, Any], field: str, floor: int) -> list[str]:
    return [
        canonical_card_id(entry.get("key") or entry.get("name") or "")
        for entry in record.get(field) or []
        if _entry_floor(entry) is not None
        and _entry_floor(entry) < floor
        and canonical_card_id(entry.get("key") or entry.get("name") or "")
    ]


def encode_run_summary_card_state(
    record: dict[str, Any],
    floor: int,
    config: A20OnlineCardRankingConfig | None = None,
) -> tuple[float, ...]:
    if str(record.get("character")) != "IRONCLAD":
        raise ValueError("online card cold start currently supports Ironclad only")
    config = config or A20OnlineCardRankingConfig()
    max_hp = max(1.0, _floor_value(record.get("max_hp_per_floor"), floor, 80.0))
    return _encode_inventory_state(
        floor=floor,
        act=_act_for_floor(floor),
        hp=_floor_value(record.get("current_hp_per_floor"), floor, max_hp),
        max_hp=max_hp,
        gold=_floor_value(record.get("gold_per_floor"), floor, 0.0),
        ascension=int(record.get("ascension_level") or 0),
        deck=_reconstructed_deck_before_choice(record, floor),
        relics=[*IRONCLAD_STARTING_RELICS, *_prefix_inventory(record, "relics_obtained", floor)],
        potions=_prefix_inventory(record, "potions_obtained", floor),
        config=config,
    )


def _choice_candidates(choice: dict[str, Any]) -> tuple[tuple[str, ...], int] | None:
    selected = canonical_card_id(choice.get("picked") or "")
    if not selected:
        return None
    if selected == canonical_card_id("SKIP"):
        selected = SKIP_CARD
    candidates: list[str] = []
    for candidate in choice.get("not_picked") or []:
        normalized = canonical_card_id(candidate)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    if selected != SKIP_CARD and selected not in candidates:
        candidates.append(selected)
    if SKIP_CARD not in candidates:
        candidates.append(SKIP_CARD)
    if selected not in candidates:
        return None
    return tuple(candidates), candidates.index(selected)


def build_online_card_choice_examples(
    record: dict[str, Any],
    config: A20OnlineCardRankingConfig | None = None,
) -> tuple[OnlineCardChoiceExample, ...]:
    if not record.get("run_id"):
        raise ValueError("online card-ranking record requires run_id")
    config = config or A20OnlineCardRankingConfig()
    final_floor = int(record.get("floor_reached") or 0)
    result: list[OnlineCardChoiceExample] = []
    seen: set[tuple[int, tuple[str, ...], int]] = set()
    for choice in record.get("card_choices") or []:
        floor = _entry_floor(choice)
        if floor is None or floor > final_floor:
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
            OnlineCardChoiceExample(
                run_id=str(record["run_id"]),
                floor=floor,
                state_features=encode_run_summary_card_state(record, floor, config),
                candidates=candidates,
                selected_index=selected_index,
            )
        )
    return tuple(result)


class A20OnlineCardRanker(nn.Module):
    def __init__(
        self,
        state_dimension: int,
        config: A20OnlineCardRankingConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or A20OnlineCardRankingConfig()
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


def split_online_card_choice_examples(
    examples: list[OnlineCardChoiceExample],
) -> dict[str, list[OnlineCardChoiceExample]]:
    splits = {"train": [], "validation": [], "test": []}
    for example in examples:
        value = int(hashlib.blake2b(example.run_id.encode("utf-8"), digest_size=2).hexdigest(), 16) % 100
        split = "train" if value < 80 else "validation" if value < 90 else "test"
        splits[split].append(example)
    return splits


def _batch_tensors(
    examples: list[OnlineCardChoiceExample],
    indices: torch.Tensor,
    config: A20OnlineCardRankingConfig,
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


def evaluate_online_card_ranker(
    model: A20OnlineCardRanker,
    examples: list[OnlineCardChoiceExample],
    device: torch.device,
) -> dict[str, float]:
    if not examples:
        raise ValueError("cannot evaluate an empty online card-choice split")
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


def train_online_card_ranker(
    examples: list[OnlineCardChoiceExample],
    *,
    config: A20OnlineCardRankingConfig | None = None,
    epochs: int = 8,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    seed: int = 17,
    device: str = "cpu",
) -> tuple[A20OnlineCardRanker, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train an online card ranker without examples")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("invalid online card-ranker training parameters")
    config = config or A20OnlineCardRankingConfig()
    if any(len(example.state_features) != config.feature_dimension for example in examples):
        raise ValueError("online card-ranking examples do not match the feature contract")
    torch.manual_seed(seed)
    target_device = torch.device(device)
    splits = split_online_card_choice_examples(examples)
    if any(not values for values in splits.values()):
        raise ValueError("online card-ranker split is empty")
    model = A20OnlineCardRanker(config.feature_dimension, config).to(target_device)
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
                **evaluate_online_card_ranker(model, splits["validation"], target_device),
            }
        )
    metrics["test"] = evaluate_online_card_ranker(model, splits["test"], target_device)
    return model, metrics


def checkpoint_config(config: A20OnlineCardRankingConfig) -> dict[str, int]:
    return {
        "card_buckets": config.card_buckets,
        "relic_buckets": config.relic_buckets,
        "potion_buckets": config.potion_buckets,
        "candidate_buckets": config.candidate_buckets,
        "hidden_dimension": config.hidden_dimension,
    }


def load_online_card_ranker(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> A20OnlineCardRanker:
    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    if checkpoint.get("protocol") != "a20-heart-online-card-ranking":
        raise ValueError("checkpoint is not an observation-aligned A20 card ranker")
    config = A20OnlineCardRankingConfig(**checkpoint["config"])
    model = A20OnlineCardRanker(int(checkpoint["state_dimension"]), config).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def card_reward_candidate_id(action: Action) -> str | None:
    if action.kind is ActionKind.LEAVE and (
        action.option_type == "skip_card" or action.source_id == "skip_card"
    ):
        return SKIP_CARD
    if action.kind is not ActionKind.CHOOSE_CARD:
        return None
    source = action.source_id if isinstance(action.source_id, str) else action.label
    normalized = canonical_card_id(source)
    return normalized or None


class A20OnlineCardRewardPolicy:
    def __init__(
        self,
        model: A20OnlineCardRanker,
        fallback: Callable[[Observation, int], Action] | None = None,
    ) -> None:
        self._model = model
        self._fallback = fallback or HeuristicPolicy()

    def __call__(self, observation: Observation, step: int = 0) -> Action:
        if observation.phase is not Phase.CARD_REWARD:
            return self._fallback(observation, step)
        ranked_actions = [
            action
            for action in observation.legal_actions
            if card_reward_candidate_id(action) is not None
        ]
        if not ranked_actions:
            return self._fallback(observation, step)
        device = next(self._model.parameters()).device
        state = torch.tensor(
            [encode_online_observation(observation, self._model.config)],
            dtype=torch.float32,
            device=device,
        )
        candidate_ids = torch.tensor(
            [[_bucket(card_reward_candidate_id(action), self._model.config.candidate_buckets) for action in ranked_actions]],
            dtype=torch.long,
            device=device,
        )
        self._model.eval()
        with torch.no_grad():
            scores = self._model(state, candidate_ids)[0]
        return ranked_actions[int(scores.argmax().item())]

