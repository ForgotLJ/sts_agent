from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.nn import functional

from sts_env.training.map_counterfactual import validate_map_counterfactual_corpus
from sts_env.training.policies import HeuristicPolicy
from sts_env.types import Action, ActionKind, Observation, Phase


_ROOM_SYMBOLS = ("?", "M", "E", "R", "$", "T", "B")


@dataclass(frozen=True, slots=True)
class MapActionValueConfig:
    card_buckets: int = 96
    relic_buckets: int = 48
    potion_buckets: int = 24
    history_buckets: int = 16
    hidden_dimension: int = 128
    dropout: float = 0.10
    final_floor_scale: float = 60.0

    def __post_init__(self) -> None:
        if min(
            self.card_buckets,
            self.relic_buckets,
            self.potion_buckets,
            self.history_buckets,
            self.hidden_dimension,
        ) <= 0:
            raise ValueError("map-action value dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("map-action value dropout must be in [0, 1)")
        if self.final_floor_scale <= 0:
            raise ValueError("map-action final-floor scale must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class MapActionValueExample:
    root_seed: int
    decision_index: int
    act: int
    floor: int
    candidate_index: int
    is_behavior_action: bool
    features: tuple[float, ...]
    mean_final_floor: float
    mean_environment_return: float
    final_floor_variance: float

    def __post_init__(self) -> None:
        if self.root_seed < 0 or self.decision_index < 0:
            raise ValueError("map-action example identity must be non-negative")
        if self.act not in {1, 2, 3} or self.floor <= 0 or self.candidate_index < 0:
            raise ValueError("map-action example location is invalid")
        if not self.features or not all(math.isfinite(value) for value in self.features):
            raise ValueError("map-action example features must be finite")
        if not all(
            math.isfinite(value)
            for value in (
                self.mean_final_floor,
                self.mean_environment_return,
                self.final_floor_variance,
            )
        ):
            raise ValueError("map-action example labels must be finite")
        if self.final_floor_variance < 0:
            raise ValueError("map-action final-floor variance must be non-negative")

    @property
    def group_id(self) -> tuple[int, int]:
        return (self.root_seed, self.decision_index)


class MapActionFeatureEncoder:
    def __init__(self, config: MapActionValueConfig | None = None) -> None:
        self.config = config or MapActionValueConfig()
        self.dimension = len(self._empty_features())

    def encode(self, observation: Observation, action: Action) -> tuple[float, ...]:
        features = tuple(self._features(observation, action))
        if len(features) != self.dimension:
            raise AssertionError("map-action feature dimension changed unexpectedly")
        if not all(math.isfinite(value) for value in features):
            raise ValueError("map-action encoder produced a non-finite feature")
        return features

    def _empty_features(self) -> list[float]:
        return (
            [0.0] * 18
            + [0.0] * 12
            + [0.0] * (len(_ROOM_SYMBOLS) * 4)
            + [0.0] * 8
            + [0.0] * self.config.card_buckets
            + [0.0] * self.config.relic_buckets
            + [0.0] * self.config.potion_buckets
            + [0.0] * self.config.history_buckets
        )

    def _features(self, observation: Observation, action: Action) -> list[float]:
        player = observation.player
        history = observation.history
        nodes = {(node.x, node.y): node for node in observation.map_nodes}
        target = nodes.get((action.target_x, action.target_y))
        reachable = self._reachable_nodes(target, nodes)
        immediate_children = (
            [nodes[coordinate] for coordinate in target.children if coordinate in nodes]
            if target is not None
            else []
        )
        all_nodes = list(nodes.values())
        core = [
            player.hp / max(1.0, float(player.max_hp)),
            player.max_hp / 200.0,
            player.gold / 500.0,
            len(player.statuses) / 12.0,
            sum(max(0, value) for _, value in player.statuses) / 100.0,
            observation.ascension / 20.0,
            observation.act / 3.0,
            observation.floor / 56.0,
            observation.map_x / 7.0,
            observation.map_y / 16.0,
            action.target_x / 7.0,
            action.target_y / 16.0,
            (action.target_y - observation.map_y) / 16.0,
            action.choice_index / 7.0 if action.choice_index is not None else 0.0,
            len(observation.legal_actions) / 6.0,
            len(all_nodes) / 60.0,
            float(observation.ruby_key),
            float(observation.emerald_key or observation.sapphire_key),
        ]
        run = [
            history.decisions / 100.0,
            history.rooms_visited / 60.0,
            history.combats_won / 60.0,
            history.elites_won / 10.0,
            history.bosses_won / 4.0,
            history.acts_cleared / 3.0,
            history.cards_added / 80.0,
            history.cards_removed / 20.0,
            history.potions_used / 20.0,
            history.gold_spent / 1500.0,
            history.hp_lost / 500.0,
            history.hp_healed / 500.0,
        ]
        topology = (
            self._room_histogram(all_nodes)
            + self._room_one_hot(target.symbol if target is not None else "")
            + self._room_histogram(immediate_children)
            + self._room_histogram(reachable)
        )
        topology_numeric = [
            float(target is not None),
            float(target.burning_elite) if target is not None else 0.0,
            len(immediate_children) / 4.0,
            len(reachable) / 60.0,
            self._max_remaining_layers(reachable) / 16.0,
            sum(len(node.children) > 1 for node in reachable) / max(1.0, float(len(reachable))),
            math.log1p(self._path_count(target, nodes)) / 12.0 if target is not None else 0.0,
            sum(node.burning_elite for node in reachable) / max(1.0, float(len(reachable))),
        ]
        deck = self._weighted_histogram(observation.deck, self.config.card_buckets)
        relics = self._weighted_histogram(observation.relics, self.config.relic_buckets)
        potions = self._histogram(observation.potions, self.config.potion_buckets)
        history_features = self._histogram(
            tuple(history.recent_rooms) + tuple(history.recent_actions),
            self.config.history_buckets,
        )
        return core + run + topology + topology_numeric + deck + relics + potions + history_features

    @staticmethod
    def _reachable_nodes(target: Any, nodes: dict[tuple[int, int], Any]) -> list[Any]:
        if target is None:
            return []
        reachable: list[Any] = []
        pending = [target]
        seen: set[tuple[int, int]] = set()
        while pending:
            node = pending.pop()
            coordinate = (node.x, node.y)
            if coordinate in seen:
                continue
            seen.add(coordinate)
            reachable.append(node)
            pending.extend(nodes[child] for child in node.children if child in nodes)
        return reachable

    @staticmethod
    def _path_count(target: Any, nodes: dict[tuple[int, int], Any]) -> int:
        if target is None:
            return 0
        cache: dict[tuple[int, int], int] = {}
        visiting: set[tuple[int, int]] = set()

        def count(node: Any) -> int:
            coordinate = (node.x, node.y)
            if coordinate in cache:
                return cache[coordinate]
            if coordinate in visiting:
                return 0
            visiting.add(coordinate)
            children = [nodes[child] for child in node.children if child in nodes]
            value = 1 if not children else sum(count(child) for child in children)
            visiting.remove(coordinate)
            cache[coordinate] = min(value, 1_000_000)
            return cache[coordinate]

        return count(target)

    @staticmethod
    def _max_remaining_layers(nodes: Iterable[Any]) -> int:
        values = [node.y for node in nodes]
        return max(values, default=0)

    @staticmethod
    def _room_one_hot(value: str) -> list[float]:
        normalized = str(value).upper()
        return [float(normalized == symbol) for symbol in _ROOM_SYMBOLS]

    @classmethod
    def _room_histogram(cls, nodes: Iterable[Any]) -> list[float]:
        values = [str(node.symbol).upper() for node in nodes]
        counts = [float(values.count(symbol)) for symbol in _ROOM_SYMBOLS]
        scale = max(1.0, float(len(values)))
        return [value / scale for value in counts]

    @classmethod
    def _weighted_histogram(cls, values: Iterable[tuple[str, int]], buckets: int) -> list[float]:
        result = [0.0] * buckets
        total = 0
        for value, count in values:
            copies = max(0, int(count))
            if not value or copies <= 0:
                continue
            result[cls._bucket(str(value), buckets)] += copies
            total += copies
        scale = max(1, total)
        return [value / scale for value in result]

    @classmethod
    def _histogram(cls, values: Iterable[str], buckets: int) -> list[float]:
        result = [0.0] * buckets
        total = 0
        for value in values:
            if not str(value):
                continue
            result[cls._bucket(str(value), buckets)] += 1.0
            total += 1
        scale = max(1, total)
        return [value / scale for value in result]

    @staticmethod
    def _bucket(value: str, buckets: int) -> int:
        digest = hashlib.blake2b(value.lower().encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % buckets


class MapActionValueNetwork(nn.Module):
    def __init__(self, feature_dimension: int, config: MapActionValueConfig) -> None:
        super().__init__()
        if feature_dimension <= 0:
            raise ValueError("map-action feature dimension must be positive")
        self.body = nn.Sequential(
            nn.Linear(feature_dimension, config.hidden_dimension),
            nn.LayerNorm(config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dimension, config.hidden_dimension),
            nn.GELU(),
        )
        self.value = nn.Linear(config.hidden_dimension, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(self.body(features)).squeeze(-1)


class A20MapActionValuePolicy:
    def __init__(
        self,
        model: MapActionValueNetwork,
        encoder: MapActionFeatureEncoder,
        fallback: Callable[[Observation, int], Action] | Any | None = None,
        *,
        override_margin: float,
        record_only: bool = False,
    ) -> None:
        if override_margin < 0:
            raise ValueError("map-action override margin must be non-negative")
        self._model = model
        self._encoder = encoder
        self._fallback = fallback or HeuristicPolicy()
        self._override_margin = override_margin
        self._record_only = record_only
        self._map_decisions = 0
        self._candidate_actions_scored = 0
        self._unscorable_baselines = 0
        self._model_failures = 0
        self._heuristic_actions_retained = 0
        self._overrides = 0
        self._override_advantage_total = 0.0
        self._value_best_matches_heuristic = 0
        self._best_advantages: list[float] = []

    @property
    def total_simulator_calls(self) -> int:
        return int(getattr(self._fallback, "total_simulator_calls", 0))

    @property
    def best_advantages(self) -> tuple[float, ...]:
        return tuple(self._best_advantages)

    def telemetry(self) -> dict[str, float | int]:
        return {
            "map_decisions": self._map_decisions,
            "candidate_actions_scored": self._candidate_actions_scored,
            "unscorable_baselines": self._unscorable_baselines,
            "model_failures": self._model_failures,
            "heuristic_actions_retained": self._heuristic_actions_retained,
            "overrides": self._overrides,
            "override_advantage_total": self._override_advantage_total,
            "value_best_matches_heuristic": self._value_best_matches_heuristic,
        }

    def __call__(self, observation: Observation, step: int = 0) -> Action:
        baseline = self._fallback_action(observation, None, step)
        return self._select_map_action(observation, baseline)

    def select(self, environment: Any) -> Action:
        observation = environment.observation
        baseline = self._fallback_action(observation, environment, 0)
        return self._select_map_action(observation, baseline)

    def _fallback_action(
        self,
        observation: Observation,
        environment: Any | None,
        step: int,
    ) -> Action:
        if environment is not None and hasattr(self._fallback, "select"):
            return self._fallback.select(environment)
        return self._fallback(observation, step)

    def _select_map_action(self, observation: Observation, baseline: Action) -> Action:
        if observation.phase is not Phase.MAP:
            return baseline
        candidates = tuple(
            action
            for action in observation.legal_actions
            if action.kind is ActionKind.CHOOSE_MAP_NODE
        )
        if len(candidates) < 2:
            return baseline
        self._map_decisions += 1
        if baseline not in candidates:
            self._unscorable_baselines += 1
            return baseline
        try:
            device = next(self._model.parameters()).device
            features = torch.tensor(
                [self._encoder.encode(observation, action) for action in candidates],
                dtype=torch.float32,
                device=device,
            )
            self._model.eval()
            with torch.no_grad():
                values = self._model(features).detach().cpu().tolist()
        except Exception:
            self._model_failures += 1
            return baseline
        if len(values) != len(candidates) or not all(math.isfinite(value) for value in values):
            self._model_failures += 1
            return baseline
        self._candidate_actions_scored += len(candidates)
        baseline_index = candidates.index(baseline)
        best_index = max(range(len(candidates)), key=lambda index: values[index])
        advantage = values[best_index] - values[baseline_index]
        self._best_advantages.append(advantage)
        best = candidates[best_index]
        if best is baseline:
            self._value_best_matches_heuristic += 1
        if self._record_only or best is baseline or advantage < self._override_margin:
            self._heuristic_actions_retained += 1
            return baseline
        self._overrides += 1
        self._override_advantage_total += advantage
        return best


def load_map_action_value_examples(
    corpus_root: str | Path,
    *,
    encoder: MapActionFeatureEncoder | None = None,
    require_complete: bool = True,
) -> tuple[list[MapActionValueExample], dict[str, Any]]:
    root = Path(corpus_root)
    validation = validate_map_counterfactual_corpus(root)
    if not validation["valid"]:
        raise ValueError("map counterfactual corpus is invalid: " + "; ".join(validation["errors"]))
    if require_complete and not validation["complete"]:
        raise ValueError("map counterfactual corpus is not complete")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records_path = root / str(manifest["records"]["path"])
    active_encoder = encoder or MapActionFeatureEncoder()
    examples: list[MapActionValueExample] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        observation = Observation.from_dict(record["observation"])
        behavior_action = Action.from_dict(record["behavior_action"])
        for candidate_index, candidate in enumerate(record["candidates"]):
            action = Action.from_dict(candidate["action"])
            floors = [float(rollout["final_floor"]) for rollout in candidate["rollouts"]]
            returns = [float(rollout["environment_return"]) for rollout in candidate["rollouts"]]
            mean_floor = sum(floors) / len(floors)
            mean_return = sum(returns) / len(returns)
            variance = sum((value - mean_floor) ** 2 for value in floors) / len(floors)
            examples.append(
                MapActionValueExample(
                    root_seed=int(record["seed"]),
                    decision_index=int(record["decision_index"]),
                    act=int(record["act"]),
                    floor=int(record["floor"]),
                    candidate_index=candidate_index,
                    is_behavior_action=action == behavior_action,
                    features=active_encoder.encode(observation, action),
                    mean_final_floor=mean_floor,
                    mean_environment_return=mean_return,
                    final_floor_variance=variance,
                )
            )
    if not examples:
        raise ValueError("map counterfactual corpus contains no action examples")
    if any(len(example.features) != active_encoder.dimension for example in examples):
        raise AssertionError("map-action corpus feature dimension changed unexpectedly")
    return examples, manifest


def split_map_action_value_examples(
    examples: list[MapActionValueExample],
) -> dict[str, list[MapActionValueExample]]:
    if not examples:
        raise ValueError("cannot split empty map-action examples")
    splits = {"train": [], "validation": [], "test": []}
    for example in examples:
        digest = hashlib.blake2b(str(example.root_seed).encode("ascii"), digest_size=2).digest()
        bucket = int.from_bytes(digest, "little") % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        splits[split].append(example)
    return splits


def evaluate_map_action_value_model(
    model: MapActionValueNetwork,
    examples: list[MapActionValueExample],
    *,
    config: MapActionValueConfig,
    device: torch.device,
) -> dict[str, float]:
    groups = _group_examples(examples)
    if not groups:
        raise ValueError("cannot evaluate an empty map-action split")
    model.eval()
    absolute_error = 0.0
    examples_seen = 0
    pairwise_correct = 0.0
    pairwise_total = 0
    top1_oracle_matches = 0
    behavior_oracle_matches = 0
    predicted_floors: list[float] = []
    behavior_floors: list[float] = []
    oracle_floors: list[float] = []
    with torch.no_grad():
        for group in groups:
            features = torch.tensor(
                [example.features for example in group], dtype=torch.float32, device=device
            )
            predictions = model(features).cpu().tolist()
            targets = [example.mean_final_floor for example in group]
            absolute_error += sum(
                abs(prediction * config.final_floor_scale - target)
                for prediction, target in zip(predictions, targets)
            )
            examples_seen += len(group)
            predicted_index = max(range(len(group)), key=lambda index: predictions[index])
            oracle_index = max(range(len(group)), key=lambda index: targets[index])
            behavior_index = next(
                (index for index, example in enumerate(group) if example.is_behavior_action),
                None,
            )
            if behavior_index is None:
                raise ValueError("map-action group has no behavior action")
            top1_oracle_matches += int(predicted_index == oracle_index)
            behavior_oracle_matches += int(behavior_index == oracle_index)
            predicted_floors.append(targets[predicted_index])
            behavior_floors.append(targets[behavior_index])
            oracle_floors.append(targets[oracle_index])
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    target_difference = targets[left] - targets[right]
                    if abs(target_difference) < 1e-9:
                        continue
                    prediction_difference = predictions[left] - predictions[right]
                    pairwise_total += 1
                    if prediction_difference == 0:
                        pairwise_correct += 0.5
                    elif prediction_difference * target_difference > 0:
                        pairwise_correct += 1.0
    group_count = len(groups)
    return {
        "groups": float(group_count),
        "examples": float(examples_seen),
        "floor_mae": absolute_error / examples_seen,
        "top1_oracle_match": top1_oracle_matches / group_count,
        "behavior_oracle_match": behavior_oracle_matches / group_count,
        "pairwise_accuracy": pairwise_correct / pairwise_total if pairwise_total else float("nan"),
        "pairwise_examples": float(pairwise_total),
        "predicted_action_mean_final_floor": sum(predicted_floors) / group_count,
        "behavior_action_mean_final_floor": sum(behavior_floors) / group_count,
        "oracle_action_mean_final_floor": sum(oracle_floors) / group_count,
        "model_minus_behavior_final_floor": (
            sum(predicted_floors) - sum(behavior_floors)
        )
        / group_count,
        "oracle_minus_behavior_final_floor": (
            sum(oracle_floors) - sum(behavior_floors)
        )
        / group_count,
    }


def train_map_action_value_model(
    examples: list[MapActionValueExample],
    *,
    config: MapActionValueConfig | None = None,
    epochs: int = 60,
    groups_per_batch: int = 32,
    learning_rate: float = 3e-4,
    seed: int = 17,
    device: str = "cpu",
) -> tuple[MapActionValueNetwork, MapActionFeatureEncoder, dict[str, Any]]:
    if epochs <= 0 or groups_per_batch <= 0 or learning_rate <= 0:
        raise ValueError("map-action training parameters must be positive")
    active_config = config or MapActionValueConfig()
    encoder = MapActionFeatureEncoder(active_config)
    if not examples:
        raise ValueError("cannot train map-action model without examples")
    if any(len(example.features) != encoder.dimension for example in examples):
        raise ValueError("map-action examples do not match the feature contract")
    target_device = torch.device(device)
    torch.manual_seed(seed)
    splits = split_map_action_value_examples(examples)
    split_groups = {name: _group_examples(values) for name, values in splits.items()}
    if any(not groups for groups in split_groups.values()):
        raise ValueError("map-action root-seed split is empty")
    model = MapActionValueNetwork(encoder.dimension, active_config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    metrics: dict[str, Any] = {
        "splits": {
            name: {
                "examples": len(values),
                "groups": len(split_groups[name]),
                "root_seeds": len({example.root_seed for example in values}),
            }
            for name, values in splits.items()
        },
        "epochs": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_selection = float("-inf")
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(split_groups["train"]), generator=generator).tolist()
        losses: list[float] = []
        regression_losses: list[float] = []
        ranking_losses: list[float] = []
        for start in range(0, len(order), groups_per_batch):
            batch_groups = [split_groups["train"][index] for index in order[start : start + groups_per_batch]]
            features = torch.tensor(
                [example.features for group in batch_groups for example in group],
                dtype=torch.float32,
                device=target_device,
            )
            targets = torch.tensor(
                [
                    example.mean_final_floor / active_config.final_floor_scale
                    for group in batch_groups
                    for example in group
                ],
                dtype=torch.float32,
                device=target_device,
            )
            predictions = model(features)
            regression = functional.smooth_l1_loss(predictions, targets)
            ranking = _ranking_loss(predictions, targets, batch_groups)
            loss = regression + ranking
            if not torch.isfinite(loss):
                raise FloatingPointError("map-action value loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            regression_losses.append(float(regression.detach().cpu()))
            ranking_losses.append(float(ranking.detach().cpu()))
        validation = evaluate_map_action_value_model(
            model,
            splits["validation"],
            config=active_config,
            device=target_device,
        )
        selection = (
            validation["model_minus_behavior_final_floor"]
            + validation["top1_oracle_match"]
            - validation["floor_mae"] / active_config.final_floor_scale
        )
        metrics["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": sum(losses) / len(losses),
                "train_regression_loss": sum(regression_losses) / len(regression_losses),
                "train_ranking_loss": sum(ranking_losses) / len(ranking_losses),
                "validation_selection": selection,
                **validation,
            }
        )
        if selection > best_selection:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_selection = selection
    if best_state is None:
        raise AssertionError("map-action value training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics["best_epoch"] = best_epoch
    metrics["best_validation_selection"] = best_selection
    metrics["test"] = evaluate_map_action_value_model(
        model,
        splits["test"],
        config=active_config,
        device=target_device,
    )
    return model, encoder, metrics


def save_map_action_value_checkpoint(
    path: str | Path,
    model: MapActionValueNetwork,
    encoder: MapActionFeatureEncoder,
    *,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": "a20-map-action-value",
            "schema_version": 1,
            "config": encoder.config.to_dict(),
            "feature_dimension": encoder.dimension,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "metadata": metadata,
        },
        destination,
    )


def load_map_action_value_model(
    path: str | Path,
    device: str = "cpu",
) -> tuple[MapActionValueNetwork, MapActionFeatureEncoder, dict[str, Any]]:
    target_device = torch.device(device)
    checkpoint = torch.load(Path(path), map_location=target_device, weights_only=True)
    if checkpoint.get("protocol") != "a20-map-action-value":
        raise ValueError("checkpoint is not an A20 map-action value model")
    if int(checkpoint.get("schema_version", -1)) != 1:
        raise ValueError("unsupported map-action value checkpoint schema")
    config = MapActionValueConfig(**checkpoint["config"])
    encoder = MapActionFeatureEncoder(config)
    if int(checkpoint.get("feature_dimension", -1)) != encoder.dimension:
        raise ValueError("map-action value checkpoint feature dimension is inconsistent")
    model = MapActionValueNetwork(encoder.dimension, config).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, encoder, {
        "metrics": dict(checkpoint.get("metrics") or {}),
        "metadata": dict(checkpoint.get("metadata") or {}),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_examples(
    examples: list[MapActionValueExample],
) -> list[tuple[MapActionValueExample, ...]]:
    grouped: dict[tuple[int, int], list[MapActionValueExample]] = {}
    for example in examples:
        grouped.setdefault(example.group_id, []).append(example)
    groups: list[tuple[MapActionValueExample, ...]] = []
    for group_id in sorted(grouped):
        group = tuple(sorted(grouped[group_id], key=lambda example: example.candidate_index))
        if len(group) < 2:
            raise ValueError("map-action group has fewer than two candidates")
        if sum(example.is_behavior_action for example in group) != 1:
            raise ValueError("map-action group must contain exactly one behavior action")
        expected = list(range(len(group)))
        if [example.candidate_index for example in group] != expected:
            raise ValueError("map-action group candidate order is incomplete")
        groups.append(group)
    return groups


def _ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    groups: list[tuple[MapActionValueExample, ...]],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for group in groups:
        count = len(group)
        group_predictions = predictions[offset : offset + count]
        group_targets = targets[offset : offset + count]
        offset += count
        target_differences = group_targets.unsqueeze(1) - group_targets.unsqueeze(0)
        prediction_differences = group_predictions.unsqueeze(1) - group_predictions.unsqueeze(0)
        mask = torch.triu(target_differences.abs() > 1e-9, diagonal=1)
        if mask.any():
            signs = torch.sign(target_differences[mask])
            losses.append(functional.softplus(-prediction_differences[mask] * signs).mean())
    return torch.stack(losses).mean() if losses else predictions.new_zeros(())
