from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Callable

from sts_env.env import StsEnv
from sts_env.search.belief import BeliefSource, public_observation_key
from sts_env.types import Action, Observation, Phase


PriorProvider = Callable[[Observation], tuple[float, ...]]
LeafEvaluator = Callable[[Observation], float]
RolloutPolicy = Callable[[Observation], Action]


@dataclass(frozen=True, slots=True)
class BeliefSearchConfig:
    simulations: int = 128
    simulator_call_budget: int | None = None
    max_depth: int = 32
    discount: float = 0.99
    exploration: float = 1.5
    chance_widening_coefficient: float = 2.0
    chance_widening_exponent: float = 0.5
    rollout_depth: int = 0

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.simulator_call_budget is not None and self.simulator_call_budget <= 0:
            raise ValueError("simulator call budget must be positive when provided")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if self.exploration < 0:
            raise ValueError("exploration must be non-negative")
        if self.chance_widening_coefficient <= 0:
            raise ValueError("chance widening coefficient must be positive")
        if not 0 < self.chance_widening_exponent <= 1:
            raise ValueError("chance widening exponent must be in (0, 1]")
        if self.rollout_depth < 0:
            raise ValueError("rollout depth must be non-negative")

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, int | float | None]) -> BeliefSearchConfig:
        return cls(**payload)


@dataclass(slots=True)
class _DecisionNode:
    observation: Observation
    visits: int = 0
    edges: dict[Action, _ActionEdge] = field(default_factory=dict)


@dataclass(slots=True)
class _ActionEdge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    outcomes: dict[str, _DecisionNode] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True, slots=True)
class SearchActionStat:
    action: Action
    visits: int
    probability: float
    mean_value: float
    prior: float
    outcome_count: int


@dataclass(frozen=True, slots=True)
class BeliefSearchResult:
    selected_action: Action
    root_value: float
    actions: tuple[SearchActionStat, ...]
    simulator_calls: int
    expanded_nodes: int
    sampled_worlds: int

    @property
    def policy(self) -> dict[Action, float]:
        return {stat.action: stat.probability for stat in self.actions}


class ParticleBeliefSearch:
    def __init__(
        self,
        config: BeliefSearchConfig | None = None,
        *,
        prior_provider: PriorProvider | None = None,
        leaf_evaluator: LeafEvaluator | None = None,
        rollout_policy: RolloutPolicy | None = None,
        seed: int = 0,
    ):
        self.config = config or BeliefSearchConfig()
        self._prior_provider = prior_provider or uniform_prior
        self._leaf_evaluator = leaf_evaluator or default_leaf_value
        self._rollout_policy = rollout_policy
        self._random = random.Random(seed)
        self._simulator_calls = 0
        self._expanded_nodes = 0

    def search(self, source: BeliefSource) -> BeliefSearchResult:
        observation = source.observation
        if not observation.legal_actions:
            raise ValueError("cannot search a terminal observation")
        root = self._new_node(observation)
        self._simulator_calls = 0
        self._expanded_nodes = 1
        sampled_worlds = 0

        for _ in range(self.config.simulations):
            if self._budget_exhausted():
                break
            search_seed = self._random.getrandbits(64)
            environment = source.sample(search_seed)
            if environment.observation != observation:
                raise RuntimeError("belief source returned a different public root")
            self._simulate(root, environment, depth=0, root_phase=observation.phase)
            sampled_worlds += 1

        total_visits = sum(edge.visits for edge in root.edges.values())
        if total_visits != sampled_worlds:
            raise AssertionError("each simulation must update exactly one root action")
        if total_visits == 0:
            raise RuntimeError("search budget did not permit a root action evaluation")
        stats = tuple(
            SearchActionStat(
                action=action,
                visits=edge.visits,
                probability=edge.visits / total_visits,
                mean_value=edge.mean_value,
                prior=edge.prior,
                outcome_count=len(edge.outcomes),
            )
            for action, edge in root.edges.items()
        )
        selected = max(
            enumerate(stats),
            key=lambda item: (item[1].visits, item[1].mean_value, -item[0]),
        )[1].action
        root_value = sum(stat.probability * stat.mean_value for stat in stats)
        return BeliefSearchResult(
            selected_action=selected,
            root_value=root_value,
            actions=stats,
            simulator_calls=self._simulator_calls,
            expanded_nodes=self._expanded_nodes,
            sampled_worlds=sampled_worlds,
        )

    def _simulate(
        self,
        node: _DecisionNode,
        environment: StsEnv,
        *,
        depth: int,
        root_phase: Phase,
    ) -> float:
        if node.observation != environment.observation:
            raise RuntimeError("tree node and sampled world disagree on public observation")
        if self._budget_exhausted():
            return self._leaf_evaluator(node.observation)
        edge_action, edge = self._select_edge(node)
        next_observation, reward, terminated, truncated, _ = environment.step(edge_action)
        self._simulator_calls += 1

        phase_changed = next_observation.phase is not root_phase
        if terminated or truncated or phase_changed:
            continuation = self._leaf_evaluator(next_observation)
        elif depth + 1 >= self.config.max_depth:
            continuation = self._evaluate_leaf(environment, root_phase)
        else:
            outcome_key = public_observation_key(next_observation)
            child = edge.outcomes.get(outcome_key)
            if child is None and len(edge.outcomes) < self._outcome_capacity(edge.visits + 1):
                child = self._new_node(next_observation)
                edge.outcomes[outcome_key] = child
                self._expanded_nodes += 1
                continuation = self._evaluate_leaf(environment, root_phase)
            elif child is None:
                continuation = self._evaluate_leaf(environment, root_phase)
            else:
                continuation = self._simulate(
                    child,
                    environment,
                    depth=depth + 1,
                    root_phase=root_phase,
                )

        value = reward + self.config.discount * continuation
        node.visits += 1
        edge.visits += 1
        edge.value_sum += value
        return value

    def _select_edge(self, node: _DecisionNode) -> tuple[Action, _ActionEdge]:
        unvisited = [item for item in node.edges.items() if item[1].visits == 0]
        if unvisited:
            return max(
                enumerate(unvisited),
                key=lambda item: (item[1][1].prior, -item[0]),
            )[1]
        parent_scale = math.sqrt(max(1, node.visits))
        return max(
            enumerate(node.edges.items()),
            key=lambda item: (
                item[1][1].mean_value
                + self.config.exploration
                * item[1][1].prior
                * parent_scale
                / (1 + item[1][1].visits),
                -item[0],
            ),
        )[1]

    def _new_node(self, observation: Observation) -> _DecisionNode:
        priors = self._prior_provider(observation)
        if len(priors) != len(observation.legal_actions):
            raise ValueError("prior count must equal legal action count")
        if any(not math.isfinite(value) or value < 0 for value in priors):
            raise ValueError("action priors must be finite and non-negative")
        total = sum(priors)
        if total <= 0:
            raise ValueError("at least one action prior must be positive")
        normalized = tuple(value / total for value in priors)
        return _DecisionNode(
            observation=observation,
            edges={
                action: _ActionEdge(prior=prior)
                for action, prior in zip(observation.legal_actions, normalized, strict=True)
            },
        )

    def _outcome_capacity(self, visits: int) -> int:
        return max(
            1,
            math.ceil(
                self.config.chance_widening_coefficient
                * visits**self.config.chance_widening_exponent
            ),
        )

    def _budget_exhausted(self) -> bool:
        budget = self.config.simulator_call_budget
        return budget is not None and self._simulator_calls >= budget

    def _evaluate_leaf(self, environment: StsEnv, root_phase: Phase) -> float:
        if self._rollout_policy is None or self.config.rollout_depth == 0:
            return self._leaf_evaluator(environment.observation)
        value = 0.0
        discount = 1.0
        for _ in range(self.config.rollout_depth):
            if self._budget_exhausted():
                break
            observation = environment.observation
            if observation.phase is not root_phase or not observation.legal_actions:
                break
            action = self._rollout_policy(observation)
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            self._simulator_calls += 1
            value += discount * reward
            discount *= self.config.discount
            if terminated or truncated or next_observation.phase is not root_phase:
                return value + discount * self._leaf_evaluator(next_observation)
        return value + discount * self._leaf_evaluator(environment.observation)


def uniform_prior(observation: Observation) -> tuple[float, ...]:
    if not observation.legal_actions:
        return ()
    probability = 1.0 / len(observation.legal_actions)
    return (probability,) * len(observation.legal_actions)


def default_leaf_value(observation: Observation) -> float:
    hp_ratio = observation.player.hp / max(1, observation.player.max_hp)
    if observation.phase is Phase.TERMINAL:
        return hp_ratio
    living = [enemy for enemy in observation.enemies if enemy.hp > 0]
    enemy_ratio = (
        sum(enemy.hp / max(1, enemy.max_hp) for enemy in living) / len(living)
        if living
        else 0.0
    )
    incoming = sum(
        enemy.intent_damage * max(1, enemy.intent_hits)
        for enemy in living
    )
    incoming_scale = incoming / max(1, observation.player.max_hp)
    return hp_ratio - 0.5 * enemy_ratio - 0.1 * incoming_scale
