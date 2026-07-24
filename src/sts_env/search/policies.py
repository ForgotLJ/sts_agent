from __future__ import annotations

import random
from typing import Callable

import torch

from sts_env.env import StsEnv
from sts_env.search.belief import EnvironmentBeliefSource
from sts_env.search.diagnostics.oracle import ExactCloneOracleSource
from sts_env.search.distillation import PolicyValueTrainer
from sts_env.search.mcts import (
    BeliefSearchConfig,
    BeliefSearchResult,
    LeafEvaluator,
    ParticleBeliefSearch,
    PriorProvider,
    RolloutPolicy,
)
from sts_env.training.candidate_q import CandidateQTrainer
from sts_env.types import Action, Observation


class ParticleSearchPolicy:
    def __init__(
        self,
        config: BeliefSearchConfig,
        *,
        seed: int,
        prior_provider: PriorProvider | None = None,
        leaf_evaluator: LeafEvaluator | None = None,
        rollout_policy: RolloutPolicy | None = None,
        exact_clone_oracle: bool = False,
    ):
        self._random = random.Random(seed)
        self._config = config
        self._prior_provider = prior_provider
        self._leaf_evaluator = leaf_evaluator
        self._rollout_policy = rollout_policy
        self._exact_clone_oracle = exact_clone_oracle
        self.total_simulator_calls = 0
        self.decisions = 0
        self.last_result: BeliefSearchResult | None = None

    def select(self, environment: StsEnv) -> Action:
        search = ParticleBeliefSearch(
            self._config,
            prior_provider=self._prior_provider,
            leaf_evaluator=self._leaf_evaluator,
            rollout_policy=self._rollout_policy,
            seed=self._random.getrandbits(64),
        )
        source = (
            ExactCloneOracleSource(environment, allow_hidden_state=True)
            if self._exact_clone_oracle
            else EnvironmentBeliefSource(environment)
        )
        result = search.search(source)
        self.total_simulator_calls += result.simulator_calls
        self.decisions += 1
        self.last_result = result
        return result.selected_action


class PolicyValueGreedyPolicy:
    def __init__(self, trainer: PolicyValueTrainer):
        self._trainer = trainer

    def __call__(self, observation: Observation, _: int = 0) -> Action:
        return self._trainer.greedy_action(observation)


def policy_value_prior(trainer: PolicyValueTrainer, temperature: float = 1.0) -> PriorProvider:
    return lambda observation: trainer.policy(observation, temperature=temperature)


def policy_value_leaf(trainer: PolicyValueTrainer) -> LeafEvaluator:
    return trainer.value


def candidate_q_prior(
    trainer: CandidateQTrainer,
    temperature: float = 1.0,
) -> PriorProvider:
    if temperature <= 0:
        raise ValueError("prior temperature must be positive")

    def prior(observation: Observation) -> tuple[float, ...]:
        values = trainer.q_values(observation)
        if values.numel() == 0:
            return ()
        probabilities = torch.softmax(values / temperature, dim=0)
        return tuple(float(value) for value in probabilities.cpu())

    return prior
