from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from sts_env import StsEnv
from sts_env.search.belief import EnvironmentBeliefSource
from sts_env.search.distillation import (
    PolicyValueConfig,
    PolicyValueTrainer,
    SearchTarget,
    SearchTargetBuffer,
)
from sts_env.search.fixture_backend import (
    FixtureBeliefSource,
    StochasticCombatFixtureBackend,
    exact_fixture_action_values,
)
from sts_env.search.mcts import BeliefSearchConfig, ParticleBeliefSearch
from sts_env.search.policies import (
    ParticleSearchPolicy,
    PolicyValueGreedyPolicy,
    policy_value_leaf,
    policy_value_prior,
)
from sts_env.training.candidate_q import CandidateQConfig, CandidateQTrainer
from sts_env.training.collector import SynchronousVectorCollector
from sts_env.training.evaluation import EvaluationSummary, evaluate_policy
from sts_env.training.experiment import build_runtime_manifest
from sts_env.training.policies import HeuristicPolicy, RandomPolicy
from sts_env.training.seeds import SeedSplit


@dataclass(frozen=True, slots=True)
class FixtureExperimentConfig:
    run_seeds: tuple[int, ...] = (17, 29, 43)
    candidate_q_steps: int = 4_000
    candidate_q_num_environments: int = 4
    candidate_q_collection_chunk: int = 32
    candidate_q_train_every: int = 2
    teacher_episodes: int = 128
    teacher_call_budget: int = 128
    distillation_updates: int = 600
    search_call_budgets: tuple[int, ...] = (16, 64, 256)
    training_seed_start: int = 0
    training_seed_count: int = 10_000
    evaluation_seed_start: int = 1_000_000
    evaluation_seed_count: int = 256
    max_episode_steps: int = 8
    device: str = "cpu"

    def __post_init__(self) -> None:
        if len(self.run_seeds) < 3:
            raise ValueError("formal M5 evaluation requires at least three run seeds")
        counts = (
            self.candidate_q_steps,
            self.candidate_q_num_environments,
            self.candidate_q_collection_chunk,
            self.candidate_q_train_every,
            self.teacher_episodes,
            self.teacher_call_budget,
            self.distillation_updates,
            self.max_episode_steps,
        )
        if min(counts) <= 0 or not self.search_call_budgets or min(self.search_call_budgets) <= 0:
            raise ValueError("experiment counts and budgets must be positive")
        SeedSplit(
            self.training_seed_start,
            self.training_seed_count,
            self.evaluation_seed_start,
            self.evaluation_seed_count,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_seeds"] = list(self.run_seeds)
        payload["search_call_budgets"] = list(self.search_call_budgets)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FixtureExperimentConfig:
        values = dict(payload)
        values["run_seeds"] = tuple(int(seed) for seed in values["run_seeds"])
        values["search_call_budgets"] = tuple(
            int(budget) for budget in values["search_call_budgets"]
        )
        return cls(**values)


def run_fixture_experiment(
    output_directory: str | Path,
    experiment_config: FixtureExperimentConfig,
    candidate_q_config: CandidateQConfig,
    policy_value_config: PolicyValueConfig,
    search_config: BeliefSearchConfig,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "experiment": experiment_config.to_dict(),
        "candidate_q": candidate_q_config.to_dict(),
        "policy_value": policy_value_config.to_dict(),
        "search": search_config.to_dict(),
    }
    (output / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    split = SeedSplit(
        experiment_config.training_seed_start,
        experiment_config.training_seed_count,
        experiment_config.evaluation_seed_start,
        experiment_config.evaluation_seed_count,
    )
    environment_factory = lambda: StsEnv(StochasticCombatFixtureBackend())
    evaluation_seeds = split.evaluation_seeds
    random_summary = evaluate_policy(
        environment_factory,
        RandomPolicy(seed=0xBAD5EED),
        evaluation_seeds,
        max_steps=experiment_config.max_episode_steps,
    )
    heuristic_summary = evaluate_policy(
        environment_factory,
        HeuristicPolicy(),
        evaluation_seeds,
        max_steps=experiment_config.max_episode_steps,
    )
    exact_values = exact_fixture_action_values()
    exact_action = max(exact_values, key=exact_values.get)
    exact_summary = evaluate_policy(
        environment_factory,
        lambda observation, _: (
            exact_action if exact_action in observation.legal_actions else observation.legal_actions[0]
        ),
        evaluation_seeds,
        max_steps=experiment_config.max_episode_steps,
    )
    runtime_manifest = build_runtime_manifest(Path(__file__).resolve().parents[3])
    runs: list[dict[str, Any]] = []

    for run_seed in experiment_config.run_seeds:
        run_directory = output / f"seed-{run_seed}"
        run_directory.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_directory / "tensorboard"))
        candidate = _train_candidate_q(
            candidate_q_config,
            experiment_config,
            split.training_seeds,
            run_seed,
            writer,
        )
        candidate_summary = evaluate_policy(
            environment_factory,
            lambda observation, _: candidate.greedy_action(observation),
            evaluation_seeds,
            max_steps=experiment_config.max_episode_steps,
        )
        candidate.save_checkpoint(
            run_directory / "candidate-q.pt",
            metadata={"evaluation": candidate_summary.to_dict(), "runtime_manifest": runtime_manifest},
        )

        target_buffer = _collect_search_targets(
            experiment_config,
            search_config,
            split.training_seeds,
            run_seed,
        )
        target_buffer.write_jsonl(run_directory / "search-targets.jsonl")
        policy_value = PolicyValueTrainer(
            policy_value_config,
            seed=run_seed,
            device=experiment_config.device,
        )
        last_metrics: dict[str, float] = {}
        for update in range(experiment_config.distillation_updates):
            last_metrics = policy_value.train_from_buffer(target_buffer, updates=1)
            if update % 10 == 0:
                for name, value in last_metrics.items():
                    writer.add_scalar(f"distillation/{name}", value, update)
        policy_value.save(run_directory / "policy-value.pt")
        network_summary = evaluate_policy(
            environment_factory,
            PolicyValueGreedyPolicy(policy_value),
            evaluation_seeds,
            max_steps=experiment_config.max_episode_steps,
        )
        restored_policy_value = PolicyValueTrainer.load(run_directory / "policy-value.pt")
        restored_summary = evaluate_policy(
            environment_factory,
            PolicyValueGreedyPolicy(restored_policy_value),
            evaluation_seeds,
            max_steps=experiment_config.max_episode_steps,
        )
        if restored_summary.to_dict() != network_summary.to_dict():
            raise RuntimeError("policy-value checkpoint did not reproduce evaluation exactly")

        evaluations: dict[str, dict[str, Any]] = {
            "candidate_q": _evaluation_record(candidate_summary, 0, 0.0),
            "network": _evaluation_record(network_summary, 0, 0.0),
        }
        for budget in experiment_config.search_call_budgets:
            budget_config = _with_call_budget(search_config, budget)
            pure_policy = ParticleSearchPolicy(budget_config, seed=run_seed ^ budget)
            pure_summary, pure_seconds = _timed_evaluation(
                environment_factory,
                pure_policy,
                evaluation_seeds,
                experiment_config.max_episode_steps,
            )
            combined_policy = ParticleSearchPolicy(
                budget_config,
                seed=run_seed ^ budget ^ 0xC0B1,
                prior_provider=policy_value_prior(policy_value),
                leaf_evaluator=policy_value_leaf(policy_value),
            )
            combined_summary, combined_seconds = _timed_evaluation(
                environment_factory,
                combined_policy,
                evaluation_seeds,
                experiment_config.max_episode_steps,
            )
            oracle_policy = ParticleSearchPolicy(
                budget_config,
                seed=run_seed ^ budget ^ 0x0A11CE,
                exact_clone_oracle=True,
            )
            oracle_summary, oracle_seconds = _timed_evaluation(
                environment_factory,
                oracle_policy,
                evaluation_seeds,
                experiment_config.max_episode_steps,
            )
            evaluations[f"search_{budget}"] = _evaluation_record(
                pure_summary,
                pure_policy.total_simulator_calls,
                pure_seconds,
            )
            evaluations[f"combined_{budget}"] = _evaluation_record(
                combined_summary,
                combined_policy.total_simulator_calls,
                combined_seconds,
            )
            evaluations[f"oracle_{budget}"] = _evaluation_record(
                oracle_summary,
                oracle_policy.total_simulator_calls,
                oracle_seconds,
            )
            writer.add_scalar(f"evaluation/search_{budget}", pure_summary.mean_score, 0)
            writer.add_scalar(f"evaluation/combined_{budget}", combined_summary.mean_score, 0)
            writer.add_scalar(f"evaluation/oracle_{budget}", oracle_summary.mean_score, 0)

        writer.add_scalar("evaluation/candidate_q", candidate_summary.mean_score, 0)
        writer.add_scalar("evaluation/network", network_summary.mean_score, 0)
        writer.flush()
        writer.close()
        run_record = {
            "run_seed": run_seed,
            "target_count": len(target_buffer),
            "candidate_q_steps": candidate.environment_steps,
            "distillation_steps": policy_value.gradient_steps,
            "distillation_last_metrics": last_metrics,
            "evaluations": evaluations,
        }
        (run_directory / "summary.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        runs.append(run_record)

    aggregate = _aggregate_runs(runs)
    highest_budget = max(experiment_config.search_call_budgets)
    lowest_budget = min(experiment_config.search_call_budgets)
    exhaustive_validation = all(
        ParticleBeliefSearch(
            _with_call_budget(search_config, max(256, highest_budget)),
            seed=run_seed,
        ).search(FixtureBeliefSource()).selected_action
        == exact_action
        for run_seed in experiment_config.run_seeds
    )
    search_budget_scores = [
        aggregate[f"search_{budget}"]["mean_score"]
        for budget in experiment_config.search_call_budgets
    ]
    distillation_denominator = exact_summary.mean_score - heuristic_summary.mean_score
    distillation_recovery = (
        aggregate["network"]["mean_score"] - heuristic_summary.mean_score
    ) / max(1e-12, distillation_denominator)
    gates = {
        "observation_seed_isolated": "seed" not in environment_factory().reset(seed=0)[0].to_dict(),
        "exact_fixture_search_matches": exhaustive_validation,
        "search_beats_heuristic": aggregate[f"search_{highest_budget}"]["mean_score"]
        > heuristic_summary.mean_score,
        "network_beats_heuristic": aggregate["network"]["mean_score"]
        > heuristic_summary.mean_score,
        "distillation_recovers_search_gain": distillation_recovery >= 0.8,
        "combined_low_budget_improves_search": aggregate[f"combined_{lowest_budget}"]["mean_score"]
        > aggregate[f"search_{lowest_budget}"]["mean_score"],
        "search_budget_non_decreasing": all(
            later >= earlier - 0.02
            for earlier, later in zip(search_budget_scores, search_budget_scores[1:])
        ),
        "equal_call_budget_enforced": all(
            run["evaluations"][f"search_{budget}"]["simulator_calls"]
            == run["evaluations"][f"combined_{budget}"]["simulator_calls"]
            for run in runs
            for budget in experiment_config.search_call_budgets
        ),
        "checkpoint_reproduction": True,
    }
    summary = {
        "config": configuration,
        "seed_split": {
            "training": [split.training_seeds[0], split.training_seeds[-1]],
            "evaluation": [evaluation_seeds[0], evaluation_seeds[-1]],
        },
        "exact_fixture": {
            "action_values": {action.label: value for action, value in exact_values.items()},
            "best_action": exact_action.label,
            "evaluation": exact_summary.to_dict(),
        },
        "baselines": {
            "random": random_summary.to_dict(),
            "heuristic": heuristic_summary.to_dict(),
        },
        "runs": runs,
        "aggregate": aggregate,
        "paired_improvements": {
            method: _paired_run_improvement(runs, method, heuristic_summary)
            for method in aggregate
        },
        "gates": gates,
        "claim_supported": all(gates.values()),
        "runtime_manifest": runtime_manifest,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _train_candidate_q(
    trainer_config: CandidateQConfig,
    experiment_config: FixtureExperimentConfig,
    training_seeds: tuple[int, ...],
    run_seed: int,
    writer: SummaryWriter,
) -> CandidateQTrainer:
    trainer = CandidateQTrainer(
        trainer_config,
        seed=run_seed,
        device=experiment_config.device,
    )
    collector = SynchronousVectorCollector(
        environment_factory=lambda: StsEnv(StochasticCombatFixtureBackend()),
        num_environments=experiment_config.candidate_q_num_environments,
        seeds=training_seeds,
    )
    while trainer.environment_steps < experiment_config.candidate_q_steps:
        remaining = experiment_config.candidate_q_steps - trainer.environment_steps
        batch = collector.collect(
            min(experiment_config.candidate_q_collection_chunk, remaining),
            lambda observation, _: trainer.select_action(observation, explore=True),
        )
        for transition in batch.transitions:
            trainer.observe(transition)
            if trainer.environment_steps % experiment_config.candidate_q_train_every == 0:
                metrics = trainer.train_step()
                if metrics is not None:
                    writer.add_scalar("candidate_q/loss", metrics["loss"], trainer.environment_steps)
            writer.add_scalar("candidate_q/epsilon", trainer.epsilon, trainer.environment_steps)
    return trainer


def _collect_search_targets(
    experiment_config: FixtureExperimentConfig,
    search_config: BeliefSearchConfig,
    training_seeds: tuple[int, ...],
    run_seed: int,
) -> SearchTargetBuffer:
    buffer = SearchTargetBuffer(
        capacity=experiment_config.teacher_episodes * experiment_config.max_episode_steps,
        seed=run_seed,
    )
    teacher_config = _with_call_budget(search_config, experiment_config.teacher_call_budget)
    for episode_index, seed in enumerate(training_seeds[: experiment_config.teacher_episodes]):
        environment = StsEnv(StochasticCombatFixtureBackend())
        observation, _ = environment.reset(seed=seed)
        for step_index in range(experiment_config.max_episode_steps):
            search = ParticleBeliefSearch(
                teacher_config,
                seed=run_seed ^ (episode_index << 16) ^ step_index,
            )
            result = search.search(EnvironmentBeliefSource(environment))
            buffer.add(SearchTarget.from_search_result(observation, result))
            observation, _, terminated, truncated, _ = environment.step(result.selected_action)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("teacher episode exceeded fixture step limit")
    return buffer


def _with_call_budget(config: BeliefSearchConfig, budget: int) -> BeliefSearchConfig:
    return BeliefSearchConfig(
        **{
            **config.to_dict(),
            "simulations": max(config.simulations, budget),
            "simulator_call_budget": budget,
        }
    )


def _timed_evaluation(
    environment_factory: Any,
    policy: Any,
    seeds: tuple[int, ...],
    max_steps: int,
) -> tuple[EvaluationSummary, float]:
    started = time.perf_counter()
    summary = evaluate_policy(environment_factory, policy, seeds, max_steps=max_steps)
    return summary, time.perf_counter() - started


def _evaluation_record(
    summary: EvaluationSummary,
    simulator_calls: int,
    wall_seconds: float,
) -> dict[str, Any]:
    decisions = sum(episode.length for episode in summary.episodes)
    return {
        **summary.to_dict(),
        "simulator_calls": simulator_calls,
        "calls_per_decision": simulator_calls / max(1, decisions),
        "wall_seconds": wall_seconds,
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    methods = runs[0]["evaluations"].keys()
    aggregate: dict[str, dict[str, float]] = {}
    for method in methods:
        scores = [run["evaluations"][method]["mean_score"] for run in runs]
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        aggregate[method] = {
            "mean_score": statistics.mean(scores),
            "run_ci95": 1.96 * score_std / math.sqrt(len(scores)),
            "mean_win_rate": statistics.mean(
                run["evaluations"][method]["win_rate"] for run in runs
            ),
            "mean_simulator_calls": statistics.mean(
                run["evaluations"][method]["simulator_calls"] for run in runs
            ),
            "mean_calls_per_decision": statistics.mean(
                run["evaluations"][method]["calls_per_decision"] for run in runs
            ),
            "mean_wall_seconds": statistics.mean(
                run["evaluations"][method]["wall_seconds"] for run in runs
            ),
        }
    return aggregate


def _paired_run_improvement(
    runs: list[dict[str, Any]],
    method: str,
    baseline: EvaluationSummary,
) -> dict[str, float]:
    run_differences: list[float] = []
    baseline_scores = {episode.seed: episode.score for episode in baseline.episodes}
    for run in runs:
        episodes = run["evaluations"][method]["episodes"]
        differences = [episode["score"] - baseline_scores[episode["seed"]] for episode in episodes]
        run_differences.append(statistics.mean(differences))
    difference_std = statistics.stdev(run_differences) if len(run_differences) > 1 else 0.0
    return {
        "mean_difference": statistics.mean(run_differences),
        "run_ci95": 1.96 * difference_std / math.sqrt(len(run_differences)),
    }
