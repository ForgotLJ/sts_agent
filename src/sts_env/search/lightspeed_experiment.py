from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from sts_env import LightspeedBackend, Phase, StsEnv
from sts_env.search.belief import EnvironmentBeliefSource
from sts_env.search.distillation import (
    PolicyValueConfig,
    PolicyValueTrainer,
    SearchTarget,
    SearchTargetBuffer,
)
from sts_env.search.mcts import BeliefSearchConfig, ParticleBeliefSearch, default_leaf_value
from sts_env.search.policies import (
    ParticleSearchPolicy,
    PolicyValueGreedyPolicy,
    policy_value_leaf,
    policy_value_prior,
)
from sts_env.training.candidate_q import CandidateQConfig, CandidateQTrainer
from sts_env.training.experiment import build_runtime_manifest
from sts_env.training.policies import HeuristicPolicy, RandomPolicy
from sts_env.training.replay import ReplayTransition
from sts_env.training.seeds import SeedSplit
from sts_env.types import Action, Observation


@dataclass(frozen=True, slots=True)
class LightspeedCombatExperimentConfig:
    run_seeds: tuple[int, ...] = (17, 29, 43)
    candidate_q_combat_steps: int = 10_000
    candidate_q_train_every: int = 2
    teacher_episodes: int = 64
    teacher_call_budget: int = 128
    distillation_updates: int = 1_000
    search_call_budgets: tuple[int, ...] = (16, 64, 256)
    training_seed_start: int = 0
    training_seed_count: int = 2_000
    evaluation_seed_start: int = 1_000_000
    evaluation_seed_count: int = 64
    max_navigation_steps: int = 32
    max_combat_steps: int = 128
    device: str = "cpu"

    def __post_init__(self) -> None:
        if len(self.run_seeds) < 3:
            raise ValueError("formal Lightspeed combat evaluation requires three run seeds")
        counts = (
            self.candidate_q_combat_steps,
            self.candidate_q_train_every,
            self.teacher_episodes,
            self.teacher_call_budget,
            self.distillation_updates,
            self.max_navigation_steps,
            self.max_combat_steps,
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
    def from_dict(cls, payload: dict[str, Any]) -> LightspeedCombatExperimentConfig:
        values = dict(payload)
        values["run_seeds"] = tuple(int(seed) for seed in values["run_seeds"])
        values["search_call_budgets"] = tuple(
            int(budget) for budget in values["search_call_budgets"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CombatEvaluationEpisode:
    seed: int
    score: float
    won: bool
    final_hp: int
    max_hp: int
    length: int
    simulator_calls: int


@dataclass(frozen=True, slots=True)
class CombatEvaluationSummary:
    episodes: tuple[CombatEvaluationEpisode, ...]
    mean_score: float
    score_ci95: float
    win_rate: float
    mean_final_hp: float
    mean_length: float
    simulator_calls: int
    calls_per_decision: float
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [asdict(episode) for episode in self.episodes],
            "mean_score": self.mean_score,
            "score_ci95": self.score_ci95,
            "win_rate": self.win_rate,
            "mean_final_hp": self.mean_final_hp,
            "mean_length": self.mean_length,
            "simulator_calls": self.simulator_calls,
            "calls_per_decision": self.calls_per_decision,
            "wall_seconds": self.wall_seconds,
        }


def run_lightspeed_combat_experiment(
    output_directory: str | Path,
    experiment_config: LightspeedCombatExperimentConfig,
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
    evaluation_seeds = split.evaluation_seeds
    heuristic_summary = evaluate_first_combat(
        HeuristicPolicy(), evaluation_seeds, experiment_config
    )
    random_summary = evaluate_first_combat(
        RandomPolicy(seed=0xBAD5EED), evaluation_seeds, experiment_config
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
        candidate_summary = evaluate_first_combat(
            lambda observation, _: candidate.greedy_action(observation),
            evaluation_seeds,
            experiment_config,
        )
        candidate.save_checkpoint(
            run_directory / "candidate-q.pt",
            metadata={"evaluation": candidate_summary.to_dict(), "runtime_manifest": runtime_manifest},
        )

        targets = _collect_teacher_targets(
            experiment_config,
            search_config,
            split.training_seeds,
            run_seed,
        )
        targets.write_jsonl(run_directory / "search-targets.jsonl")
        policy_value = PolicyValueTrainer(
            policy_value_config,
            seed=run_seed,
            device=experiment_config.device,
        )
        last_metrics: dict[str, float] = {}
        for update in range(experiment_config.distillation_updates):
            last_metrics = policy_value.train_from_buffer(targets)
            if update % 10 == 0:
                for name, value in last_metrics.items():
                    writer.add_scalar(f"distillation/{name}", value, update)
        policy_value.save(run_directory / "policy-value.pt")
        network_summary = evaluate_first_combat(
            PolicyValueGreedyPolicy(policy_value),
            evaluation_seeds,
            experiment_config,
        )
        restored = PolicyValueTrainer.load(run_directory / "policy-value.pt")
        restored_summary = evaluate_first_combat(
            PolicyValueGreedyPolicy(restored),
            evaluation_seeds,
            experiment_config,
        )
        if _evaluation_signature(restored_summary) != _evaluation_signature(network_summary):
            raise RuntimeError("Lightspeed policy-value checkpoint did not reproduce evaluation")

        evaluations: dict[str, dict[str, Any]] = {
            "candidate_q": candidate_summary.to_dict(),
            "network": network_summary.to_dict(),
        }
        for budget in experiment_config.search_call_budgets:
            budget_config = _with_call_budget(search_config, budget)
            pure_policy = ParticleSearchPolicy(
                budget_config,
                seed=run_seed ^ budget,
                rollout_policy=HeuristicPolicy(),
            )
            pure_summary = evaluate_first_combat(
                pure_policy,
                evaluation_seeds,
                experiment_config,
            )
            combined_policy = ParticleSearchPolicy(
                budget_config,
                seed=run_seed ^ budget ^ 0xC0B1,
                prior_provider=_blended_prior(policy_value),
                leaf_evaluator=_blended_leaf(policy_value),
                rollout_policy=HeuristicPolicy(),
            )
            combined_summary = evaluate_first_combat(
                combined_policy,
                evaluation_seeds,
                experiment_config,
            )
            oracle_policy = ParticleSearchPolicy(
                budget_config,
                seed=run_seed ^ budget ^ 0x0A11CE,
                rollout_policy=HeuristicPolicy(),
                exact_clone_oracle=True,
            )
            oracle_summary = evaluate_first_combat(
                oracle_policy,
                evaluation_seeds,
                experiment_config,
            )
            evaluations[f"search_{budget}"] = pure_summary.to_dict()
            evaluations[f"combined_{budget}"] = combined_summary.to_dict()
            evaluations[f"oracle_{budget}"] = oracle_summary.to_dict()
            writer.add_scalar(f"evaluation/search_{budget}", pure_summary.mean_score, 0)
            writer.add_scalar(f"evaluation/combined_{budget}", combined_summary.mean_score, 0)
            writer.add_scalar(f"evaluation/oracle_{budget}", oracle_summary.mean_score, 0)

        writer.add_scalar("evaluation/candidate_q", candidate_summary.mean_score, 0)
        writer.add_scalar("evaluation/network", network_summary.mean_score, 0)
        writer.flush()
        writer.close()
        run_record = {
            "run_seed": run_seed,
            "target_count": len(targets),
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
    lowest_budget = min(experiment_config.search_call_budgets)
    highest_budget = max(experiment_config.search_call_budgets)
    search_scores = [
        aggregate[f"search_{budget}"]["mean_score"]
        for budget in experiment_config.search_call_budgets
    ]
    search_gain_over_random = (
        aggregate[f"search_{highest_budget}"]["mean_score"] - random_summary.mean_score
    )
    distillation_retention = (
        aggregate["network"]["mean_score"] - random_summary.mean_score
    ) / max(1e-12, search_gain_over_random)
    gates = {
        "observation_seed_isolated": "seed" not in _first_combat_environment(0, experiment_config).observation.to_dict(),
        "search_beats_heuristic": aggregate[f"search_{highest_budget}"]["mean_score"]
        > heuristic_summary.mean_score,
        "search_beats_network": aggregate[f"search_{highest_budget}"]["mean_score"]
        > aggregate["network"]["mean_score"],
        "network_beats_random": aggregate["network"]["mean_score"]
        > random_summary.mean_score,
        "distillation_retains_search_gain": distillation_retention >= 0.7,
        "combined_low_budget_not_worse": aggregate[f"combined_{lowest_budget}"]["mean_score"]
        >= aggregate[f"search_{lowest_budget}"]["mean_score"] - 0.01,
        "equal_call_budget_enforced": all(
            math.isclose(
                run["evaluations"][f"search_{budget}"]["calls_per_decision"],
                budget,
            )
            and math.isclose(
                run["evaluations"][f"combined_{budget}"]["calls_per_decision"],
                budget,
            )
            for run in runs
            for budget in experiment_config.search_call_budgets
        ),
        "search_budget_non_decreasing": all(
            later >= earlier - 0.03
            for earlier, later in zip(search_scores, search_scores[1:])
        ),
        "checkpoint_reproduction": True,
    }
    summary = {
        "scope": "Ironclad A0 first combat only; no claim about full-run play",
        "config": configuration,
        "seed_split": {
            "training": [split.training_seeds[0], split.training_seeds[-1]],
            "evaluation": [evaluation_seeds[0], evaluation_seeds[-1]],
        },
        "baselines": {
            "random": random_summary.to_dict(),
            "heuristic": heuristic_summary.to_dict(),
        },
        "runs": runs,
        "aggregate": aggregate,
        "paired_improvements": {
            method: _paired_improvement(runs, method, heuristic_summary)
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


def evaluate_first_combat(
    policy: Any,
    seeds: tuple[int, ...],
    config: LightspeedCombatExperimentConfig,
) -> CombatEvaluationSummary:
    started = time.perf_counter()
    episodes: list[CombatEvaluationEpisode] = []
    total_calls_before = int(getattr(policy, "total_simulator_calls", 0))
    for episode_index, seed in enumerate(seeds):
        environment = _first_combat_environment(seed, config)
        calls_before = int(getattr(policy, "total_simulator_calls", 0))
        for step_index in range(config.max_combat_steps):
            observation = environment.observation
            action = (
                policy.select(environment)
                if hasattr(policy, "select")
                else policy(observation, episode_index)
            )
            next_observation, _, terminated, truncated, _ = environment.step(action)
            combat_over = next_observation.phase is not Phase.COMBAT
            if terminated or truncated or combat_over:
                length = step_index + 1
                won = combat_over and next_observation.player.hp > 0 and not (
                    terminated and next_observation.phase is Phase.TERMINAL
                )
                break
        else:
            length = config.max_combat_steps
            won = False
            next_observation = environment.observation
        score = (
            (1.0 if won else -1.0)
            + next_observation.player.hp / max(1, next_observation.player.max_hp)
            - 0.01 * length
        )
        episodes.append(
            CombatEvaluationEpisode(
                seed=seed,
                score=score,
                won=won,
                final_hp=next_observation.player.hp,
                max_hp=next_observation.player.max_hp,
                length=length,
                simulator_calls=int(getattr(policy, "total_simulator_calls", 0)) - calls_before,
            )
        )
    scores = [episode.score for episode in episodes]
    score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    simulator_calls = int(getattr(policy, "total_simulator_calls", 0)) - total_calls_before
    decisions = sum(episode.length for episode in episodes)
    return CombatEvaluationSummary(
        episodes=tuple(episodes),
        mean_score=statistics.mean(scores),
        score_ci95=1.96 * score_std / math.sqrt(len(scores)),
        win_rate=sum(episode.won for episode in episodes) / len(episodes),
        mean_final_hp=statistics.mean(episode.final_hp for episode in episodes),
        mean_length=statistics.mean(episode.length for episode in episodes),
        simulator_calls=simulator_calls,
        calls_per_decision=simulator_calls / max(1, decisions),
        wall_seconds=time.perf_counter() - started,
    )


def _first_combat_environment(
    seed: int,
    config: LightspeedCombatExperimentConfig,
) -> StsEnv:
    environment = StsEnv(LightspeedBackend(neow_history="skipped"))
    observation, _ = environment.reset(seed=seed)
    navigator = HeuristicPolicy()
    for _ in range(config.max_navigation_steps):
        if observation.phase is Phase.COMBAT:
            return environment
        observation, _, terminated, truncated, _ = environment.step(
            navigator(observation)
        )
        if terminated or truncated:
            raise RuntimeError(f"seed {seed} terminated before its first combat")
    raise RuntimeError(f"seed {seed} did not reach combat within the navigation limit")


def _train_candidate_q(
    trainer_config: CandidateQConfig,
    experiment_config: LightspeedCombatExperimentConfig,
    training_seeds: tuple[int, ...],
    run_seed: int,
    writer: SummaryWriter,
) -> CandidateQTrainer:
    trainer = CandidateQTrainer(
        trainer_config,
        seed=run_seed,
        device=experiment_config.device,
    )
    seed_index = 0
    while trainer.environment_steps < experiment_config.candidate_q_combat_steps:
        environment = _first_combat_environment(training_seeds[seed_index], experiment_config)
        seed_index += 1
        observation = environment.observation
        for _ in range(experiment_config.max_combat_steps):
            action = trainer.select_action(observation, explore=True)
            next_observation, _, terminated, truncated, info = environment.step(action)
            combat_over = next_observation.phase is not Phase.COMBAT
            won = combat_over and next_observation.player.hp > 0 and not (
                terminated and next_observation.phase is Phase.TERMINAL
            )
            reward = 1.0 if won else -1.0 if terminated and next_observation.player.hp <= 0 else 0.0
            trainer.observe(
                ReplayTransition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated or combat_over,
                    truncated=truncated,
                    info=info,
                )
            )
            if trainer.environment_steps % experiment_config.candidate_q_train_every == 0:
                metrics = trainer.train_step()
                if metrics is not None:
                    writer.add_scalar("candidate_q/loss", metrics["loss"], trainer.environment_steps)
            observation = next_observation
            if trainer.environment_steps >= experiment_config.candidate_q_combat_steps:
                break
            if terminated or truncated or combat_over:
                break
        if seed_index >= len(training_seeds):
            raise RuntimeError("candidate-Q exhausted the Lightspeed training seed stream")
    return trainer


def _collect_teacher_targets(
    experiment_config: LightspeedCombatExperimentConfig,
    search_config: BeliefSearchConfig,
    training_seeds: tuple[int, ...],
    run_seed: int,
) -> SearchTargetBuffer:
    buffer = SearchTargetBuffer(
        capacity=experiment_config.teacher_episodes * experiment_config.max_combat_steps,
        seed=run_seed,
    )
    teacher_config = _with_call_budget(search_config, experiment_config.teacher_call_budget)
    for episode_index, seed in enumerate(training_seeds[: experiment_config.teacher_episodes]):
        environment = _first_combat_environment(seed, experiment_config)
        observation = environment.observation
        for step_index in range(experiment_config.max_combat_steps):
            result = ParticleBeliefSearch(
                teacher_config,
                rollout_policy=HeuristicPolicy(),
                seed=run_seed ^ (episode_index << 16) ^ step_index,
            ).search(EnvironmentBeliefSource(environment))
            buffer.add(
                SearchTarget.from_search_result(
                    observation,
                    result,
                    selected_action_only=True,
                )
            )
            observation, _, terminated, truncated, _ = environment.step(result.selected_action)
            if terminated or truncated or observation.phase is not Phase.COMBAT:
                break
        else:
            raise RuntimeError(f"teacher seed {seed} exceeded the combat step limit")
    if len(buffer) < 1:
        raise RuntimeError("teacher collection produced no search targets")
    return buffer


def _with_call_budget(config: BeliefSearchConfig, budget: int) -> BeliefSearchConfig:
    return BeliefSearchConfig(
        **{
            **config.to_dict(),
            "simulations": max(config.simulations, budget),
            "simulator_call_budget": budget,
        }
    )


def _blended_prior(
    trainer: PolicyValueTrainer,
    network_weight: float = 0.25,
):
    network_prior = policy_value_prior(trainer)

    def prior(observation: Observation) -> tuple[float, ...]:
        probabilities = network_prior(observation)
        if not probabilities:
            return ()
        uniform = 1.0 / len(probabilities)
        return tuple(
            network_weight * probability + (1.0 - network_weight) * uniform
            for probability in probabilities
        )

    return prior


def _blended_leaf(
    trainer: PolicyValueTrainer,
    network_weight: float = 0.25,
):
    network_leaf = policy_value_leaf(trainer)
    return lambda observation: (
        network_weight * network_leaf(observation)
        + (1.0 - network_weight) * default_leaf_value(observation)
    )


def _evaluation_signature(summary: CombatEvaluationSummary) -> tuple[Any, ...]:
    return (
        summary.episodes,
        summary.mean_score,
        summary.score_ci95,
        summary.win_rate,
        summary.mean_final_hp,
        summary.mean_length,
        summary.simulator_calls,
        summary.calls_per_decision,
    )


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for method in runs[0]["evaluations"]:
        scores = [run["evaluations"][method]["mean_score"] for run in runs]
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        aggregate[method] = {
            "mean_score": statistics.mean(scores),
            "run_ci95": 1.96 * score_std / math.sqrt(len(scores)),
            "mean_win_rate": statistics.mean(
                run["evaluations"][method]["win_rate"] for run in runs
            ),
            "mean_final_hp": statistics.mean(
                run["evaluations"][method]["mean_final_hp"] for run in runs
            ),
            "mean_calls_per_decision": statistics.mean(
                run["evaluations"][method]["calls_per_decision"] for run in runs
            ),
            "mean_wall_seconds": statistics.mean(
                run["evaluations"][method]["wall_seconds"] for run in runs
            ),
        }
    return aggregate


def _paired_improvement(
    runs: list[dict[str, Any]],
    method: str,
    baseline: CombatEvaluationSummary,
) -> dict[str, float]:
    baseline_scores = {episode.seed: episode.score for episode in baseline.episodes}
    differences: list[float] = []
    for run in runs:
        episodes = run["evaluations"][method]["episodes"]
        differences.append(
            statistics.mean(
                episode["score"] - baseline_scores[episode["seed"]]
                for episode in episodes
            )
        )
    difference_std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    return {
        "mean_difference": statistics.mean(differences),
        "run_ci95": 1.96 * difference_std / math.sqrt(len(differences)),
    }
