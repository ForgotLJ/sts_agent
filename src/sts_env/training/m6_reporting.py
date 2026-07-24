from __future__ import annotations

import hashlib
import random
import statistics
from typing import Any


_RUN_METRICS = (
    "win_rate",
    "act1_clear_rate",
    "act2_clear_rate",
    "act3_clear_rate",
    "mean_floor",
    "median_floor",
    "mean_final_hp",
    "mean_proxy_score",
    "mean_decisions",
    "total_simulator_calls",
    "total_wall_seconds",
    "errors",
    "crashes",
    "illegal_actions",
    "recovery_failures",
    "truncations",
    "timeouts",
    "cycles",
)

_PAIRED_METRICS = {
    "win": lambda episode: float(bool(episode["won"])),
    "final_floor": lambda episode: float(episode["final_floor"]),
    "final_hp": lambda episode: float(episode["final_hp"]),
    "proxy_score": lambda episode: float(episode["proxy_score"]),
    "decisions": lambda episode: float(episode["decisions"]),
    "simulator_calls": lambda episode: float(episode["simulator_calls"]),
    "wall_seconds": lambda episode: float(episode["wall_seconds"]),
}


def summarize_m6_evaluations(
    evaluations: tuple[dict[str, Any], ...],
    *,
    reference_method: str = "heuristic",
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if not evaluations or not reference_method or bootstrap_samples <= 0:
        raise ValueError("M6 reporting configuration is invalid")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for evaluation in evaluations:
        method = str(evaluation["method"])
        run_seed_value = evaluation.get("run_seed")
        run_seed = int(
            evaluation["policy_seed"] if run_seed_value is None else run_seed_value
        )
        key = (run_seed, method)
        if key in indexed:
            raise ValueError(f"duplicate M6 evaluation for run {run_seed}, method {method}")
        indexed[key] = evaluation

    run_seeds = sorted({run_seed for run_seed, _ in indexed})
    runs: dict[str, Any] = {}
    for run_seed in run_seeds:
        methods = {
            method: evaluation
            for (candidate_seed, method), evaluation in indexed.items()
            if candidate_seed == run_seed
        }
        run_payload: dict[str, Any] = {
            "methods": {
                method: {
                    key: value
                    for key, value in dict(evaluation["summary"]).items()
                    if key != "episodes"
                }
                for method, evaluation in sorted(methods.items())
            },
            "paired_comparisons": {},
        }
        reference = methods.get(reference_method)
        if reference is not None:
            for method, evaluation in sorted(methods.items()):
                if method == reference_method:
                    continue
                run_payload["paired_comparisons"][f"{method}_minus_{reference_method}"] = (
                    paired_evaluation_difference(
                        evaluation,
                        reference,
                        bootstrap_samples=bootstrap_samples,
                        seed=_stable_seed(run_seed, method, reference_method),
                    )
                )
        runs[str(run_seed)] = run_payload

    aggregate: dict[str, Any] = {}
    methods = sorted({method for _, method in indexed})
    for method in methods:
        summaries = [
            dict(evaluation["summary"])
            for (run_seed, candidate_method), evaluation in indexed.items()
            if candidate_method == method
        ]
        metrics: dict[str, Any] = {}
        for metric in _RUN_METRICS:
            values = [float(summary[metric]) for summary in summaries if metric in summary]
            if not values:
                continue
            metrics[metric] = {
                "mean": statistics.mean(values),
                "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        aggregate[method] = {"run_count": len(summaries), "metrics": metrics}

    return {
        "schema_version": 1,
        "reference_method": reference_method,
        "bootstrap_samples": bootstrap_samples,
        "runs": runs,
        "aggregate": aggregate,
    }


def paired_evaluation_difference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_episodes = _episodes_by_seed(candidate)
    reference_episodes = _episodes_by_seed(reference)
    if candidate_episodes.keys() != reference_episodes.keys():
        raise ValueError("paired M6 evaluations must contain exactly the same environment seeds")
    comparisons: dict[str, Any] = {}
    ordered_seeds = sorted(candidate_episodes)
    for metric_index, (metric, extractor) in enumerate(_PAIRED_METRICS.items()):
        differences = [
            extractor(candidate_episodes[environment_seed])
            - extractor(reference_episodes[environment_seed])
            for environment_seed in ordered_seeds
        ]
        comparisons[metric] = {
            "mean_difference": statistics.mean(differences),
            "bootstrap_ci95": bootstrap_mean_interval(
                differences,
                samples=bootstrap_samples,
                seed=seed + metric_index,
            ),
        }
    return {
        "sample_count": len(ordered_seeds),
        "seed_range": [ordered_seeds[0], ordered_seeds[-1]],
        "metrics": comparisons,
    }


def bootstrap_mean_interval(
    values: list[float] | tuple[float, ...],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values or samples <= 0:
        raise ValueError("paired bootstrap requires values and a positive sample count")
    source = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[source.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return (
        means[int(0.025 * (samples - 1))],
        means[int(0.975 * (samples - 1))],
    )


def _episodes_by_seed(evaluation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = tuple(dict(episode) for episode in evaluation["summary"]["episodes"])
    indexed = {int(episode["seed"]): episode for episode in episodes}
    if len(indexed) != len(episodes) or not indexed:
        raise ValueError("M6 evaluation episodes must have unique environment seeds")
    return indexed


def _stable_seed(run_seed: int, method: str, reference_method: str) -> int:
    digest = hashlib.sha256(
        f"{run_seed}:{method}:{reference_method}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")
