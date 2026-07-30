from __future__ import annotations

import hashlib
import statistics
from typing import Any, Callable

import torch


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

_PAIRED_METRICS: dict[str, Callable[[dict[str, Any]], float]] = {
    "win": lambda episode: float(bool(episode["won"])),
    "act1_clear": lambda episode: float(
        int(episode["final_act"]) >= 2 or bool(episode["won"])
    ),
    "final_floor": lambda episode: float(episode["final_floor"]),
    "final_hp": lambda episode: float(episode["final_hp"]),
    "proxy_score": lambda episode: float(episode["proxy_score"]),
    "decisions": lambda episode: float(episode["decisions"]),
    "simulator_calls": lambda episode: float(episode["simulator_calls"]),
    "wall_seconds": lambda episode: float(episode["wall_seconds"]),
}


def summarize_m7_evaluations(
    evaluations: tuple[dict[str, Any], ...],
    *,
    reference_method: str = "heuristic",
    reference_methods: tuple[str, ...] | None = None,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    requested_references = (
        (reference_method,) if reference_methods is None else tuple(reference_methods)
    )
    if (
        not evaluations
        or not requested_references
        or any(not method for method in requested_references)
        or len(set(requested_references)) != len(requested_references)
        or bootstrap_samples <= 0
    ):
        raise ValueError("M7 reporting configuration is invalid")
    by_method: dict[str, list[dict[str, Any]]] = {}
    identities: set[tuple[str, int | None, int]] = set()
    for evaluation in evaluations:
        method = str(evaluation["method"])
        run_seed_value = evaluation.get("run_seed")
        run_seed = None if run_seed_value is None else int(run_seed_value)
        policy_seed = int(evaluation["policy_seed"])
        identity = (method, run_seed, policy_seed)
        if identity in identities:
            raise ValueError(f"duplicate M7 evaluation identity: {identity}")
        identities.add(identity)
        _episodes_by_seed(evaluation)
        by_method.setdefault(method, []).append(evaluation)
    missing_references = tuple(
        method for method in requested_references if method not in by_method
    )
    if missing_references:
        raise ValueError(
            f"M7 reference methods are missing: {', '.join(missing_references)}"
        )

    aggregate = {
        method: _aggregate_method(method_evaluations)
        for method, method_evaluations in sorted(by_method.items())
    }
    comparisons = {}
    for requested_reference in requested_references:
        references = tuple(by_method[requested_reference])
        for method, candidates in sorted(by_method.items()):
            if method == requested_reference:
                continue
            comparisons[f"{method}_minus_{requested_reference}"] = (
                hierarchical_paired_evaluation_difference(
                    tuple(candidates),
                    references,
                    bootstrap_samples=bootstrap_samples,
                    seed=_stable_seed(method, requested_reference),
                )
            )
    warnings = []
    for method, method_evaluations in sorted(by_method.items()):
        duplicate_sets = _duplicate_episode_set_count(method_evaluations)
        if duplicate_sets:
            warnings.append(
                {
                    "method": method,
                    "type": "duplicate_episode_records",
                    "duplicate_evaluation_count": duplicate_sets,
                    "detail": (
                        "identical episode records are reported once per evaluation run and "
                        "are not treated as additional environment seeds"
                    ),
                }
            )
    return {
        "schema_version": 1,
        "protocol": "m7",
        "reference_method": requested_references[0],
        "reference_methods": list(requested_references),
        "bootstrap_samples": bootstrap_samples,
        "independence_units": {
            "environment": "environment seed",
            "training": "training run seed",
            "bootstrap": "two-level resampling of training runs and environment seeds",
        },
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "warnings": warnings,
    }


def hierarchical_paired_evaluation_difference(
    candidates: tuple[dict[str, Any], ...],
    references: tuple[dict[str, Any], ...],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not candidates or not references or bootstrap_samples <= 0:
        raise ValueError("hierarchical pairing requires evaluations and bootstrap samples")
    reference_by_run = {
        int(
            reference["run_seed"]
            if reference.get("run_seed") is not None
            else reference["policy_seed"]
        ): reference
        for reference in references
    }
    paired: list[tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]] = []
    for candidate in candidates:
        run_seed_value = candidate.get("run_seed")
        candidate_pairing_seed = int(
            run_seed_value
            if run_seed_value is not None
            else candidate["policy_seed"]
        )
        reference = reference_by_run.get(candidate_pairing_seed)
        if reference is None and len(references) == 1:
            reference = references[0]
        if reference is None:
            raise ValueError("M7 candidate has no unambiguous paired reference")
        candidate_episodes = _episodes_by_seed(candidate)
        reference_episodes = _episodes_by_seed(reference)
        if candidate_episodes.keys() != reference_episodes.keys():
            raise ValueError("paired M7 evaluations require identical environment seeds")
        paired.append((candidate_episodes, reference_episodes))

    ordered_seeds = sorted(paired[0][0])
    if any(sorted(candidate) != ordered_seeds for candidate, _ in paired):
        raise ValueError("M7 candidate runs use different environment seed sets")
    metrics = {}
    for metric_index, (metric, extractor) in enumerate(_PAIRED_METRICS.items()):
        differences = [
            [
                extractor(candidate[environment_seed])
                - extractor(reference[environment_seed])
                for environment_seed in ordered_seeds
            ]
            for candidate, reference in paired
        ]
        flattened = [value for run_values in differences for value in run_values]
        metric_payload: dict[str, Any] = {
            "mean_difference": statistics.mean(flattened),
            "hierarchical_bootstrap_ci95": _hierarchical_bootstrap_interval(
                differences,
                samples=bootstrap_samples,
                seed=seed + metric_index,
            ),
            "run_mean_differences": [statistics.mean(values) for values in differences],
        }
        if metric == "final_floor":
            metric_payload["paired_outcomes"] = {
                "better": sum(value > 0 for value in flattened),
                "equal": sum(value == 0 for value in flattened),
                "worse": sum(value < 0 for value in flattened),
            }
        metrics[metric] = metric_payload
    return {
        "training_run_count": len(paired),
        "environment_seed_count": len(ordered_seeds),
        "episode_record_count": len(paired) * len(ordered_seeds),
        "seed_range": [ordered_seeds[0], ordered_seeds[-1]],
        "metrics": metrics,
    }


def _aggregate_method(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [dict(evaluation["summary"]) for evaluation in evaluations]
    episode_sets = [_episodes_by_seed(evaluation) for evaluation in evaluations]
    all_episodes = [episode for episodes in episode_sets for episode in episodes.values()]
    metrics = {}
    for metric in _RUN_METRICS:
        values = [float(summary[metric]) for summary in summaries if metric in summary]
        if not values:
            continue
        metrics[metric] = {
            "mean": statistics.mean(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    unique_environment_wins = {
        int(episode["seed"]) for episode in all_episodes if bool(episode["won"])
    }
    return {
        "evaluation_run_count": len(evaluations),
        "episode_record_count": len(all_episodes),
        "unique_environment_seed_count": len(
            {environment_seed for episodes in episode_sets for environment_seed in episodes}
        ),
        "record_wins": sum(bool(episode["won"]) for episode in all_episodes),
        "unique_environment_wins": len(unique_environment_wins),
        "winning_environment_seeds": sorted(unique_environment_wins),
        "metrics": metrics,
    }


def _episodes_by_seed(evaluation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = tuple(dict(episode) for episode in evaluation["summary"]["episodes"])
    indexed = {int(episode["seed"]): episode for episode in episodes}
    if len(indexed) != len(episodes) or not indexed:
        raise ValueError("M7 evaluation episodes must contain unique environment seeds")
    return indexed


def _duplicate_episode_set_count(evaluations: list[dict[str, Any]]) -> int:
    fingerprints = []
    for evaluation in evaluations:
        episodes = _episodes_by_seed(evaluation)
        payload = tuple(
            (
                environment_seed,
                bool(episode["won"]),
                int(episode["final_floor"]),
                int(episode["final_hp"]),
                int(episode["decisions"]),
                float(episode["proxy_score"]),
                int(episode["simulator_calls"]),
            )
            for environment_seed, episode in sorted(episodes.items())
        )
        fingerprints.append(hashlib.sha256(repr(payload).encode("utf-8")).hexdigest())
    return len(fingerprints) - len(set(fingerprints))


def _hierarchical_bootstrap_interval(
    values: list[list[float]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values or not values[0] or samples <= 0:
        raise ValueError("hierarchical bootstrap requires a rectangular value matrix")
    environment_count = len(values[0])
    if any(len(run_values) != environment_count for run_values in values):
        raise ValueError("hierarchical bootstrap value matrix is ragged")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    run_count = len(values)
    matrix = torch.tensor(values, dtype=torch.float64)
    means = []
    batch_size = 256
    for batch_start in range(0, samples, batch_size):
        current_batch = min(batch_size, samples - batch_start)
        sampled_runs = torch.randint(
            run_count,
            (current_batch, run_count),
            generator=generator,
        )
        sampled_environments = torch.randint(
            environment_count,
            (current_batch, environment_count),
            generator=generator,
        )
        sampled = matrix[
            sampled_runs[:, :, None],
            sampled_environments[:, None, :],
        ]
        means.append(sampled.mean(dim=(1, 2)))
    ordered = torch.sort(torch.cat(means)).values
    return (
        float(ordered[int(0.025 * (samples - 1))].item()),
        float(ordered[int(0.975 * (samples - 1))].item()),
    )


def _stable_seed(method: str, reference_method: str) -> int:
    digest = hashlib.sha256(
        f"m7:{method}:{reference_method}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")
