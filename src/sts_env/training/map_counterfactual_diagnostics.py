from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any

from sts_env.training.map_counterfactual import validate_map_counterfactual_corpus
from sts_env.types import Action


def diagnose_map_counterfactual_corpus(
    root: str | Path,
    *,
    min_records: int = 16,
    min_contrasting_fraction: float = 0.20,
) -> dict[str, Any]:
    if min_records <= 0:
        raise ValueError("map counterfactual diagnostic minimum records must be positive")
    if not 0.0 <= min_contrasting_fraction <= 1.0:
        raise ValueError("map counterfactual contrast fraction must be in [0, 1]")
    destination = Path(root)
    validation = validate_map_counterfactual_corpus(destination)
    if not validation["valid"]:
        return {
            "protocol": "map-counterfactual-corpus-diagnostic",
            "schema_version": 1,
            "validation": validation,
            "scale_gate": {
                "eligible": False,
                "reasons": ["corpus validation failed"],
            },
        }
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    records_path = destination / str(manifest["records"]["path"])
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_counts: Counter[int] = Counter()
    room_symbols: Counter[str] = Counter()
    target_coordinates: Counter[str] = Counter()
    root_seeds: set[int] = set()
    group_floor_spreads: list[float] = []
    group_return_spreads: list[float] = []
    particle_floor_spreads: list[float] = []
    particle_return_spreads: list[float] = []
    behavior_regrets: list[float] = []
    behavior_return_regrets: list[float] = []
    behavior_best_groups = 0
    for record in records:
        root_seeds.add(int(record["seed"]))
        behavior = Action.from_dict(record["behavior_action"])
        candidates = list(record["candidates"])
        candidate_counts[len(candidates)] += 1
        candidate_floors: list[float] = []
        candidate_returns: list[float] = []
        behavior_floor: float | None = None
        behavior_return: float | None = None
        for candidate in candidates:
            action = Action.from_dict(candidate["action"])
            room_symbols[action.option_type.upper() or "<empty>"] += 1
            target_coordinates[f"{action.target_x},{action.target_y}"] += 1
            floors = [float(rollout["final_floor"]) for rollout in candidate["rollouts"]]
            returns = [float(rollout["environment_return"]) for rollout in candidate["rollouts"]]
            mean_floor = statistics.fmean(floors)
            mean_return = statistics.fmean(returns)
            candidate_floors.append(mean_floor)
            candidate_returns.append(mean_return)
            particle_floor_spreads.append(max(floors) - min(floors))
            particle_return_spreads.append(max(returns) - min(returns))
            if action == behavior:
                behavior_floor = mean_floor
                behavior_return = mean_return
        group_floor_spreads.append(max(candidate_floors) - min(candidate_floors))
        group_return_spreads.append(max(candidate_returns) - min(candidate_returns))
        if behavior_floor is None or behavior_return is None:
            raise AssertionError("validated corpus lacks a behavior action")
        best_floor = max(candidate_floors)
        best_return = max(candidate_returns)
        behavior_regrets.append(best_floor - behavior_floor)
        behavior_return_regrets.append(best_return - behavior_return)
        behavior_best_groups += int(math.isclose(best_floor, behavior_floor, abs_tol=1e-9))
    contrasting_groups = sum(spread > 1e-9 for spread in group_floor_spreads)
    contrasting_fraction = contrasting_groups / len(records) if records else 0.0
    root_seed_fraction = len(root_seeds) / len(records) if records else 0.0
    reasons: list[str] = []
    if not validation["complete"]:
        reasons.append("collection is incomplete")
    if len(records) < min_records:
        reasons.append("record count is below the pilot requirement")
    if root_seed_fraction < 1.0:
        reasons.append("pilot records do not map one-to-one to root seeds")
    if contrasting_fraction < min_contrasting_fraction:
        reasons.append("too few map decisions have counterfactual final-floor contrast")
    if len(target_coordinates) < 2:
        reasons.append("candidate map target coordinates lack diversity")
    return {
        "protocol": "map-counterfactual-corpus-diagnostic",
        "schema_version": 1,
        "validation": validation,
        "source": {
            "seed_range_name": manifest.get("seed_range_name"),
            "seed_range": manifest.get("seed_range"),
            "records_sha256": validation["records_sha256"],
            "particles_per_action": manifest.get("particles_per_action"),
            "rollout_max_steps": manifest.get("rollout_max_steps"),
        },
        "records": {
            "count": len(records),
            "unique_root_seeds": len(root_seeds),
            "root_seed_fraction": root_seed_fraction,
            "candidate_count_histogram": {
                str(count): value for count, value in sorted(candidate_counts.items())
            },
            "candidate_room_symbols": dict(sorted(room_symbols.items())),
            "candidate_target_coordinates": dict(sorted(target_coordinates.items())),
        },
        "counterfactual_contrast": {
            "final_floor_spread": _summary(group_floor_spreads),
            "environment_return_spread": _summary(group_return_spreads),
            "groups_with_final_floor_contrast": contrasting_groups,
            "final_floor_contrasting_fraction": contrasting_fraction,
        },
        "particle_variation": {
            "final_floor_spread": _summary(particle_floor_spreads),
            "environment_return_spread": _summary(particle_return_spreads),
            "candidates_with_final_floor_variation": sum(
                spread > 1e-9 for spread in particle_floor_spreads
            ),
        },
        "behavior_action": {
            "empirical_best_fraction": behavior_best_groups / len(records) if records else 0.0,
            "oracle_minus_behavior_final_floor": _summary(behavior_regrets),
            "oracle_minus_behavior_return": _summary(behavior_return_regrets),
        },
        "scale_gate": {
            "eligible": not reasons,
            "min_records": min_records,
            "min_contrasting_fraction": min_contrasting_fraction,
            "reasons": reasons,
        },
    }


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
