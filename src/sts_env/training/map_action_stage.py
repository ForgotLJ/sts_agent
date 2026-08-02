from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sts_env.training.map_action_protocol import (
    MAP_ACTION_EVALUATION_RANGE_NAMES,
    require_map_action_seed_range,
)


_SAFETY_FIELDS = (
    "errors",
    "crashes",
    "illegal_actions",
    "recovery_failures",
    "truncations",
    "timeouts",
    "cycles",
)
_EVALUATION_PROTOCOL = "a20-map-action-value-paired-lightspeed-evaluation"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_profile_margin(
    profile: dict[str, Any],
    *,
    expected_map_checkpoint_sha256: str,
    expected_card_checkpoint_sha256: str,
    quantile: str = "p80",
    expected_range_name: str = "map_value_profile",
    expected_trained_acts: frozenset[int] | None = None,
    expected_trained_floor_range: tuple[int, int] | None = None,
    expected_label_mode: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_range_name not in MAP_ACTION_EVALUATION_RANGE_NAMES:
        raise ValueError(f"unknown map profile range: {expected_range_name}")
    if profile.get("protocol") != _EVALUATION_PROTOCOL:
        errors.append("profile protocol differs")
    if not profile.get("record_only"):
        errors.append("profile is not record-only")
    _require_range(profile, expected_range_name, errors)
    _require_checkpoint(profile, "map_checkpoint", expected_map_checkpoint_sha256, errors)
    _require_checkpoint(profile, "card_checkpoint", expected_card_checkpoint_sha256, errors)
    _require_trained_acts(profile, expected_trained_acts, errors)
    _require_trained_floor_range(profile, expected_trained_floor_range, errors)
    _require_label_mode(profile, expected_label_mode, errors)
    _require_safety(profile, errors)
    telemetry = dict(profile.get("candidate_map_telemetry") or {})
    quantiles = dict(telemetry.get("best_advantage_quantiles") or {})
    try:
        margin = float(quantiles[quantile])
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"profile margin quantile is invalid: {error}")
        margin = 0.0
    if not math.isfinite(margin) or margin < 0:
        errors.append("profile margin is non-finite or negative")
    if float(telemetry.get("map_decisions", 0.0)) <= 0:
        errors.append("profile has no map decisions")
    if errors:
        raise ValueError("invalid map profile: " + "; ".join(errors))
    return {
        "protocol": "a20-map-action-margin-selection",
        "schema_version": 1,
        "quantile": quantile,
        "override_margin": margin,
        "map_checkpoint_sha256": expected_map_checkpoint_sha256,
        "card_checkpoint_sha256": expected_card_checkpoint_sha256,
        "profile_seed_range": profile["seed_range"],
        "profile_seed_range_name": profile["seed_range_name"],
    }


def map_evaluation_gate(
    evaluation: dict[str, Any],
    *,
    expected_range_name: str,
    expected_map_checkpoint_sha256: str,
    expected_card_checkpoint_sha256: str,
    require_effect: bool,
    expected_trained_acts: frozenset[int] | None = None,
    expected_trained_floor_range: tuple[int, int] | None = None,
    expected_label_mode: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if expected_range_name not in MAP_ACTION_EVALUATION_RANGE_NAMES:
        raise ValueError(f"unknown map evaluation range: {expected_range_name}")
    if evaluation.get("protocol") != _EVALUATION_PROTOCOL:
        errors.append("evaluation protocol differs")
    if evaluation.get("record_only"):
        errors.append("record-only evaluation cannot pass this gate")
    _require_range(evaluation, expected_range_name, errors)
    _require_checkpoint(evaluation, "map_checkpoint", expected_map_checkpoint_sha256, errors)
    _require_checkpoint(evaluation, "card_checkpoint", expected_card_checkpoint_sha256, errors)
    _require_trained_acts(evaluation, expected_trained_acts, errors)
    _require_trained_floor_range(evaluation, expected_trained_floor_range, errors)
    _require_label_mode(evaluation, expected_label_mode, errors)
    _require_safety(evaluation, errors)
    if evaluation.get("candidate", {}).get("method") != "a20-map-action-value":
        errors.append("candidate method differs")
    if evaluation.get("reference", {}).get("method") != "a20-clone-value-card-reward":
        errors.append("reference method differs")
    paired = dict(evaluation.get("paired_difference") or {})
    final_floor = dict(dict(paired.get("metrics") or {}).get("final_floor") or {})
    try:
        final_floor_mean = float(final_floor["mean_difference"])
        final_floor_ci = tuple(float(value) for value in final_floor["bootstrap_ci95"])
        if len(final_floor_ci) != 2 or not all(math.isfinite(value) for value in final_floor_ci):
            raise ValueError("CI is non-finite")
        candidate_act1 = float(evaluation["candidate"]["summary"]["act1_clear_rate"])
        reference_act1 = float(evaluation["reference"]["summary"]["act1_clear_rate"])
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"effect metrics are malformed: {error}")
        final_floor_mean = 0.0
        final_floor_ci = (0.0, 0.0)
        candidate_act1 = 0.0
        reference_act1 = 0.0
    act1_difference = candidate_act1 - reference_act1
    safety_clear = not any("safety field" in error for error in errors)
    effect_clear = final_floor_ci[0] > 0.0 and act1_difference >= 0.0
    if require_effect and not effect_clear:
        errors.append("effect gate did not pass")
    return {
        "protocol": "a20-map-action-evaluation-gate",
        "schema_version": 1,
        "range_name": expected_range_name,
        "require_effect": require_effect,
        "safety_clear": safety_clear,
        "effect_clear": effect_clear,
        "final_floor": {
            "mean_difference": final_floor_mean,
            "bootstrap_ci95": list(final_floor_ci),
        },
        "act1_clear_mean_difference": act1_difference,
        "passed": not errors,
        "errors": errors,
    }


def _require_range(payload: dict[str, Any], expected_name: str, errors: list[str]) -> None:
    try:
        require_map_action_seed_range(
            expected_name,
            start=int(payload["seed_range"][0]),
            count=int(payload["seed_count"]),
            allowed_names=frozenset({expected_name}),
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"seed range is invalid: {error}")
    if payload.get("seed_range_name") != expected_name:
        errors.append("seed range name differs")


def _require_checkpoint(
    payload: dict[str, Any],
    field: str,
    expected: str,
    errors: list[str],
) -> None:
    actual = str(dict(payload.get(field) or {}).get("sha256") or "")
    if actual != expected:
        errors.append(f"{field} SHA-256 differs")


def _require_trained_acts(
    payload: dict[str, Any],
    expected: frozenset[int] | None,
    errors: list[str],
) -> None:
    if expected is None:
        return
    try:
        actual = frozenset(int(value) for value in payload["map_policy_trained_acts"])
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"map policy trained acts are invalid: {error}")
        return
    if actual != expected:
        errors.append("map policy trained acts differ")


def _require_trained_floor_range(
    payload: dict[str, Any],
    expected: tuple[int, int] | None,
    errors: list[str],
) -> None:
    if expected is None:
        return
    try:
        actual = tuple(int(value) for value in payload["map_policy_trained_floor_range"])
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"map policy trained floor range is invalid: {error}")
        return
    if actual != expected:
        errors.append("map policy trained floor range differs")


def _require_safety(payload: dict[str, Any], errors: list[str]) -> None:
    for role in ("candidate", "reference"):
        summary = dict(payload.get(role, {}).get("summary") or {})
        for field in _SAFETY_FIELDS:
            try:
                value = float(summary[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{role} safety field {field} is invalid")
                continue
            if value != 0.0:
                errors.append(f"{role} safety field {field} is nonzero")


def _require_label_mode(
    payload: dict[str, Any],
    expected: str | None,
    errors: list[str],
) -> None:
    if expected is None:
        return
    actual = payload.get("map_policy_label_mode")
    if actual != expected:
        errors.append("map policy label mode differs")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
