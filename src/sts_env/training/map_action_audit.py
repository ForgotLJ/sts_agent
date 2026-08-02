from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sts_env.training.m6_reporting import paired_evaluation_difference
from sts_env.training.map_action_protocol import require_map_action_seed_range


_SAFETY_FIELDS = (
    "errors",
    "crashes",
    "illegal_actions",
    "recovery_failures",
    "truncations",
    "timeouts",
    "cycles",
)


def audit_map_policy_evaluations(
    formal_path: str | Path,
    replication_path: str | Path,
    *,
    expected_map_checkpoint_sha256: str | None = None,
    expected_card_checkpoint_sha256: str | None = None,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    formal = _read_json(formal_path)
    replication = _read_json(replication_path)
    errors: list[str] = []
    required_protocol = "a20-map-action-value-paired-lightspeed-evaluation"
    for name, payload in (("formal", formal), ("replication", replication)):
        if payload.get("protocol") != required_protocol:
            errors.append(f"{name}: protocol differs")
        if payload.get("record_only"):
            errors.append(f"{name}: record-only result cannot be promoted")
        if payload.get("seed_count") != 512:
            errors.append(f"{name}: seed count is not 512")
        expected_range = "map_value_formal" if name == "formal" else "map_value_replication"
        try:
            require_map_action_seed_range(
                expected_range,
                start=int(payload["seed_range"][0]),
                count=int(payload["seed_count"]),
                allowed_names=frozenset({expected_range}),
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{name}: seed range is not frozen: {error}")
        if payload.get("seed_range_name") != expected_range:
            errors.append(f"{name}: seed range name differs")
        _check_checkpoint(
            errors,
            payload,
            name=name,
            field="map_checkpoint",
            expected=expected_map_checkpoint_sha256,
        )
        _check_checkpoint(
            errors,
            payload,
            name=name,
            field="card_checkpoint",
            expected=expected_card_checkpoint_sha256,
        )
        for role in ("candidate", "reference"):
            summary = dict(payload.get(role, {}).get("summary") or {})
            for field in _SAFETY_FIELDS:
                if float(summary.get(field, 1.0)) != 0.0:
                    errors.append(f"{name} {role}: safety field {field} is nonzero")
        if payload.get("candidate", {}).get("method") != "a20-map-action-value":
            errors.append(f"{name}: candidate method differs")
        if payload.get("reference", {}).get("method") != "a20-clone-value-card-reward":
            errors.append(f"{name}: reference method differs")
    formal_seeds = _episode_seeds(formal, "formal")
    replication_seeds = _episode_seeds(replication, "replication")
    if formal_seeds & replication_seeds:
        errors.append("formal and replication seed sets overlap")
    formal_result = _single_result(formal, "formal", errors)
    replication_result = _single_result(replication, "replication", errors)
    pooled_result: dict[str, Any] | None = None
    if not errors:
        pooled_candidate = _combined_evaluation(formal, replication, "candidate")
        pooled_reference = _combined_evaluation(formal, replication, "reference")
        pooled_difference = paired_evaluation_difference(
            pooled_candidate,
            pooled_reference,
            bootstrap_samples=bootstrap_samples,
            seed=17,
        )
        pooled_result = {
            "sample_count": pooled_difference["sample_count"],
            "seed_range": pooled_difference["seed_range"],
            "final_floor": pooled_difference["metrics"]["final_floor"],
            "act1_clear_mean_difference": (
                float(pooled_candidate["summary"]["act1_clear_rate"])
                - float(pooled_reference["summary"]["act1_clear_rate"])
            ),
        }
    formal_pass = _passes(formal_result)
    replication_pass = _passes(replication_result)
    verdict = "replicated_improved" if not errors and formal_pass and replication_pass else "FAIL"
    return {
        "protocol": "a20-map-action-value-audit",
        "schema_version": 1,
        "formal": formal_result,
        "replication": replication_result,
        "pooled": pooled_result,
        "errors": errors,
        "verdict": verdict,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_checkpoint(
    errors: list[str],
    payload: dict[str, Any],
    *,
    name: str,
    field: str,
    expected: str | None,
) -> None:
    checkpoint = dict(payload.get(field) or {})
    digest = str(checkpoint.get("sha256") or "")
    if len(digest) != 64:
        errors.append(f"{name}: {field} SHA-256 is missing or malformed")
    if expected is not None and digest != expected:
        errors.append(f"{name}: {field} SHA-256 differs from expected")


def _episode_seeds(payload: dict[str, Any], name: str) -> set[int]:
    episodes = payload.get("candidate", {}).get("summary", {}).get("episodes") or []
    seeds = {int(episode["seed"]) for episode in episodes}
    if len(seeds) != len(episodes):
        raise ValueError(f"{name}: candidate episode seeds are duplicated")
    return seeds


def _single_result(
    payload: dict[str, Any],
    name: str,
    errors: list[str],
) -> dict[str, Any]:
    paired = dict(payload.get("paired_difference") or {})
    metrics = dict(paired.get("metrics") or {})
    final_floor = dict(metrics.get("final_floor") or {})
    candidate_summary = dict(payload.get("candidate", {}).get("summary") or {})
    reference_summary = dict(payload.get("reference", {}).get("summary") or {})
    try:
        final_floor_mean = float(final_floor["mean_difference"])
        final_floor_ci = tuple(float(value) for value in final_floor["bootstrap_ci95"])
        act1_difference = float(candidate_summary["act1_clear_rate"]) - float(
            reference_summary["act1_clear_rate"]
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"{name}: paired metrics are malformed: {error}")
        final_floor_mean = 0.0
        final_floor_ci = (0.0, 0.0)
        act1_difference = 0.0
    return {
        "seed_range_name": payload.get("seed_range_name"),
        "seed_range": payload.get("seed_range"),
        "sample_count": payload.get("seed_count"),
        "final_floor": {
            "mean_difference": final_floor_mean,
            "bootstrap_ci95": list(final_floor_ci),
        },
        "act1_clear_mean_difference": act1_difference,
        "safety_clear": all(
            float(dict(payload.get(role, {}).get("summary") or {}).get(field, 1.0)) == 0.0
            for role in ("candidate", "reference")
            for field in _SAFETY_FIELDS
        ),
    }


def _passes(result: dict[str, Any]) -> bool:
    ci = result["final_floor"]["bootstrap_ci95"]
    return bool(
        result["safety_clear"]
        and float(ci[0]) > 0.0
        and float(result["act1_clear_mean_difference"]) >= 0.0
    )


def _combined_evaluation(
    formal: dict[str, Any],
    replication: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    episodes = (
        list(formal[role]["summary"]["episodes"])
        + list(replication[role]["summary"]["episodes"])
    )
    return {
        "summary": {
            "episodes": episodes,
            "act1_clear_rate": (
                float(formal[role]["summary"]["act1_clear_rate"])
                * int(formal["seed_count"])
                + float(replication[role]["summary"]["act1_clear_rate"])
                * int(replication["seed_count"])
            )
            / (int(formal["seed_count"]) + int(replication["seed_count"])),
        }
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
