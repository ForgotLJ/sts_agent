from __future__ import annotations

import unittest

from sts_env.training.map_action_stage import (
    map_evaluation_gate,
    map_override_coverage_gate,
    select_profile_margin,
    select_profile_margin_by_coverage,
)


MAP_SHA = "a" * 64
CARD_SHA = "b" * 64


def summary(act1_clear_rate: float) -> dict[str, float]:
    return {
        "act1_clear_rate": act1_clear_rate,
        "errors": 0.0,
        "crashes": 0.0,
        "illegal_actions": 0.0,
        "recovery_failures": 0.0,
        "truncations": 0.0,
        "timeouts": 0.0,
        "cycles": 0.0,
    }


def evaluation(
    range_name: str,
    seed_start: int,
    seed_count: int,
    *,
    record_only: bool = False,
    ci_low: float = 0.1,
) -> dict[str, object]:
    return {
        "protocol": "a20-map-action-value-paired-lightspeed-evaluation",
        "record_only": record_only,
        "seed_range_name": range_name,
        "seed_range": [seed_start, seed_start + seed_count - 1],
        "seed_count": seed_count,
        "map_checkpoint": {"sha256": MAP_SHA},
        "card_checkpoint": {"sha256": CARD_SHA},
        "candidate": {
            "method": "a20-map-action-value",
            "summary": summary(0.20),
        },
        "reference": {
            "method": "a20-clone-value-card-reward",
            "summary": summary(0.10),
        },
        "paired_difference": {
            "metrics": {
                "final_floor": {
                    "mean_difference": 0.5,
                    "bootstrap_ci95": [ci_low, 0.9],
                }
            }
        },
    }


class MapActionStageTests(unittest.TestCase):
    def test_selects_the_frozen_p80_profile_margin(self) -> None:
        profile = evaluation("map_value_profile", 2_318_000, 128, record_only=True)
        profile["candidate_map_telemetry"] = {
            "map_decisions": 100,
            "best_advantage_quantiles": {"p80": 0.024},
        }
        selected = select_profile_margin(
            profile,
            expected_map_checkpoint_sha256=MAP_SHA,
            expected_card_checkpoint_sha256=CARD_SHA,
        )
        self.assertEqual(selected["quantile"], "p80")
        self.assertEqual(selected["override_margin"], 0.024)

    def test_profile_requires_record_only_identity_mode(self) -> None:
        profile = evaluation("map_value_profile", 2_318_000, 128)
        profile["candidate_map_telemetry"] = {
            "map_decisions": 100,
            "best_advantage_quantiles": {"p80": 0.024},
        }
        with self.assertRaisesRegex(ValueError, "record-only"):
            select_profile_margin(
                profile,
                expected_map_checkpoint_sha256=MAP_SHA,
                expected_card_checkpoint_sha256=CARD_SHA,
            )

    def test_profile_requires_the_checkpoint_declared_act_scope(self) -> None:
        profile = evaluation("map_act1_value_profile_v2", 2_328_000, 128, record_only=True)
        profile["candidate_map_telemetry"] = {
            "map_decisions": 100,
            "best_advantage_quantiles": {"p80": 0.024},
        }
        profile["map_policy_trained_acts"] = [1]
        profile["map_policy_trained_floor_range"] = [0, 0]
        selected = select_profile_margin(
            profile,
            expected_map_checkpoint_sha256=MAP_SHA,
            expected_card_checkpoint_sha256=CARD_SHA,
            expected_range_name="map_act1_value_profile_v2",
            expected_trained_acts=frozenset({1}),
            expected_trained_floor_range=(0, 0),
        )
        self.assertEqual(selected["profile_seed_range_name"], "map_act1_value_profile_v2")

    def test_smoke_requires_safety_but_not_effect_confidence(self) -> None:
        smoke = evaluation("map_value_smoke", 2_319_000, 32, ci_low=-0.2)
        result = map_evaluation_gate(
            smoke,
            expected_range_name="map_value_smoke",
            expected_map_checkpoint_sha256=MAP_SHA,
            expected_card_checkpoint_sha256=CARD_SHA,
            require_effect=False,
        )
        self.assertTrue(result["safety_clear"])
        self.assertFalse(result["effect_clear"])
        self.assertTrue(result["passed"])

    def test_formal_requires_positive_interval_and_nonnegative_act1(self) -> None:
        formal = evaluation("map_value_formal", 2_320_000, 512, ci_low=-0.1)
        result = map_evaluation_gate(
            formal,
            expected_range_name="map_value_formal",
            expected_map_checkpoint_sha256=MAP_SHA,
            expected_card_checkpoint_sha256=CARD_SHA,
            require_effect=True,
        )
        self.assertFalse(result["passed"])
        self.assertIn("effect gate did not pass", result["errors"])

    def test_selects_margin_for_a_frozen_override_rate(self) -> None:
        profile = evaluation(
            "map_act1_value_profile_v6",
            2_360_000,
            512,
            record_only=True,
        )
        profile["candidate_map_decision_events"] = [
            {
                "event_type": "scored",
                "predicted_best_advantage": float(index),
                "applied_override": False,
            }
            for index in range(1, 101)
        ]
        selected = select_profile_margin_by_coverage(
            profile,
            expected_map_checkpoint_sha256=MAP_SHA,
            expected_card_checkpoint_sha256=CARD_SHA,
            target_override_rate=0.075,
            minimum_override_rate=0.05,
            maximum_override_rate=0.10,
            expected_range_name="map_act1_value_profile_v6",
        )
        self.assertEqual(selected["profile_override_count"], 7)
        self.assertEqual(selected["profile_override_rate"], 0.07)
        self.assertEqual(selected["override_margin"], 94.0)

    def test_coverage_gate_requires_multiple_interventions(self) -> None:
        smoke = evaluation("map_act1_value_smoke_v6", 2_361_000, 64)
        smoke["candidate_map_telemetry"] = {"map_decisions": 64, "overrides": 4}
        result = map_override_coverage_gate(
            smoke,
            minimum_override_rate=0.03,
            maximum_override_rate=0.15,
            minimum_overrides=2,
        )
        self.assertTrue(result["passed"])
        sparse = map_override_coverage_gate(
            smoke,
            minimum_override_rate=0.03,
            maximum_override_rate=0.15,
            minimum_overrides=5,
        )
        self.assertFalse(sparse["passed"])


if __name__ == "__main__":
    unittest.main()
