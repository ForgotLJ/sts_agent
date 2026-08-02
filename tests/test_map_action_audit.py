from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from sts_env.training.map_action_audit import audit_map_policy_evaluations


MAP_SHA = "a" * 64
CARD_SHA = "b" * 64


def episode(seed: int, floor: int) -> dict[str, object]:
    return {
        "seed": seed,
        "outcome": "terminal",
        "won": False,
        "final_act": 1,
        "final_floor": floor,
        "final_hp": 40,
        "max_hp": 80,
        "environment_return": 0.0,
        "proxy_score": floor / 60.0,
        "decisions": 20,
        "simulator_calls": 10,
        "wall_seconds": 1.0,
        "error": "",
        "error_category": "",
    }


def evaluation(range_name: str, seed_start: int, candidate_gain: int) -> dict[str, object]:
    seeds = [seed_start + index for index in range(512)]
    candidate_episodes = [episode(seed, 10 + candidate_gain) for seed in seeds]
    reference_episodes = [episode(seed, 10) for seed in seeds]

    def summary(episodes: list[dict[str, object]], act1_clear_rate: float) -> dict[str, object]:
        return {
            "episodes": episodes,
            "act1_clear_rate": act1_clear_rate,
            "errors": 0,
            "crashes": 0,
            "illegal_actions": 0,
            "recovery_failures": 0,
            "truncations": 0,
            "timeouts": 0,
            "cycles": 0,
        }

    return {
        "protocol": "a20-map-action-value-paired-lightspeed-evaluation",
        "schema_version": 1,
        "record_only": False,
        "seed_range": [seed_start, seed_start + 511],
        "seed_range_name": range_name,
        "seed_count": 512,
        "map_checkpoint": {"sha256": MAP_SHA},
        "card_checkpoint": {"sha256": CARD_SHA},
        "map_policy_trained_acts": [1],
        "candidate": {
            "method": "a20-map-action-value",
            "summary": summary(candidate_episodes, 0.25),
        },
        "reference": {
            "method": "a20-clone-value-card-reward",
            "summary": summary(reference_episodes, 0.0),
        },
        "paired_difference": {
            "metrics": {
                "final_floor": {
                    "mean_difference": float(candidate_gain),
                    "bootstrap_ci95": [0.5, 1.5],
                }
            }
        },
    }


class MapActionAuditTests(unittest.TestCase):
    def test_audit_accepts_two_disjoint_improved_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal.json"
            replication = root / "replication.json"
            formal.write_text(json.dumps(evaluation("map_value_formal", 2_320_000, 1)), encoding="utf-8")
            replication.write_text(
                json.dumps(evaluation("map_value_replication", 2_321_000, 1)),
                encoding="utf-8",
            )
            result = audit_map_policy_evaluations(
                formal,
                replication,
                expected_map_checkpoint_sha256=MAP_SHA,
                expected_card_checkpoint_sha256=CARD_SHA,
                bootstrap_samples=32,
            )
        self.assertEqual(result["verdict"], "replicated_improved")
        self.assertEqual(result["pooled"]["sample_count"], 1024)

    def test_audit_rejects_a_nonpositive_formal_interval(self) -> None:
        formal_payload = evaluation("map_value_formal", 2_320_000, 1)
        formal_payload["paired_difference"]["metrics"]["final_floor"]["bootstrap_ci95"] = [-0.1, 1.5]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal.json"
            replication = root / "replication.json"
            formal.write_text(json.dumps(formal_payload), encoding="utf-8")
            replication.write_text(
                json.dumps(evaluation("map_value_replication", 2_321_000, 1)),
                encoding="utf-8",
            )
            result = audit_map_policy_evaluations(formal, replication)
        self.assertEqual(result["verdict"], "FAIL")

    def test_audit_rejects_overlapping_episode_seeds(self) -> None:
        replication_payload = evaluation("map_value_replication", 2_320_000, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal.json"
            replication = root / "replication.json"
            formal.write_text(json.dumps(evaluation("map_value_formal", 2_320_000, 1)), encoding="utf-8")
            replication.write_text(json.dumps(replication_payload), encoding="utf-8")
            result = audit_map_policy_evaluations(formal, replication)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("seed sets overlap", " ".join(result["errors"]))

    def test_audit_accepts_the_act1_v2_ranges_and_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal.json"
            replication = root / "replication.json"
            formal.write_text(
                json.dumps(evaluation("map_act1_value_formal_v2", 2_330_000, 1)),
                encoding="utf-8",
            )
            replication.write_text(
                json.dumps(evaluation("map_act1_value_replication_v2", 2_331_000, 1)),
                encoding="utf-8",
            )
            result = audit_map_policy_evaluations(
                formal,
                replication,
                expected_map_checkpoint_sha256=MAP_SHA,
                expected_card_checkpoint_sha256=CARD_SHA,
                expected_formal_range_name="map_act1_value_formal_v2",
                expected_replication_range_name="map_act1_value_replication_v2",
                expected_trained_acts=frozenset({1}),
                bootstrap_samples=32,
            )
        self.assertEqual(result["verdict"], "replicated_improved")


if __name__ == "__main__":
    unittest.main()
