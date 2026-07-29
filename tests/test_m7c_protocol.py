from __future__ import annotations

import unittest

from sts_env.training.m7c_protocol import (
    M7C_EVALUATION_RANGES,
    M7C_DEVELOPMENT_RANGES,
    M7C_HISTORICAL_RANGES,
    M7C_TRAINING_ROUNDS,
    M7C_VALIDATION_RANGES,
    SeedRange,
    m7c_seed_registry,
    require_registered_seed_range,
    validate_seed_ranges,
    m7c_frozen_inputs_identity,
)


class M7CProtocolTests(unittest.TestCase):
    def test_frozen_inputs_are_bound_to_historical_teacher_range(self) -> None:
        identity = m7c_frozen_inputs_identity()
        teacher = dict(identity["teacher_corpus"])
        checkpoint = dict(identity["initial_checkpoint"])
        m6_baseline = dict(identity["m6_baseline_checkpoint"])
        registry = m7c_seed_registry()
        teacher_range = registry["m7b_training_teacher"]
        self.assertEqual(teacher["seed_start"], teacher_range.start)
        self.assertEqual(teacher["seed_count"], teacher_range.count)
        self.assertEqual(len(str(teacher["aggregate_sha256"])), 64)
        self.assertEqual(checkpoint["protocol"], "m7b")
        self.assertEqual(checkpoint["run_seed"], 17)
        self.assertEqual(len(str(checkpoint["sha256"])), 64)
        self.assertEqual(m6_baseline["protocol"], "m6")
        self.assertEqual(m6_baseline["run_seed"], 17)
        self.assertEqual(len(str(m6_baseline["sha256"])), 64)
    def test_registry_is_disjoint_and_preserves_all_locked_ranges(self) -> None:
        registry = m7c_seed_registry()
        self.assertIn("m7_final_blind", registry)
        self.assertTrue(registry["m7_final_blind"].locked)
        self.assertEqual(registry["dagger_round_0"].start, 2_200_000)
        self.assertEqual(registry["formal_gate"].end, 2_221_511)
        self.assertEqual(
            len(registry),
            len(M7C_HISTORICAL_RANGES)
            + len(M7C_TRAINING_ROUNDS)
            + len(M7C_VALIDATION_RANGES)
            + len(M7C_EVALUATION_RANGES)
            + len(M7C_DEVELOPMENT_RANGES),
        )

    def test_registered_range_requires_exact_bounds(self) -> None:
        selected = require_registered_seed_range(
            "on_policy_round_1",
            start=2_212_000,
            count=512,
        )
        self.assertEqual(selected.end, 2_212_511)
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            require_registered_seed_range(
                "on_policy_round_1",
                start=2_212_001,
                count=512,
            )

    def test_range_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_seed_ranges(
                (
                    SeedRange("first", 10, 4, "first"),
                    SeedRange("second", 13, 4, "second"),
                )
            )


if __name__ == "__main__":
    unittest.main()
