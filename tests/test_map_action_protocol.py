from __future__ import annotations

import unittest

from sts_env.training.map_action_protocol import (
    MAP_ACTION_COLLECTION_RANGE_NAMES,
    MAP_ACTION_EVALUATION_RANGE_NAMES,
    map_action_seed_registry,
    require_map_action_seed_range,
)


class MapActionProtocolTests(unittest.TestCase):
    def test_registry_is_non_overlapping(self) -> None:
        registry = map_action_seed_registry()
        self.assertIn("map_act1_pilot", registry)
        self.assertIn("map_value_replication", registry)

    def test_registered_range_requires_exact_bounds_and_purpose(self) -> None:
        selected = require_map_action_seed_range(
            "map_act1_pilot",
            start=2_310_000,
            count=64,
            allowed_names=MAP_ACTION_COLLECTION_RANGE_NAMES,
        )
        self.assertEqual(selected.end, 2_310_063)
        with self.assertRaises(ValueError):
            require_map_action_seed_range(
                "map_value_formal",
                start=2_320_000,
                count=512,
                allowed_names=MAP_ACTION_COLLECTION_RANGE_NAMES,
            )
        with self.assertRaises(ValueError):
            require_map_action_seed_range(
                "map_value_formal",
                start=2_320_001,
                count=512,
                allowed_names=MAP_ACTION_EVALUATION_RANGE_NAMES,
            )


if __name__ == "__main__":
    unittest.main()
