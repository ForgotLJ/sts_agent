import unittest

from sts_env.training.a20_coldstart import (
    A20ColdStartConfig,
    A20ValueNetwork,
    build_prefix_examples,
    encode_prefix,
    train_value_model,
)


def sample_record(character: str = "IRONCLAD") -> dict:
    return {
        "run_id": f"run-{character}",
        "character": character,
        "ascension_level": 20,
        "is_ascension_mode": True,
        "victory": True,
        "heart_victory": True,
        "floor_reached": 57,
        "path_per_floor": ["M", "E", "R", "?", "$", "M"] * 10,
        "card_choices": [
            {"floor": 1, "picked": "PommelStrike", "not_picked": ["Flex", "ShrugItOff"]}
        ],
        "relics_obtained": [{"floor": 5, "key": "Vajra"}],
        "potions_obtained": [{"floor": 4, "key": "Strength Potion"}],
        "event_choices": [{"floor": 3, "event_name": "Golden Idol"}],
        "items_purchased": ["Potion Belt"],
        "item_purchase_floors": [6],
        "current_hp_per_floor": [70] * 60,
        "max_hp_per_floor": [80] * 60,
        "gold_per_floor": [100] * 60,
        "damage_taken": [{"floor": 2, "damage": 5}],
    }


class A20ColdStartTests(unittest.TestCase):
    def test_feature_dimension_and_prefix_avoid_final_deck(self) -> None:
        config = A20ColdStartConfig()
        features = encode_prefix(sample_record(), 10, config)
        self.assertEqual(len(features), config.feature_dimension)

    def test_examples_keep_character_specific_records(self) -> None:
        examples = build_prefix_examples(sample_record("WATCHER"))
        self.assertEqual(len(examples), 5)
        self.assertEqual(examples[0].heart_win, 1.0)

    def test_examples_do_not_create_states_after_a_run_ends(self) -> None:
        record = sample_record()
        record["floor_reached"] = 21
        self.assertEqual([example.prefix_floor for example in build_prefix_examples(record)], [10, 20])

    def test_small_value_model_training_is_finite(self) -> None:
        examples = []
        for index in range(100):
            record = sample_record("IRONCLAD")
            record["run_id"] = f"run-{index}"
            record["heart_victory"] = index % 2 == 0
            examples.extend(build_prefix_examples(record))
        model, metrics = train_value_model(examples, epochs=1, batch_size=8)
        self.assertIsInstance(model, A20ValueNetwork)
        self.assertTrue(metrics["test"]["examples"] > 0)
        self.assertIn("10", metrics["test"]["by_prefix_floor"])


if __name__ == "__main__":
    unittest.main()
