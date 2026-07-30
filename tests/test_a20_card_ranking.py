import unittest

from sts_env.training.a20_card_ranking import (
    A20CardRanker,
    build_card_choice_examples,
    train_card_ranker,
)


def sample_record(selected: str = "PommelStrike") -> dict:
    return {
        "run_id": "card-choice-run",
        "character": "IRONCLAD",
        "ascension_level": 20,
        "is_ascension_mode": True,
        "heart_victory": True,
        "floor_reached": 57,
        "path_per_floor": ["M"] * 57,
        "card_choices": [
            {"floor": 1, "picked": selected, "not_picked": ["Flex", "ShrugItOff"]},
            {"floor": 2, "picked": "SKIP", "not_picked": ["Anger", "Clash"]},
        ],
        "current_hp_per_floor": [75] * 57,
        "max_hp_per_floor": [80] * 57,
        "gold_per_floor": [100] * 57,
        "relics_obtained": [],
        "potions_obtained": [],
        "event_choices": [],
        "items_purchased": [],
        "item_purchase_floors": [],
        "damage_taken": [],
    }


class A20CardRankingTests(unittest.TestCase):
    def test_candidate_set_includes_skip_and_target(self) -> None:
        examples = build_card_choice_examples(sample_record())
        self.assertEqual(len(examples), 2)
        self.assertIn("SKIP", examples[0].candidates)
        self.assertEqual(examples[0].candidates[examples[0].selected_index], "PommelStrike")
        self.assertEqual(examples[1].candidates[examples[1].selected_index], "SKIP")

    def test_selected_card_is_not_part_of_its_state(self) -> None:
        first = build_card_choice_examples(sample_record("PommelStrike"))[0]
        second = build_card_choice_examples(sample_record("Flex"))[0]
        self.assertEqual(first.state_features, second.state_features)

    def test_small_training_is_finite(self) -> None:
        examples = []
        for index in range(100):
            record = sample_record("PommelStrike" if index % 2 else "Flex")
            record["run_id"] = f"run-{index}"
            examples.extend(build_card_choice_examples(record))
        model, metrics = train_card_ranker(examples, epochs=1, batch_size=16)
        self.assertIsInstance(model, A20CardRanker)
        self.assertGreater(metrics["test"]["top1_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
