import unittest

from scripts.format_sts_runs import _heart_evidence, _is_normal_run, normalize_event


class FormatStsRunsTests(unittest.TestCase):
    def test_a20_floor_57_victory_is_heart_win(self):
        event = {
            "victory": True,
            "floor_reached": 57,
            "killed_by": None,
            "ascension_level": 20,
        }
        self.assertEqual(_heart_evidence(event, 57), (True, "victory_and_floor_reached_ge_57"))

    def test_heart_death_is_not_a_win(self):
        event = {"victory": False, "floor_reached": 56, "killed_by": "The Heart"}
        self.assertFalse(_heart_evidence(event, 57)[0])

    def test_act_three_victory_is_not_heart_win(self):
        event = {"victory": True, "floor_reached": 52, "killed_by": None}
        self.assertFalse(_heart_evidence(event, 57)[0])

    def test_beta_run_is_not_normal(self):
        event = {
            "is_ascension_mode": True,
            "is_trial": False,
            "is_daily": False,
            "is_beta": True,
            "is_endless": False,
        }
        self.assertFalse(_is_normal_run(event))

    def test_normalized_record_has_stable_labels(self):
        record = normalize_event(
            {
                "play_id": "run-1",
                "character_chosen": "IRONCLAD",
                "ascension_level": 20,
                "is_ascension_mode": True,
                "victory": True,
                "floor_reached": 57,
            },
            "sample.json",
            57,
        )
        self.assertEqual(record["run_id"], "run-1")
        self.assertEqual(record["character"], "IRONCLAD")
        self.assertTrue(record["heart_victory"])

    def test_normalized_record_preserves_source_field_presence(self):
        record = normalize_event(
            {
                "play_id": "run-2",
                "character_chosen": "IRONCLAD",
                "ascension_level": 20,
                "is_ascension_mode": True,
                "victory": False,
                "floor_reached": 10,
                "card_choices": [],
            },
            "sample.json",
            57,
        )
        self.assertIn("card_choices", record["source_fields_present"])
        self.assertNotIn("event_choices", record["source_fields_present"])


if __name__ == "__main__":
    unittest.main()
