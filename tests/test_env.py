from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from sts_env import ActionKind, Phase, RunJournal, StsEnv, ToyCombatBackend


class EnvironmentTests(unittest.TestCase):
    def test_reset_is_deterministic_for_a_fixed_seed(self) -> None:
        first = StsEnv(ToyCombatBackend())
        second = StsEnv(ToyCombatBackend())

        first_observation, _ = first.reset(seed=19)
        second_observation, _ = second.reset(seed=19)

        self.assertEqual(first_observation, second_observation)

    def test_public_observation_does_not_expose_rng_or_draw_order(self) -> None:
        env = StsEnv(ToyCombatBackend())
        observation, _ = env.reset(seed=5)
        payload = observation.to_dict()

        self.assertNotIn("seed", payload)
        self.assertNotIn("rng", payload)
        self.assertIsInstance(payload["draw_pile"], list)
        self.assertTrue(all(len(item) == 2 for item in payload["draw_pile"]))

    def test_invalid_action_index_is_rejected(self) -> None:
        env = StsEnv(ToyCombatBackend())
        observation, _ = env.reset(seed=3)

        with self.assertRaises(IndexError):
            env.step(len(observation.legal_actions))

    def test_cloned_environment_replays_the_same_branch(self) -> None:
        env = StsEnv(ToyCombatBackend())
        observation, _ = env.reset(seed=11)
        clone = env.clone()
        action_index = len(observation.legal_actions) - 1

        original_transition = env.step(action_index)
        cloned_transition = clone.step(action_index)

        self.assertEqual(original_transition, cloned_transition)

    def test_public_history_is_derived_from_executed_actions(self) -> None:
        env = StsEnv(ToyCombatBackend())
        observation, _ = env.reset(seed=11)

        self.assertEqual(observation.history.decisions, 0)
        action = observation.legal_actions[-1]
        observation, _, _, _, _ = env.step(action)

        self.assertEqual(observation.history.decisions, 1)
        self.assertEqual(len(observation.history.recent_actions), 1)
        self.assertTrue(
            observation.history.recent_actions[0].startswith(action.kind.value)
        )

    def test_run_journal_recovers_and_continues_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            journal = RunJournal.start(
                StsEnv(ToyCombatBackend()),
                path,
                seed=37,
                durable=False,
            )
            for _ in range(3):
                journal.step(len(journal.environment.observation.legal_actions) - 1)
            expected = journal.environment.clone()
            journal.close()

            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"type":"step"')

            recovered = RunJournal.recover(
                StsEnv(ToyCombatBackend()),
                path,
                durable=False,
            )
            self.assertEqual(recovered.environment.observation, expected.observation)
            action = recovered.environment.observation.legal_actions[-1]
            self.assertEqual(recovered.step(action), expected.step(action))
            recovered.close()

            with path.open("r", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(records), 5)

    def test_enemy_intent_is_applied_on_end_turn(self) -> None:
        env = StsEnv(ToyCombatBackend())
        observation, _ = env.reset(seed=13)

        observation, _, _, _, _ = env.step(len(observation.legal_actions) - 1)
        hp_before_attack = observation.player.hp
        observation, _, _, _, info = env.step(len(observation.legal_actions) - 1)

        self.assertGreater(info["damage_taken"], 0)
        self.assertLess(observation.player.hp, hp_before_attack)

    def test_simple_attack_policy_can_finish_combat(self) -> None:
        env = StsEnv(ToyCombatBackend())
        observation, _ = env.reset(seed=23)

        for _ in range(200):
            attack_indices = [
                index
                for index, action in enumerate(observation.legal_actions)
                if action.kind is ActionKind.PLAY_CARD
                and any(
                    card.instance_id == action.source_id and card.requires_target
                    for card in observation.hand
                )
            ]
            action_index = (
                attack_indices[0] if attack_indices else len(observation.legal_actions) - 1
            )
            observation, reward, terminated, truncated, info = env.step(action_index)
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["terminal_reason"], "combat_won")
        self.assertIs(observation.phase, Phase.TERMINAL)
        self.assertEqual(observation.legal_actions, ())

    def test_action_mask_matches_dynamic_action_count(self) -> None:
        env = StsEnv(ToyCombatBackend())
        observation, _ = env.reset(seed=29)
        mask = env.action_mask(capacity=16)

        self.assertEqual(sum(mask), len(observation.legal_actions))
        self.assertEqual(len(mask), 16)


if __name__ == "__main__":
    unittest.main()
