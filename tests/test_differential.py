from __future__ import annotations

from dataclasses import replace
import unittest

from sts_env import LightspeedBackend, Phase, StsEnv
from sts_env.differential import AllowlistEntry, DifferentialAllowlist, compare_observations


class DifferentialTests(unittest.TestCase):
    def test_equal_observations_have_no_differences(self) -> None:
        first = StsEnv(LightspeedBackend())
        second = StsEnv(LightspeedBackend())
        first_observation, _ = first.reset(seed=107)
        second_observation, _ = second.reset(seed=107)

        self.assertEqual(compare_observations(first_observation, second_observation), ())

    def test_allowlist_marks_but_does_not_delete_differences(self) -> None:
        first = StsEnv(LightspeedBackend())
        second = StsEnv(LightspeedBackend())
        first_observation, _ = first.reset(seed=109)
        second_observation, _ = second.reset(seed=109)
        second_observation = replace(second_observation, floor=second_observation.floor + 1)
        allowlist = DifferentialAllowlist((AllowlistEntry("run.floor", "fixture"),))

        differences = compare_observations(first_observation, second_observation, allowlist)

        self.assertEqual(len(differences), 1)
        self.assertTrue(differences[0].allowed)
        self.assertEqual(differences[0].reason, "fixture")

    def test_terminal_comparison_ignores_combat_only_player_statuses(self) -> None:
        env = StsEnv(LightspeedBackend())
        observation, _ = env.reset(seed=113)
        reference = replace(
            observation,
            phase=Phase.TERMINAL,
            player=replace(observation.player, statuses=(("CONFUSED", 1),)),
        )
        candidate = replace(observation, phase=Phase.TERMINAL)

        self.assertEqual(compare_observations(reference, candidate), ())


if __name__ == "__main__":
    unittest.main()
