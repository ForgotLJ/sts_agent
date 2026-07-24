from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from sts_env import ActionKind, LightspeedBackend, Phase, StsEnv, ToyCombatBackend
from sts_env.search import (
    BeliefConstraints,
    BeliefSearchConfig,
    EnvironmentBeliefSource,
    FixtureBeliefSource,
    ParticleBeliefSearch,
    PolicyValueConfig,
    PolicyValueTrainer,
    PublicHistory,
    SearchTarget,
    SearchTargetBuffer,
    exact_fixture_action_values,
)


class BeliefStateTests(unittest.TestCase):
    def test_public_observation_excludes_episode_seed(self) -> None:
        observation, info = StsEnv(ToyCombatBackend()).reset(seed=123)

        self.assertEqual(info["seed"], 123)
        self.assertNotIn("seed", observation.to_dict())

    def test_history_updates_explicit_draw_constraints(self) -> None:
        environment = StsEnv(ToyCombatBackend())
        observation, _ = environment.reset(seed=7)
        history = PublicHistory(
            observation,
            constraints=BeliefConstraints(known_top=("strike", "defend")),
        )
        next_observation, *_ = environment.step(observation.legal_actions[-1])
        history = history.append(
            observation.legal_actions[-1],
            next_observation,
            drawn_cards=1,
        )

        self.assertEqual(history.constraints.known_top, ("defend",))

    def test_fixture_exact_solver_prefers_defense(self) -> None:
        values = exact_fixture_action_values()
        best_action = max(values, key=values.get)

        self.assertIn("defend", best_action.label)
        self.assertGreater(values[best_action], min(values.values()))

    def test_fixture_belief_samples_preserve_public_root(self) -> None:
        source = FixtureBeliefSource()

        for seed in range(16):
            self.assertEqual(source.sample(seed).observation, source.observation)

    def test_fixture_sampler_is_balanced_over_hidden_orders(self) -> None:
        source = FixtureBeliefSource()
        drawn_counts: dict[str, int] = {}
        defend = next(action for action in source.observation.legal_actions if "defend" in action.label)
        for seed in range(512):
            sampled = source.sample(seed)
            next_observation, *_ = sampled.step(defend)
            drawn = next_observation.hand[0].card_id
            drawn_counts[drawn] = drawn_counts.get(drawn, 0) + 1

        self.assertEqual(set(drawn_counts), {"fixture_finisher", "fixture_guard"})
        self.assertLess(abs(drawn_counts["fixture_finisher"] / 512 - 0.5), 0.08)

    def test_particle_search_matches_fixture_exact_optimum(self) -> None:
        exact_values = exact_fixture_action_values()
        exact_action = max(exact_values, key=exact_values.get)
        result = ParticleBeliefSearch(
            BeliefSearchConfig(simulations=256, max_depth=6),
            seed=19,
        ).search(FixtureBeliefSource())

        self.assertEqual(result.selected_action, exact_action)
        self.assertAlmostEqual(sum(result.policy.values()), 1.0)
        self.assertEqual(result.sampled_worlds, 256)
        self.assertGreaterEqual(result.simulator_calls, 256)
        self.assertTrue(all(stat.outcome_count <= 2 for stat in result.actions))

    def test_particle_search_is_reproducible_for_fixed_seed(self) -> None:
        config = BeliefSearchConfig(simulations=64, max_depth=6)
        first = ParticleBeliefSearch(config, seed=23).search(FixtureBeliefSource())
        second = ParticleBeliefSearch(config, seed=23).search(FixtureBeliefSource())

        self.assertEqual(first, second)

    def test_particle_search_obeys_hard_simulator_call_budget(self) -> None:
        result = ParticleBeliefSearch(
            BeliefSearchConfig(
                simulations=10_000,
                simulator_call_budget=37,
                max_depth=6,
            ),
            seed=27,
        ).search(FixtureBeliefSource())

        self.assertEqual(result.simulator_calls, 37)
        self.assertLessEqual(result.sampled_worlds, 37)
        self.assertLessEqual(result.expanded_nodes, result.simulator_calls + 1)

    def test_policy_value_network_distills_fixture_search(self) -> None:
        source = FixtureBeliefSource()
        buffer = SearchTargetBuffer(capacity=64, seed=29)
        for seed in range(64):
            result = ParticleBeliefSearch(
                BeliefSearchConfig(simulations=64, max_depth=6),
                seed=seed,
            ).search(source)
            buffer.add(SearchTarget.from_search_result(source.observation, result))
        trainer = PolicyValueTrainer(
            PolicyValueConfig(
                policy_hidden_sizes=(32, 16),
                value_hidden_sizes=(32, 16),
                learning_rate=3e-3,
                batch_size=16,
            ),
            seed=31,
        )
        exact_action = max(exact_fixture_action_values(), key=exact_fixture_action_values().get)
        before_probability = trainer.policy(source.observation)[
            source.observation.legal_actions.index(exact_action)
        ]
        trainer.train_from_buffer(buffer, updates=200)
        after_probability = trainer.policy(source.observation)[
            source.observation.legal_actions.index(exact_action)
        ]

        self.assertEqual(trainer.greedy_action(source.observation), exact_action)
        self.assertGreater(after_probability, before_probability)
        self.assertGreater(after_probability, 0.8)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy-value.pt"
            trainer.save(path)
            restored = PolicyValueTrainer.load(path)
        self.assertEqual(restored.policy(source.observation), trainer.policy(source.observation))
        self.assertEqual(restored.value(source.observation), trainer.value(source.observation))

    def test_lightspeed_redeterminization_preserves_public_combat_state(self) -> None:
        environment = StsEnv(LightspeedBackend(neow_history="skipped"))
        observation, _ = environment.reset(seed=1)
        observation, *_ = environment.step(observation.legal_actions[0])
        self.assertIs(observation.phase, Phase.COMBAT)
        source = EnvironmentBeliefSource(environment)

        for seed in range(8):
            self.assertEqual(source.sample(seed).observation, observation)

    def test_lightspeed_redeterminization_changes_unseen_future(self) -> None:
        environment = StsEnv(LightspeedBackend(neow_history="skipped"))
        observation, _ = environment.reset(seed=1)
        observation, *_ = environment.step(observation.legal_actions[0])
        source = EnvironmentBeliefSource(environment)

        future_hands: set[tuple[str, ...]] = set()
        for seed in range(16):
            sampled = source.sample(seed)
            for _ in range(2):
                end_turn = next(
                    action
                    for action in sampled.observation.legal_actions
                    if action.kind is ActionKind.END_TURN
                )
                sampled.step(end_turn)
            future_hands.add(tuple(card.card_id for card in sampled.observation.hand))

        self.assertGreater(len(future_hands), 1)

    def test_lightspeed_redeterminization_preserves_noncombat_public_state(self) -> None:
        environment = StsEnv(LightspeedBackend(neow_history="skipped"))
        observation, _ = environment.reset(seed=1)

        sampled = EnvironmentBeliefSource(environment).sample(0)

        self.assertEqual(sampled.observation, observation)


if __name__ == "__main__":
    unittest.main()
