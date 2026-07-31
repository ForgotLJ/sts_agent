from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from sts_env.training.a20_online_card_ranking import (
    A20OnlineCardRanker,
    A20OnlineCardRankingConfig,
    A20OnlineCardRewardPolicy,
    A20CloneValueCardRewardPolicy,
    A20OnlineValueNetwork,
    SKIP_CARD,
    build_online_card_choice_examples,
    build_online_value_examples,
    canonical_card_id,
    card_reward_candidate_id,
    encode_online_observation,
    encode_run_summary_card_state,
    train_online_value_model,
    train_online_card_ranker,
)
from sts_env.types import Action, ActionKind, Observation, Phase, PlayerView


def sample_record(selected: str = "Pommel Strike") -> dict:
    return {
        "run_id": "online-card-choice-run",
        "character": "IRONCLAD",
        "ascension_level": 20,
        "heart_victory": True,
        "floor_reached": 57,
        "card_choices": [
            {"floor": 1, "picked": selected, "not_picked": ["Flex", "Shrug It Off"]},
            {"floor": 2, "picked": "SKIP", "not_picked": ["Anger", "Clash"]},
        ],
        "current_hp_per_floor": [75] * 57,
        "max_hp_per_floor": [80] * 57,
        "gold_per_floor": [100] * 57,
        "relics_obtained": [],
        "potions_obtained": [],
        "event_choices": [],
    }


def card_reward_observation(*actions: Action, phase: Phase = Phase.CARD_REWARD) -> Observation:
    return Observation(
        phase=phase,
        turn=0,
        player=PlayerView(hp=75, max_hp=80, block=0, energy=0, gold=100),
        hand=(),
        enemies=(),
        draw_pile=(),
        discard_pile=(),
        exhaust_pile=(),
        legal_actions=actions,
        ascension=20,
        act=1,
        floor=1,
        deck=(("Strike_R", 5), ("Defend_R", 4), ("Bash", 1)),
        relics=(("Burning Blood", 1),),
    )


class A20OnlineCardRankingTests(unittest.TestCase):
    def test_live_and_summary_state_contract_match_at_first_reward(self) -> None:
        record = sample_record()
        observation = card_reward_observation()
        self.assertEqual(
            encode_run_summary_card_state(record, 1),
            encode_online_observation(observation),
        )

    def test_candidate_ids_match_lightspeed_card_and_skip_actions(self) -> None:
        card = Action(ActionKind.CHOOSE_CARD, source_id="PommelStrike", label="Pommel Strike")
        skip = Action(ActionKind.LEAVE, source_id="skip_card", option_type="skip_card")
        self.assertEqual(canonical_card_id("Pommel Strike+2"), "pommelstrike")
        self.assertEqual(card_reward_candidate_id(card), "pommelstrike")
        self.assertEqual(card_reward_candidate_id(skip), SKIP_CARD)

    def test_current_choice_does_not_change_its_state_features(self) -> None:
        first = build_online_card_choice_examples(sample_record("Pommel Strike"))[0]
        second = build_online_card_choice_examples(sample_record("Flex"))[0]
        self.assertEqual(first.state_features, second.state_features)

    def test_policy_selects_only_a_live_legal_card_reward_action(self) -> None:
        config = A20OnlineCardRankingConfig(
            card_buckets=8,
            relic_buckets=8,
            potion_buckets=8,
            candidate_buckets=32,
            hidden_dimension=8,
        )
        model = A20OnlineCardRanker(config.feature_dimension, config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.candidate_bias.weight[10, 0] = 5.0
        card_a = Action(ActionKind.CHOOSE_CARD, source_id="pommelstrike", label="Pommel Strike")
        card_b = Action(ActionKind.CHOOSE_CARD, source_id="flex", label="Flex")
        skip = Action(ActionKind.LEAVE, source_id="skip_card", option_type="skip_card")
        actions = (card_a, card_b, skip)
        candidate_ids = [
            int.from_bytes(
                __import__("hashlib").blake2b(card_reward_candidate_id(action).encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            % config.candidate_buckets
            for action in actions
        ]
        target_index = candidate_ids.index(10) if 10 in candidate_ids else 0
        with torch.no_grad():
            model.candidate_bias.weight.zero_()
            model.candidate_bias.weight[candidate_ids[target_index], 0] = 5.0
        selected = A20OnlineCardRewardPolicy(model)(card_reward_observation(*actions))
        self.assertIn(selected, actions)
        self.assertEqual(selected, actions[target_index])

    def test_small_training_is_finite(self) -> None:
        examples = []
        for index in range(100):
            record = sample_record("Pommel Strike" if index % 2 else "Flex")
            record["run_id"] = f"online-run-{index}"
            examples.extend(build_online_card_choice_examples(record))
        model, metrics = train_online_card_ranker(examples, epochs=1, batch_size=16)
        self.assertIsInstance(model, A20OnlineCardRanker)
        self.assertGreater(metrics["test"]["top1_accuracy"], 0.0)

    def test_value_examples_use_the_same_state_contract(self) -> None:
        examples = build_online_value_examples(sample_record())
        self.assertEqual(len(examples), 2)
        self.assertEqual(len(examples[0].state_features), A20OnlineCardRankingConfig().feature_dimension)
        self.assertEqual(examples[0].heart_win, 1.0)

    def test_value_example_matches_live_post_choice_state(self) -> None:
        record = sample_record("Pommel Strike")
        post_choice = replace(
            card_reward_observation(),
            deck=(
                ("Strike_R", 5),
                ("Defend_R", 4),
                ("Bash", 1),
                ("PommelStrike", 1),
            ),
        )
        self.assertEqual(
            build_online_value_examples(record)[0].state_features,
            encode_online_observation(post_choice),
        )

    def test_small_value_model_training_is_finite(self) -> None:
        examples = []
        for index in range(100):
            record = sample_record()
            record["run_id"] = f"value-run-{index}"
            record["heart_victory"] = index % 2 == 0
            record["victory"] = index % 3 != 0
            record["floor_reached"] = 57 - index % 5
            examples.extend(build_online_value_examples(record))
        model, metrics = train_online_value_model(examples, epochs=1, batch_size=16)
        self.assertIsInstance(model, A20OnlineValueNetwork)
        self.assertGreater(metrics["test"]["examples"], 0.0)

    def test_clone_value_policy_falls_back_when_clone_is_unavailable(self) -> None:
        class BrokenCloneEnvironment:
            observation = card_reward_observation(
                Action(ActionKind.CHOOSE_CARD, source_id="pommelstrike", label="Pommel Strike"),
                Action(ActionKind.LEAVE, source_id="skip_card", option_type="skip_card"),
            )

            def clone(self):
                raise RuntimeError("clone unavailable")

        model = A20OnlineValueNetwork(A20OnlineCardRankingConfig().feature_dimension, 8)
        policy = A20CloneValueCardRewardPolicy(model)
        selected = policy.select(BrokenCloneEnvironment())
        self.assertIn(selected, BrokenCloneEnvironment.observation.legal_actions)


if __name__ == "__main__":
    unittest.main()
