from __future__ import annotations

import json
import random
from pathlib import Path
import tempfile
import unittest

from sts_env import (
    Action,
    ActionKind,
    EpisodeTrace,
    LightspeedBackend,
    Observation,
    Phase,
    StsEnv,
    record_episode,
    replay_trace,
)


class FakeRewardBridge:
    def __init__(self) -> None:
        self.steps: list[int] = []
        self.skipped_card_rewards: list[int] = []
        self.state = {
            "phase": "card_reward",
            "screen_state": "rewards",
            "outcome": "undecided",
            "terminated": False,
            "seed": 1,
            "ascension": 0,
            "act": 1,
            "floor": 1,
            "map_x": 1,
            "map_y": 0,
            "turn": 0,
            "deck": [("Strike_R", 5), ("Defend_R", 4), ("Bash", 1)],
            "relics": [("BurningBlood", 0)],
            "potions": [(0, "Potion Slot"), (1, "Potion Slot"), (2, "Potion Slot")],
            "player": {
                "hp": 74,
                "max_hp": 80,
                "block": 0,
                "energy": 0,
                "gold": 99,
                "statuses": [],
            },
            "hand": [],
            "enemies": [],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
        }
        self.descriptors = [
            {
                "domain": "game",
                "token": 101,
                "idx1": 0,
                "idx2": 0,
                "idx3": 0,
                "label": "",
                "screen_state": "rewards",
                "reward_type": 1,
            },
            {
                "domain": "game",
                "token": 201,
                "idx1": 0,
                "idx2": 0,
                "idx3": 0,
                "label": "",
                "screen_state": "rewards",
                "reward_type": 0,
                "card_id": "BodySlam",
                "card_name": "Body Slam",
            },
            {
                "domain": "game",
                "token": 202,
                "idx1": 0,
                "idx2": 1,
                "idx3": 0,
                "label": "",
                "screen_state": "rewards",
                "reward_type": 0,
                "card_id": "Clothesline",
                "card_name": "Clothesline",
            },
            {
                "domain": "game",
                "token": 301,
                "idx1": 0,
                "idx2": 0,
                "idx3": 0,
                "label": "",
                "screen_state": "rewards",
                "reward_type": 6,
            },
        ]

    def observe(self) -> dict[str, object]:
        return self.state

    def legal_actions(self) -> list[dict[str, object]]:
        return self.descriptors

    def step(self, token: int) -> None:
        self.steps.append(token)

    def skip_card_reward(self, reward_index: int) -> None:
        self.skipped_card_rewards.append(reward_index)
        self.descriptors = [
            descriptor
            for descriptor in self.descriptors
            if not (
                int(descriptor.get("reward_type", -1)) == 0
                and int(descriptor.get("idx1", -1)) == reward_index
            )
        ]


class FakeShopBridge(FakeRewardBridge):
    def __init__(self) -> None:
        super().__init__()
        self.state["phase"] = "shop"
        self.state["screen_state"] = "shop"
        self.descriptors = [
            {
                "domain": "game",
                "token": 401,
                "idx1": 0,
                "idx2": 0,
                "idx3": 0,
                "label": "",
                "screen_state": "shop",
                "reward_type": 0,
            },
            {
                "domain": "game",
                "token": 402,
                "idx1": 0,
                "idx2": 0,
                "idx3": 0,
                "label": "",
                "screen_state": "shop",
                "reward_type": 5,
            },
            {
                "domain": "game",
                "token": 403,
                "idx1": 0,
                "idx2": 0,
                "idx3": 0,
                "label": "",
                "screen_state": "shop",
                "reward_type": 6,
            },
        ]


class FakeDrawPileSelectionBridge(FakeRewardBridge):
    def __init__(self, descriptors: list[dict[str, object]]) -> None:
        super().__init__()
        self.state["phase"] = "combat"
        self.state["screen_state"] = "combat"
        self.state["turn"] = 1
        self.descriptors = descriptors


class LightspeedBackendTests(unittest.TestCase):
    def test_neow_history_modes_are_explicit(self) -> None:
        full, full_info = LightspeedBackend(neow_history="full").reset(seed=0)
        limited, limited_info = LightspeedBackend(neow_history="limited").reset(seed=0)
        skipped, skipped_info = LightspeedBackend(neow_history="skipped").reset(seed=0)

        self.assertIs(full.phase, Phase.EVENT)
        self.assertEqual(len(full.legal_actions), 4)
        self.assertTrue(all(action.source_id.startswith("neow") for action in full.legal_actions))
        self.assertTrue(all(action.description for action in full.legal_actions))
        self.assertTrue(all("Choose option" not in action.label for action in full.legal_actions))
        self.assertIs(limited.phase, Phase.EVENT)
        self.assertEqual(len(limited.legal_actions), 2)
        self.assertIs(skipped.phase, Phase.MAP)
        self.assertEqual(full_info["neow_history"], "full")
        self.assertEqual(limited_info["neow_history"], "limited")
        self.assertEqual(skipped_info["neow_history"], "skipped")

        lament_backend = LightspeedBackend(neow_history="limited")
        lament, _ = lament_backend.reset(seed=0)
        lament = lament_backend.step(lament.legal_actions[0]).observation
        self.assertIs(lament.phase, Phase.MAP)
        self.assertIn(("NeowsBlessing", 3), lament.relics)

        max_hp_backend = LightspeedBackend(neow_history="limited")
        max_hp, _ = max_hp_backend.reset(seed=0)
        max_hp = max_hp_backend.step(max_hp.legal_actions[1]).observation
        self.assertIs(max_hp.phase, Phase.MAP)
        self.assertEqual(max_hp.player.max_hp, 88)

    def test_combat_turns_are_one_based(self) -> None:
        backend = LightspeedBackend(neow_history="skipped")
        observation, _ = backend.reset(seed=1)
        self.assertTrue(
            all(
                action.source_id == f"x{action.choice_index}"
                for action in observation.legal_actions
                if action.kind is ActionKind.CHOOSE_MAP_NODE
            )
        )
        observation = backend.step(observation.legal_actions[0]).observation

        self.assertIs(observation.phase, Phase.COMBAT)
        self.assertEqual(observation.turn, 1)

    def test_full_public_run_state_is_exported(self) -> None:
        observation, _ = LightspeedBackend(neow_history="skipped").reset(seed=1)

        self.assertEqual(observation.act, 1)
        self.assertTrue(observation.act_boss)
        self.assertEqual(observation.potion_capacity, 3)
        self.assertEqual(
            (observation.ruby_key, observation.emerald_key, observation.sapphire_key),
            (False, False, False),
        )
        self.assertGreater(len(observation.map_nodes), 40)
        self.assertTrue(all(node.children for node in observation.map_nodes))
        map_actions = [
            action
            for action in observation.legal_actions
            if action.kind is ActionKind.CHOOSE_MAP_NODE
        ]
        self.assertTrue(map_actions)
        self.assertTrue(all(action.target_y == 0 for action in map_actions))
        self.assertTrue(all(action.option_type for action in map_actions))

    def test_card_rewards_are_exposed_as_two_stage_actions(self) -> None:
        backend = LightspeedBackend()
        bridge = FakeRewardBridge()
        backend._bridge = bridge
        backend._seed = 1
        backend._pending_card_reward = None
        observation = backend._read_observation()

        reward_choices = [
            (action.kind, action.source_id, action.choice_index)
            for action in observation.legal_actions
        ]
        self.assertIn((ActionKind.CHOOSE_OPTION, "gold", 0), reward_choices)
        self.assertIn((ActionKind.CHOOSE_OPTION, "card", 1), reward_choices)
        self.assertIn((ActionKind.CHOOSE_OPTION, "proceed", None), reward_choices)
        self.assertNotIn(ActionKind.CHOOSE_CARD, {action.kind for action in observation.legal_actions})

        card_entry = next(
            action for action in observation.legal_actions if action.source_id == "card"
        )
        card_screen = backend.step(card_entry).observation

        self.assertEqual(bridge.steps, [])
        self.assertEqual(card_screen.screen_state, "card_reward")
        self.assertEqual(
            [
                (action.kind, action.source_id, action.choice_index)
                for action in card_screen.legal_actions
            ],
            [
                (ActionKind.CHOOSE_CARD, "bodyslam", 0),
                (ActionKind.CHOOSE_CARD, "clothesline", 1),
                (ActionKind.LEAVE, "skip_card", None),
            ],
        )

        clothesline = next(
            action for action in card_screen.legal_actions if action.source_id == "clothesline"
        )
        backend.step(clothesline)

        self.assertEqual(bridge.steps, [202])

        skip_backend = LightspeedBackend()
        skip_bridge = FakeRewardBridge()
        skip_backend._bridge = skip_bridge
        skip_backend._seed = 1
        skip_backend._pending_card_reward = None
        skip_screen = skip_backend._read_observation()
        card_entry = next(
            action for action in skip_screen.legal_actions if action.source_id == "card"
        )
        skip_screen = skip_backend.step(card_entry).observation
        skip_action = next(
            action for action in skip_screen.legal_actions if action.source_id == "skip_card"
        )
        rewards = skip_backend.step(skip_action).observation

        self.assertEqual(skip_bridge.skipped_card_rewards, [0])
        self.assertNotIn("card", {action.source_id for action in rewards.legal_actions})

    def test_draw_pile_selection_actions_use_stable_card_identity(self) -> None:
        descriptors = [
            {
                "domain": "combat",
                "token": 31,
                "action_type": 2,
                "choice_index": 3,
                "label": "{ SECRET_TECHNIQUE (3)  }",
                "selected_card_id": "BattleTrance",
                "selected_card_name": "Battle Trance",
                "selected_card_instance_id": 7,
            },
            {
                "domain": "combat",
                "token": 32,
                "action_type": 2,
                "choice_index": 4,
                "label": "{ SECRET_TECHNIQUE (4)  }",
                "selected_card_id": "Armaments",
                "selected_card_name": "Armaments",
                "selected_card_instance_id": 6,
            },
        ]
        observations = []
        for current_descriptors in (descriptors, list(reversed(descriptors))):
            backend = LightspeedBackend()
            bridge = FakeDrawPileSelectionBridge(current_descriptors)
            backend._bridge = bridge
            backend._seed = 1
            observations.append(backend._read_observation())

        self.assertEqual(observations[0], observations[1])
        choices = observations[0].legal_actions
        self.assertEqual(
            [(choice.source_id, choice.choice_index, choice.label) for choice in choices],
            [(6, None, "Armaments"), (7, None, "Battle Trance")],
        )

        backend = LightspeedBackend()
        bridge = FakeDrawPileSelectionBridge(list(reversed(descriptors)))
        backend._bridge = bridge
        backend._seed = 1
        observation = backend._read_observation()
        backend.step(next(choice for choice in observation.legal_actions if choice.source_id == 6))
        self.assertEqual(bridge.steps, [32])

    def test_shop_is_exposed_as_entry_then_inventory(self) -> None:
        backend = LightspeedBackend()
        bridge = FakeShopBridge()
        backend._bridge = bridge
        backend._seed = 1
        backend._pending_card_reward = None
        backend._shop_open = False
        observation = backend._read_observation()

        self.assertEqual(
            [
                (action.kind, action.source_id)
                for action in observation.legal_actions
            ],
            [
                (ActionKind.CHOOSE_OPTION, "shop"),
                (ActionKind.CHOOSE_OPTION, "proceed"),
            ],
        )

        shop_entry = observation.legal_actions[0]
        inventory = backend.step(shop_entry).observation

        self.assertEqual(bridge.steps, [])
        self.assertEqual(inventory.screen_state, "shop_screen")
        self.assertEqual(
            {action.kind for action in inventory.legal_actions},
            {ActionKind.BUY, ActionKind.REMOVE_CARD, ActionKind.LEAVE},
        )
        leave = next(
            action for action in inventory.legal_actions if action.kind is ActionKind.LEAVE
        )
        self.assertIsNone(leave.choice_index)

        shop_room = backend.step(leave).observation

        self.assertEqual(bridge.steps, [])
        self.assertEqual(
            [action.source_id for action in shop_room.legal_actions],
            ["shop", "proceed"],
        )

    def test_dead_enemy_statuses_are_not_exposed(self) -> None:
        backend = LightspeedBackend()
        bridge = FakeRewardBridge()
        bridge.state["phase"] = "combat"
        bridge.state["screen_state"] = "battle"
        bridge.state["enemies"] = [
            {
                "enemy_id": 0,
                "monster_id": "AcidSlime_S",
                "name": "Acid Slime (S)",
                "hp": 0,
                "max_hp": 10,
                "block": 0,
                "intent_damage": 10,
                "intent_hits": 1,
                "statuses": [("VULNERABLE", 2)],
            }
        ]
        bridge.descriptors = []
        backend._bridge = bridge
        backend._seed = 1
        backend._pending_card_reward = None
        backend._shop_open = False

        observation = backend._read_observation()

        self.assertEqual(observation.enemies[0].statuses, ())
        self.assertEqual(observation.enemies[0].intent_damage, 0)
        self.assertEqual(observation.enemies[0].intent_hits, 0)

    def test_runic_dome_hides_enemy_intents(self) -> None:
        backend = LightspeedBackend()
        bridge = FakeRewardBridge()
        bridge.state["phase"] = "combat"
        bridge.state["screen_state"] = "battle"
        bridge.state["relics"] = [("Runic Dome", 0)]
        bridge.state["enemies"] = [
            {
                "enemy_id": 0,
                "monster_id": "JawWorm",
                "name": "Jaw Worm",
                "hp": 40,
                "max_hp": 40,
                "block": 0,
                "intent_damage": 12,
                "intent_hits": 1,
                "statuses": [],
            }
        ]
        bridge.descriptors = []
        backend._bridge = bridge
        backend._seed = 1
        backend._pending_card_reward = None
        backend._shop_open = False

        observation = backend._read_observation()

        self.assertEqual(observation.enemies[0].intent_damage, 0)
        self.assertEqual(observation.enemies[0].intent_hits, 0)

    def test_reset_is_deterministic_for_a_fixed_seed(self) -> None:
        first = StsEnv(LightspeedBackend())
        second = StsEnv(LightspeedBackend())

        first_observation, _ = first.reset(seed=71)
        second_observation, _ = second.reset(seed=71)

        self.assertEqual(first_observation, second_observation)

    def test_profile_history_controls_initial_boss_without_advancing_boss_rng(self) -> None:
        all_seen, all_seen_info = LightspeedBackend(
            neow_history="limited",
            act1_boss_history="all_seen",
        ).reset(seed=1)
        guardian_unseen, guardian_info = LightspeedBackend(
            neow_history="limited",
            act1_boss_history="guardian_unseen",
        ).reset(seed=1)

        self.assertEqual(all_seen.act_boss, "slimeboss")
        self.assertEqual(guardian_unseen.act_boss, "theguardian")
        self.assertEqual(all_seen_info["act1_boss_history"], "all_seen")
        self.assertEqual(guardian_info["act1_boss_history"], "guardian_unseen")

    def test_profile_history_controls_burning_elite_unlock(self) -> None:
        unlocked, unlocked_info = LightspeedBackend(final_act_unlocked=True).reset(seed=1)
        locked, locked_info = LightspeedBackend(final_act_unlocked=False).reset(seed=1)

        self.assertEqual(sum(node.burning_elite for node in unlocked.map_nodes), 1)
        self.assertFalse(any(node.burning_elite for node in locked.map_nodes))
        self.assertTrue(unlocked_info["final_act_unlocked"])
        self.assertFalse(locked_info["final_act_unlocked"])

    def test_observation_json_round_trip_is_stable(self) -> None:
        env = StsEnv(LightspeedBackend())
        observation, _ = env.reset(seed=73)
        encoded = json.dumps(observation.to_dict(), ensure_ascii=False, sort_keys=True)
        restored = Observation.from_dict(json.loads(encoded))

        self.assertEqual(restored, observation)
        self.assertNotIn("seed", encoded.lower())
        self.assertNotIn("rng", encoded.lower())
        self.assertNotIn("draw_order", encoded.lower())

    def test_all_enumerated_actions_execute_on_clones(self) -> None:
        env = StsEnv(LightspeedBackend())
        observation, _ = env.reset(seed=79)
        random_source = random.Random(79)
        checked_actions = 0

        for _ in range(200):
            for action in observation.legal_actions:
                branch = env.clone()
                branch.step(action)
                checked_actions += 1

            if not observation.legal_actions:
                self.assertIs(observation.phase, Phase.TERMINAL)
                break

            observation, _, terminated, truncated, _ = env.step(
                random_source.randrange(len(observation.legal_actions))
            )
            if terminated or truncated:
                break

        self.assertGreater(checked_actions, 100)

    def test_clone_branches_are_independent(self) -> None:
        original = StsEnv(LightspeedBackend())
        observation, _ = original.reset(seed=83)
        self.assertGreaterEqual(len(observation.legal_actions), 2)
        alternate = original.clone()

        original_result = original.step(0)
        alternate_result = alternate.step(1)

        replay = StsEnv(LightspeedBackend())
        replay.reset(seed=83)
        replay_result = replay.step(0)

        self.assertEqual(original_result, replay_result)
        self.assertNotEqual(original_result[0], alternate_result[0])

    def test_illegal_action_is_rejected_before_cpp_execution(self) -> None:
        backend = LightspeedBackend()
        backend.reset(seed=89)
        illegal = Action(kind=ActionKind.CHOOSE_OPTION, choice_index=999, label="illegal")

        with self.assertRaises(ValueError):
            backend.step(illegal)

    def test_random_legal_policy_reaches_a_terminal_state(self) -> None:
        env = StsEnv(LightspeedBackend())
        observation, _ = env.reset(seed=97)
        random_source = random.Random(97)

        for _ in range(20_000):
            self.assertTrue(observation.legal_actions)
            observation, reward, terminated, truncated, _ = env.step(
                random_source.randrange(len(observation.legal_actions))
            )
            if terminated or truncated:
                break
        else:
            self.fail("random legal policy did not terminate")

        self.assertIs(observation.phase, Phase.TERMINAL)
        self.assertEqual(observation.legal_actions, ())
        self.assertIn(reward, {-1.0, 1.0})

    def test_recorded_trace_replays_exactly(self) -> None:
        random_source = random.Random(103)
        trace = record_episode(
            StsEnv(LightspeedBackend()),
            seed=103,
            policy=lambda observation: random_source.choice(observation.legal_actions),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace.write_jsonl(path)
            restored = EpisodeTrace.read_jsonl(path)

        final_observation = replay_trace(StsEnv(LightspeedBackend()), restored)
        self.assertIs(final_observation.phase, Phase.TERMINAL)


if __name__ == "__main__":
    unittest.main()
