from __future__ import annotations

from collections import deque
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any
import unittest
from unittest import mock

from sts_env import (
    ActionKind,
    CommunicationBackend,
    Phase,
    SocketRelayTransport,
    StsEnv,
)


def menu_state() -> dict[str, Any]:
    return {
        "available_commands": ["start", "state"],
        "ready_for_command": True,
        "in_game": False,
    }


def combat_state() -> dict[str, Any]:
    strike = {
        "id": "Strike_R",
        "name": "Strike",
        "cost": 1,
        "upgrades": 0,
        "is_playable": True,
        "has_target": True,
    }
    defend = {
        "id": "Defend_R",
        "name": "Defend",
        "cost": 1,
        "upgrades": 0,
        "is_playable": True,
        "has_target": False,
    }
    return {
        "available_commands": ["play", "end", "potion", "state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "screen_type": "NONE",
            "screen_state": {},
            "seed": 0,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "act": 1,
            "floor": 1,
            "ascension_level": 0,
            "room_phase": "COMBAT",
            "deck": [strike, defend],
            "relics": [{"id": "Burning Blood", "counter": -1}],
            "potions": [
                {
                    "id": "Fire Potion",
                    "name": "Fire Potion",
                    "requires_target": True,
                    "can_use": True,
                    "can_discard": True,
                }
            ],
            "combat_state": {
                "turn": 1,
                "player": {
                    "current_hp": 80,
                    "max_hp": 80,
                    "block": 0,
                    "energy": 3,
                    "powers": [],
                },
                "hand": [strike, defend],
                "draw_pile": [strike],
                "discard_pile": [],
                "exhaust_pile": [],
                "monsters": [
                    {
                        "id": "Cultist",
                        "name": "Cultist",
                        "current_hp": 48,
                        "max_hp": 48,
                        "block": 0,
                        "move_adjusted_damage": 0,
                        "move_base_damage": 0,
                        "move_hits": 0,
                        "is_gone": False,
                        "half_dead": False,
                        "powers": [],
                    }
                ],
            },
        },
    }


def terminal_state() -> dict[str, Any]:
    return {
        "available_commands": ["state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "screen_type": "GAME_OVER",
            "screen_state": {"victory": False},
            "seed": 0,
            "current_hp": 0,
            "max_hp": 80,
            "gold": 99,
            "act": 1,
            "floor": 1,
            "ascension_level": 0,
            "room_phase": "COMPLETE",
            "deck": [],
            "relics": [],
            "potions": [],
        },
    }


def reward_state(screen_type: str, choices: list[str]) -> dict[str, Any]:
    return {
        "available_commands": ["choose", "proceed", "return", "state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "screen_type": screen_type,
            "screen_state": {},
            "choice_list": choices,
            "seed": 0,
            "current_hp": 80,
            "max_hp": 80,
            "gold": 99,
            "act": 1,
            "floor": 1,
            "ascension_level": 0,
            "room_phase": "COMPLETE",
            "deck": [],
            "relics": [],
            "potions": [],
        },
    }


class FakeTransport:
    def __init__(self, initial: dict[str, Any], responses: list[dict[str, Any]]):
        self.initial = initial
        self.responses = deque(responses)
        self.commands: list[str] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def receive(self) -> dict[str, Any]:
        return self.initial

    def exchange(self, command: str) -> dict[str, Any]:
        self.commands.append(command)
        return self.responses.popleft()

    def close(self) -> None:
        self.connected = False


class CommunicationBackendTests(unittest.TestCase):
    def test_transport_waits_for_delayed_relay(self) -> None:
        connection = mock.Mock(spec=socket.socket)
        with mock.patch(
            "sts_env.communication_backend.socket.create_connection",
            side_effect=[ConnectionRefusedError(), connection],
        ) as create_connection:
            transport = SocketRelayTransport(
                timeout=5.0,
                connect_wait_timeout=1.0,
                retry_interval=0.0,
            )
            transport.connect()

        self.assertEqual(create_connection.call_count, 2)
        connection.settimeout.assert_called_once_with(5.0)

    def test_tcp_relay_preserves_line_protocol(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        script = Path(__file__).resolve().parents[1] / "scripts" / "communication-relay.py"
        process = subprocess.Popen(
            [sys.executable, str(script), "--port", str(port)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        transport = SocketRelayTransport(port=port, timeout=5.0)
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            transport.connect()
            process.stdin.write(json.dumps(menu_state()) + "\n")
            process.stdin.flush()
            self.assertEqual(transport.receive(), menu_state())

            process.stdin.write(json.dumps(combat_state()) + "\n")
            process.stdin.flush()
            self.assertEqual(transport.exchange("state"), combat_state())
            self.assertEqual(process.stdout.readline().strip(), "state")
        finally:
            transport.close()
            process.terminate()
            process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()

    def test_combat_json_maps_to_public_actions(self) -> None:
        transport = FakeTransport(menu_state(), [combat_state()])
        backend = CommunicationBackend(transport=transport)
        observation, info = backend.reset(seed=0)

        self.assertEqual(transport.commands, ["start ironclad 0 0"])
        self.assertEqual(info["game_seed"], "0")
        self.assertIs(observation.phase, Phase.COMBAT)
        self.assertEqual(observation.draw_pile, (("Strike_R", 1),))
        self.assertEqual(observation.relics, (("Burning Blood", 0),))
        self.assertEqual(
            {action.kind for action in observation.legal_actions},
            {
                ActionKind.PLAY_CARD,
                ActionKind.END_TURN,
                ActionKind.USE_POTION,
                ActionKind.DISCARD_POTION,
            },
        )

    def test_binary_confusion_power_uses_common_encoding(self) -> None:
        self.assertEqual(
            CommunicationBackend._power_state(
                [{"id": "Confusion", "name": "Confusion", "amount": -1}]
            ),
            (("CONFUSED", 1),),
        )

    def test_gremlin_nob_anger_uses_simulator_enrage_name(self) -> None:
        self.assertEqual(
            CommunicationBackend._power_state(
                [{"id": "Anger", "name": "Anger", "amount": 2}]
            ),
            (("ENRAGE", 2),),
        )

    def test_weakened_uses_common_weak_name(self) -> None:
        self.assertEqual(
            CommunicationBackend._power_state(
                [{"id": "Weakened", "name": "Weakened", "amount": 1}]
            ),
            (("WEAK", 1),),
        )

    def test_dead_enemy_combat_metadata_is_suppressed(self) -> None:
        state = combat_state()
        monster = state["game_state"]["combat_state"]["monsters"][0]
        monster["current_hp"] = 0
        monster["move_adjusted_damage"] = 10
        monster["move_hits"] = 1
        monster["powers"] = [{"id": "Strength", "amount": 3}]
        backend = CommunicationBackend(transport=FakeTransport(state, []))

        observation, _ = backend.attach(seed=0)

        self.assertEqual(observation.enemies[0].intent_damage, 0)
        self.assertEqual(observation.enemies[0].intent_hits, 0)
        self.assertEqual(observation.enemies[0].statuses, ())

    def test_runic_dome_hides_enemy_intents(self) -> None:
        state = combat_state()
        state["game_state"]["relics"] = [
            {"id": "Runic Dome", "name": "Runic Dome", "counter": -1}
        ]
        monster = state["game_state"]["combat_state"]["monsters"][0]
        monster["move_adjusted_damage"] = 12
        monster["move_hits"] = 1
        backend = CommunicationBackend(transport=FakeTransport(state, []))

        observation, _ = backend.attach(seed=0)

        self.assertEqual(observation.enemies[0].intent_damage, 0)
        self.assertEqual(observation.enemies[0].intent_hits, 0)

    def test_reward_screens_use_two_stage_semantic_actions(self) -> None:
        combat_reward = CommunicationBackend(
            transport=FakeTransport(reward_state("COMBAT_REWARD", ["gold", "card"]), [])
        )
        observation, _ = combat_reward.attach(seed=0)

        choices = {
            (action.kind, action.source_id, action.choice_index)
            for action in observation.legal_actions
        }
        self.assertIn((ActionKind.CHOOSE_OPTION, "gold", 0), choices)
        self.assertIn((ActionKind.CHOOSE_OPTION, "card", 1), choices)
        self.assertNotIn(ActionKind.CHOOSE_CARD, {action.kind for action in observation.legal_actions})

        card_reward = CommunicationBackend(
            transport=FakeTransport(
                reward_state("CARD_REWARD", ["Body Slam", "Clothesline"]),
                [],
            )
        )
        observation, _ = card_reward.attach(seed=0)

        card_choices = [
            action
            for action in observation.legal_actions
            if action.kind is ActionKind.CHOOSE_CARD
        ]
        self.assertEqual(
            [(action.source_id, action.choice_index) for action in card_choices],
            [("bodyslam", 0), ("clothesline", 1)],
        )

    def test_map_choices_use_node_coordinates_as_semantic_ids(self) -> None:
        state = reward_state("MAP", ["x=3", "x=4"])
        state["game_state"].update(
            act_boss="The Guardian",
            keys={"ruby": True, "emerald": False, "sapphire": True},
            potions=[{"id": "Potion Slot"}] * 3,
            map=[
                {
                    "x": 1,
                    "y": 0,
                    "symbol": "M",
                    "burning_elite": False,
                    "children": [{"x": 3, "y": 1}, {"x": 4, "y": 1}],
                }
            ],
        )
        state["game_state"]["screen_state"] = {
            "current_node": {"x": 1, "y": 0, "symbol": "M"},
            "next_nodes": [
                {"x": 3, "y": 1, "symbol": "?"},
                {"x": 4, "y": 1, "symbol": "E"},
            ],
            "first_node_chosen": True,
            "boss_available": False,
        }
        backend = CommunicationBackend(transport=FakeTransport(state, []))

        observation, _ = backend.attach(seed=0)

        map_actions = [
            action
            for action in observation.legal_actions
            if action.kind is ActionKind.CHOOSE_MAP_NODE
        ]
        self.assertEqual(
            [(action.source_id, action.choice_index) for action in map_actions],
            [("x3", 0), ("x4", 1)],
        )
        self.assertEqual(
            [(action.target_x, action.target_y, action.option_type) for action in map_actions],
            [(3, 1, "?"), (4, 1, "E")],
        )
        self.assertEqual((observation.map_x, observation.map_y), (1, 0))
        self.assertEqual(observation.act_boss, "theguardian")
        self.assertEqual(observation.potion_capacity, 3)
        self.assertTrue(observation.ruby_key)
        self.assertTrue(observation.sapphire_key)
        self.assertEqual(observation.map_nodes[0].children, ((3, 1), (4, 1)))

    def test_shop_choices_include_public_prices(self) -> None:
        state = reward_state("SHOP_SCREEN", ["purge", "Carnage"])
        state["game_state"]["screen_state"] = {
            "purge_available": True,
            "purge_cost": 75,
            "cards": [{"id": "Carnage", "name": "Carnage", "price": 70}],
            "relics": [],
            "potions": [],
        }
        backend = CommunicationBackend(transport=FakeTransport(state, []))

        observation, _ = backend.attach(seed=0)

        purge = next(action for action in observation.legal_actions if action.source_id == "purge")
        carnage = next(
            action for action in observation.legal_actions if action.source_id == "carnage"
        )
        self.assertEqual((purge.option_type, purge.gold_cost), ("remove_card", 75))
        self.assertEqual((carnage.option_type, carnage.gold_cost), ("card", 70))

    def test_shop_room_and_shop_screen_are_distinct_action_stages(self) -> None:
        shop_room = CommunicationBackend(
            transport=FakeTransport(reward_state("SHOP_ROOM", ["shop"]), [])
        )
        observation, _ = shop_room.attach(seed=0)

        self.assertIn(
            (ActionKind.CHOOSE_OPTION, "shop", 0),
            {
                (action.kind, action.source_id, action.choice_index)
                for action in observation.legal_actions
            },
        )

        shop_screen = CommunicationBackend(
            transport=FakeTransport(reward_state("SHOP_SCREEN", ["Carnage"]), [])
        )
        observation, _ = shop_screen.attach(seed=0)

        self.assertIn(
            (ActionKind.BUY, "carnage", 0),
            {
                (action.kind, action.source_id, action.choice_index)
                for action in observation.legal_actions
            },
        )

    def test_completed_event_leave_reports_underlying_map_phase(self) -> None:
        state = reward_state("EVENT", ["leave"])
        state["available_commands"] = ["choose", "state"]
        backend = CommunicationBackend(
            transport=FakeTransport(state, [])
        )

        observation, _ = backend.attach(seed=0)

        self.assertIs(observation.phase, Phase.MAP)
        self.assertEqual(len(observation.legal_actions), 1)
        leave = next(action for action in observation.legal_actions if action.label == "leave")
        self.assertIs(leave.kind, ActionKind.CHOOSE_OPTION)

    def test_completed_rest_screen_reports_underlying_map_phase(self) -> None:
        state = reward_state("REST", [])
        state["available_commands"] = ["proceed", "state"]
        backend = CommunicationBackend(transport=FakeTransport(state, []))

        observation, _ = backend.attach(seed=0)

        self.assertIs(observation.phase, Phase.MAP)
        self.assertEqual(observation.legal_actions[0].source_id, "proceed")

    def test_attach_recovers_matching_in_game_state(self) -> None:
        transport = FakeTransport(combat_state(), [])
        backend = CommunicationBackend(transport=transport)

        observation, info = backend.attach(seed=0)

        self.assertTrue(transport.connected)
        self.assertIs(observation.phase, Phase.COMBAT)
        self.assertTrue(info["attached"])
        self.assertEqual(transport.commands, [])

    def test_attach_rejects_wrong_seed(self) -> None:
        state = combat_state()
        state["game_state"]["seed"] = 7
        backend = CommunicationBackend(transport=FakeTransport(state, []))

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            backend.attach(seed=0)

    def test_attach_waits_for_action_ready_state(self) -> None:
        transition = combat_state()
        transition["available_commands"] = ["key", "click", "wait", "state"]
        backend = CommunicationBackend(
            transport=FakeTransport(transition, [combat_state()])
        )

        observation, _ = backend.attach(seed=0)

        self.assertIs(observation.phase, Phase.COMBAT)
        self.assertTrue(observation.hand)
        self.assertEqual(backend._transport.commands, ["wait 250"])

    def test_attach_ignores_global_potion_command_without_public_actions(self) -> None:
        transition = reward_state("NONE", [])
        transition["available_commands"] = ["potion", "key", "click", "wait", "state"]
        final = reward_state("CARD_REWARD", ["Body Slam", "Clothesline"])
        backend = CommunicationBackend(
            transport=FakeTransport(transition, [final])
        )

        observation, _ = backend.attach(seed=0)

        self.assertEqual(observation.screen_state, "card_reward")
        self.assertEqual(backend._transport.commands, ["wait 250"])

    def test_ftue_overlay_is_a_reference_only_presentation_action(self) -> None:
        ftue = reward_state("NONE", [])
        ftue["available_commands"] = ["potion", "key", "click", "wait", "state"]
        ftue["game_state"]["screen_name"] = "FTUE"
        ftue["game_state"]["room_type"] = "MonsterRoom"
        ftue["game_state"]["potions"] = [
            {
                "id": "Fire Potion",
                "name": "Fire Potion",
                "requires_target": True,
                "can_use": True,
                "can_discard": True,
            }
        ]
        settings = copy.deepcopy(ftue)
        settings["game_state"]["screen_name"] = "SETTINGS"
        final = reward_state("COMBAT_REWARD", ["card"])
        transport = FakeTransport(ftue, [settings, copy.deepcopy(settings), final])
        backend = CommunicationBackend(transport=transport)

        observation, _ = backend.attach(seed=0)

        self.assertIs(observation.phase, Phase.CARD_REWARD)
        self.assertEqual(len(observation.legal_actions), 1)
        self.assertEqual(observation.legal_actions[0].label, "Continue")

        settings_transition = backend.step(observation.legal_actions[0])

        self.assertIs(settings_transition.observation.phase, Phase.CARD_REWARD)
        self.assertEqual(settings_transition.observation.legal_actions[0].label, "Continue")

        transition = backend.step(settings_transition.observation.legal_actions[0])

        self.assertIs(transition.observation.phase, Phase.CARD_REWARD)
        self.assertEqual(
            transport.commands,
            ["key cancel 1000", "wait 250", "key cancel 1000"],
        )

    def test_attach_catches_up_from_stale_menu_snapshot(self) -> None:
        transport = FakeTransport(menu_state(), [combat_state()])
        backend = CommunicationBackend(transport=transport)

        observation, _ = backend.attach(seed=0)

        self.assertIs(observation.phase, Phase.COMBAT)
        self.assertEqual(transport.commands, ["state"])

    def test_reset_refreshes_stale_relay_state(self) -> None:
        transport = FakeTransport(combat_state(), [menu_state(), combat_state()])
        backend = CommunicationBackend(transport=transport)

        observation, _ = backend.reset(seed=0)

        self.assertIs(observation.phase, Phase.COMBAT)
        self.assertEqual(transport.commands, ["state", "start ironclad 0 0"])

    def test_reset_rejects_game_seed_substitution(self) -> None:
        state = combat_state()
        state["game_state"]["seed"] = 7
        backend = CommunicationBackend(transport=FakeTransport(menu_state(), [state]))

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            backend.reset(seed=0)

    def test_reset_waits_for_asynchronous_game_start(self) -> None:
        transport = FakeTransport(menu_state(), [menu_state(), combat_state()])
        backend = CommunicationBackend(transport=transport)

        observation, _ = backend.reset(seed=0)

        self.assertIs(observation.phase, Phase.COMBAT)
        self.assertEqual(transport.commands, ["start ironclad 0 0", "state"])

    def test_play_command_uses_one_based_hand_index(self) -> None:
        transport = FakeTransport(menu_state(), [combat_state(), terminal_state()])
        env = StsEnv(CommunicationBackend(transport=transport))
        observation, _ = env.reset(seed=0)
        strike_index = next(
            index
            for index, action in enumerate(observation.legal_actions)
            if action.kind is ActionKind.PLAY_CARD and action.source_id == 0
        )

        observation, reward, terminated, truncated, _ = env.step(strike_index)

        self.assertEqual(transport.commands[-1], "play 1 0")
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -1.0)
        self.assertIs(observation.phase, Phase.TERMINAL)
        self.assertEqual(observation.legal_actions, ())

    def test_step_waits_through_room_transition(self) -> None:
        transition = combat_state()
        transition["available_commands"] = ["key", "click", "wait", "state"]
        transition["game_state"]["combat_state"]["hand"] = []
        transport = FakeTransport(
            menu_state(),
            [combat_state(), transition, terminal_state()],
        )
        env = StsEnv(CommunicationBackend(transport=transport))
        observation, _ = env.reset(seed=0)
        strike = next(
            action
            for action in observation.legal_actions
            if action.kind is ActionKind.PLAY_CARD and action.source_id == 0
        )

        observation, _, terminated, _, _ = env.step(strike)

        self.assertTrue(terminated)
        self.assertIs(observation.phase, Phase.TERMINAL)
        self.assertEqual(transport.commands[-2:], ["play 1 0", "wait 250"])

    def test_step_ignores_stale_actionable_echo(self) -> None:
        stale = combat_state()
        stale["available_commands"].append("wait")
        transport = FakeTransport(
            menu_state(),
            [combat_state(), stale, terminal_state()],
        )
        env = StsEnv(CommunicationBackend(transport=transport))
        observation, _ = env.reset(seed=0)
        strike = next(
            action
            for action in observation.legal_actions
            if action.kind is ActionKind.PLAY_CARD and action.source_id == 0
        )

        observation, _, terminated, _, _ = env.step(strike)

        self.assertTrue(terminated)
        self.assertIs(observation.phase, Phase.TERMINAL)
        self.assertEqual(transport.commands[-2:], ["play 1 0", "wait 250"])

    def test_step_ignores_raw_state_change_without_public_observation_change(self) -> None:
        transient = combat_state()
        transient["available_commands"].append("wait")
        transient["game_state"]["current_action"] = "TransientVisualEffect"
        transport = FakeTransport(
            menu_state(),
            [combat_state(), transient, terminal_state()],
        )
        env = StsEnv(CommunicationBackend(transport=transport))
        observation, _ = env.reset(seed=0)
        strike = next(
            action
            for action in observation.legal_actions
            if action.kind is ActionKind.PLAY_CARD and action.source_id == 0
        )

        observation, _, terminated, _, _ = env.step(strike)

        self.assertTrue(terminated)
        self.assertIs(observation.phase, Phase.TERMINAL)
        self.assertEqual(transport.commands[-2:], ["play 1 0", "wait 250"])

    def test_step_waits_for_changed_observation_to_stabilize(self) -> None:
        transient = reward_state("COMBAT_REWARD", ["card"])
        transient["available_commands"].append("wait")
        final = copy.deepcopy(transient)
        final["game_state"]["deck"] = [
            {
                "id": "Body Slam",
                "name": "Body Slam",
                "cost": 1,
                "upgrades": 0,
                "is_playable": False,
                "has_target": True,
            }
        ]
        transport = FakeTransport(
            menu_state(),
            [combat_state(), transient, final, copy.deepcopy(final)],
        )
        env = StsEnv(CommunicationBackend(transport=transport))
        observation, _ = env.reset(seed=0)
        strike = next(
            action
            for action in observation.legal_actions
            if action.kind is ActionKind.PLAY_CARD and action.source_id == 0
        )

        observation, _, terminated, _, _ = env.step(strike)

        self.assertFalse(terminated)
        self.assertEqual(observation.deck, (("Body Slam", 1),))
        self.assertEqual(transport.commands[-3:], ["play 1 0", "wait 250", "wait 250"])

    def test_real_backend_explicitly_rejects_clone(self) -> None:
        transport = FakeTransport(menu_state(), [combat_state()])
        env = StsEnv(CommunicationBackend(transport=transport))
        env.reset(seed=0)

        with self.assertRaises(RuntimeError):
            env.clone()


if __name__ == "__main__":
    unittest.main()
