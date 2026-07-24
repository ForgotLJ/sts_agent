from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from sts_env import Action, ActionKind, Observation, Phase, PlayerView


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "real-game-differential.py"
SPEC = importlib.util.spec_from_file_location("real_game_differential", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load real-game-differential.py")
DIFFERENTIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIFFERENTIAL)


def observation(*actions: Action, phase: Phase = Phase.CARD_REWARD) -> Observation:
    return Observation(
        phase=phase,
        turn=0,
        player=PlayerView(hp=80, max_hp=80, block=0, energy=0),
        hand=(),
        enemies=(),
        draw_pile=(),
        discard_pile=(),
        exhaust_pile=(),
        legal_actions=actions,
    )


class RealGameDifferentialTests(unittest.TestCase):
    def test_talk_screen_requires_explicit_neow_history(self) -> None:
        talk = Action(
            kind=ActionKind.CHOOSE_OPTION,
            choice_index=0,
            label="Talk",
        )

        with self.assertRaisesRegex(RuntimeError, "not observable"):
            DIFFERENTIAL.resolve_neow_history("auto", observation(talk, phase=Phase.EVENT))

        self.assertEqual(
            DIFFERENTIAL.resolve_neow_history("full", observation(talk, phase=Phase.EVENT)),
            "full",
        )

    def test_combat_pairing_prioritizes_matching_potion_action(self) -> None:
        reference_potion = Action(
            kind=ActionKind.USE_POTION,
            source_id=0,
            label="Use Liquid Bronze",
        )
        candidate_potion = Action(
            kind=ActionKind.USE_POTION,
            source_id=0,
            label="Use Liquid Bronze",
        )
        reference_card = Action(
            kind=ActionKind.PLAY_CARD,
            source_id=0,
            target_id=0,
            label="Play Strike",
        )
        candidate_card = Action(
            kind=ActionKind.PLAY_CARD,
            source_id=1,
            target_id=0,
            label="Play Strike",
        )

        paired = DIFFERENTIAL.pair_action(
            observation(reference_card, reference_potion, phase=Phase.COMBAT),
            observation(candidate_card, candidate_potion, phase=Phase.COMBAT),
        )

        self.assertEqual(paired, (reference_potion, candidate_potion))

    def test_single_event_leave_is_reference_only_presentation(self) -> None:
        reference_leave = Action(
            kind=ActionKind.CHOOSE_OPTION,
            choice_index=0,
            label="leave",
        )
        reference_potion = Action(
            kind=ActionKind.DISCARD_POTION,
            source_id=0,
            label="Discard Liquid Bronze",
        )
        candidate_map = Action(
            kind=ActionKind.CHOOSE_MAP_NODE,
            source_id="x3",
            choice_index=3,
            label="Choose map node 3",
        )

        paired = DIFFERENTIAL.pair_action(
            observation(reference_potion, reference_leave, phase=Phase.EVENT),
            observation(candidate_map, phase=Phase.MAP),
        )

        self.assertEqual(paired, (reference_leave, None))

    def test_shop_room_prefers_proceed_after_inventory_return(self) -> None:
        reference_shop = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="shop",
            choice_index=0,
            label="shop",
        )
        reference_proceed = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="proceed",
            label="Proceed",
        )
        candidate_shop = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="shop",
            choice_index=0,
            label="shop",
        )
        candidate_proceed = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="proceed",
            label="Proceed",
        )

        paired = DIFFERENTIAL.pair_action(
            observation(reference_shop, reference_proceed, phase=Phase.SHOP),
            observation(candidate_shop, candidate_proceed, phase=Phase.SHOP),
        )

        self.assertEqual(paired, (reference_proceed, candidate_proceed))

    def test_prefix_trace_recovers_latest_initial_map_coordinate(self) -> None:
        records = [
            {
                "direction": "game_to_agent",
                "payload": {
                    "game_state": {
                        "seed": 1,
                        "screen_type": "MAP",
                        "choice_list": ["x=1", "x=3"],
                    }
                },
            },
            {"direction": "agent_to_game", "command": "choose 1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            source_id = DIFFERENTIAL.latest_map_source_from_trace(path, seed=1)

        self.assertEqual(source_id, "x3")

    def test_latest_reference_session_uses_last_matching_start(self) -> None:
        records = [
            {"direction": "agent_to_game", "command": "start ironclad 0 1"},
            {"direction": "game_to_agent", "payload": {"in_game": True}},
            {"direction": "agent_to_game", "command": "start ironclad 0 2"},
            {"direction": "game_to_agent", "payload": {"in_game": True}},
            {"direction": "agent_to_game", "command": "start ironclad 0 1"},
            {"direction": "game_to_agent", "payload": {"in_game": True}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relay.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            start, session = DIFFERENTIAL.latest_reference_session(path, seed=1)

        self.assertEqual(start, 4)
        self.assertEqual(session, records[4:])

    def test_resume_realigns_flattened_card_reward_stage(self) -> None:
        reference_card_entry = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="card",
            choice_index=0,
            label="card",
        )
        candidate_card = Action(
            kind=ActionKind.CHOOSE_CARD,
            source_id="carnage",
            choice_index=0,
            label="Carnage",
        )
        root_candidate = observation(reference_card_entry)

        class FakeBackend:
            _pending_card_reward = 0

            def _read_observation(self) -> Observation:
                return root_candidate

        backend = FakeBackend()

        aligned = DIFFERENTIAL.align_candidate_reward_stage(
            observation(reference_card_entry),
            backend,
            observation(candidate_card),
        )

        self.assertIsNone(backend._pending_card_reward)
        self.assertEqual(aligned, root_candidate)

    def test_reward_pairing_prefers_semantics_over_choice_index(self) -> None:
        reference_gold = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="gold",
            choice_index=0,
            label="gold",
        )
        reference_card = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="card",
            choice_index=1,
            label="card",
        )
        candidate_card = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="card",
            choice_index=0,
            label="card",
        )
        candidate_gold = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="gold",
            choice_index=1,
            label="gold",
        )

        paired = DIFFERENTIAL.pair_action(
            observation(reference_gold, reference_card),
            observation(candidate_card, candidate_gold),
        )

        self.assertEqual(paired, (reference_gold, candidate_gold))

    def test_card_pairing_uses_card_identity(self) -> None:
        reference = Action(
            kind=ActionKind.CHOOSE_CARD,
            source_id="bodyslam",
            choice_index=0,
            label="Body Slam",
        )
        candidate = Action(
            kind=ActionKind.CHOOSE_CARD,
            source_id="bodyslam",
            choice_index=2,
            label="Body Slam",
        )

        paired = DIFFERENTIAL.pair_action(
            observation(reference),
            observation(candidate),
        )

        self.assertEqual(paired, (reference, candidate))

    def test_proceed_presentation_action_advances_both_backends(self) -> None:
        reference = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="proceed",
            label="Proceed",
        )
        candidate = Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id="proceed",
            label="Proceed",
        )

        paired = DIFFERENTIAL.pair_action(
            observation(reference),
            observation(candidate),
        )

        self.assertEqual(paired, (reference, candidate))


if __name__ == "__main__":
    unittest.main()
