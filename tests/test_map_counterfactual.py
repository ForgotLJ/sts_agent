from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sts_env.training.map_counterfactual import (
    MapCounterfactualConfig,
    evaluate_map_counterfactuals,
    map_candidate_actions,
    validate_map_counterfactual_corpus,
    validate_map_counterfactual_record,
)
from sts_env.types import Action, ActionKind, MapNodeView, Observation, Phase, PlayerView


LEFT = Action(
    ActionKind.CHOOSE_MAP_NODE,
    source_id="x0",
    choice_index=0,
    option_type="M",
    target_x=0,
    target_y=1,
)
RIGHT = Action(
    ActionKind.CHOOSE_MAP_NODE,
    source_id="x1",
    choice_index=1,
    option_type="R",
    target_x=1,
    target_y=1,
)


def map_observation() -> Observation:
    return Observation(
        phase=Phase.MAP,
        turn=0,
        player=PlayerView(hp=80, max_hp=80, block=0, energy=0, gold=99),
        hand=(),
        enemies=(),
        draw_pile=(),
        discard_pile=(),
        exhaust_pile=(),
        legal_actions=(LEFT, RIGHT),
        ascension=20,
        act=1,
        floor=3,
        map_nodes=(
            MapNodeView(x=0, y=1, symbol="M"),
            MapNodeView(x=1, y=1, symbol="R"),
        ),
    )


class FakeMapEnvironment:
    def __init__(self, observation: Observation | None = None) -> None:
        self._observation = observation or map_observation()

    @property
    def observation(self) -> Observation:
        return self._observation

    def clone(self) -> FakeMapEnvironment:
        return copy.deepcopy(self)

    def redeterminized_clone(self, _: int) -> FakeMapEnvironment:
        return self.clone()

    def step(self, action: Action):
        if action not in self._observation.legal_actions:
            raise ValueError("illegal action")
        floor = 12 if action is LEFT else 20
        self._observation = Observation(
            phase=Phase.TERMINAL,
            turn=0,
            player=PlayerView(hp=70, max_hp=80, block=0, energy=0, gold=99),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=(),
            ascension=20,
            act=1,
            floor=floor,
        )
        return self._observation, 0.0, True, False, {"outcome": "terminal"}


class FirstLegalPolicy:
    def __call__(self, observation: Observation, _: int = 0) -> Action:
        return observation.legal_actions[0]


class MapCounterfactualTests(unittest.TestCase):
    def test_extracts_only_map_candidates(self) -> None:
        self.assertEqual(map_candidate_actions(map_observation()), (LEFT, RIGHT))

    def test_records_each_candidate_particle_outcome(self) -> None:
        environment = FakeMapEnvironment()
        record = evaluate_map_counterfactuals(
            environment,
            seed=123,
            decision_index=4,
            behavior_action=LEFT,
            rollout_policy_factory=FirstLegalPolicy,
            config=MapCounterfactualConfig(particles_per_action=2, rollout_max_steps=4),
        )
        self.assertEqual(record.behavior_action, LEFT)
        self.assertEqual(len(record.candidates), 2)
        self.assertEqual(record.candidates[0].mean_final_floor, 12.0)
        self.assertEqual(record.candidates[1].mean_final_floor, 20.0)
        self.assertEqual(len(record.candidates[0].rollouts), 2)
        self.assertEqual(environment.observation, map_observation())

    def test_validator_rejects_a_changed_particle_seed(self) -> None:
        record = evaluate_map_counterfactuals(
            FakeMapEnvironment(),
            seed=123,
            decision_index=4,
            behavior_action=LEFT,
            rollout_policy_factory=FirstLegalPolicy,
        ).to_dict()
        record["candidates"][0]["rollouts"][0]["particle_seed"] = 1
        self.assertIn("unexpected particle seed", " ".join(validate_map_counterfactual_record(record)))

    def test_validator_rejects_a_reordered_candidate_list(self) -> None:
        record = evaluate_map_counterfactuals(
            FakeMapEnvironment(),
            seed=123,
            decision_index=4,
            behavior_action=LEFT,
            rollout_policy_factory=FirstLegalPolicy,
        ).to_dict()
        record["candidates"].reverse()
        self.assertIn("order differs", " ".join(validate_map_counterfactual_record(record)))

    def test_validator_rejects_aggregate_label_drift(self) -> None:
        record = evaluate_map_counterfactuals(
            FakeMapEnvironment(),
            seed=123,
            decision_index=4,
            behavior_action=LEFT,
            rollout_policy_factory=FirstLegalPolicy,
        ).to_dict()
        record["candidates"][0]["mean_final_floor"] += 1.0
        self.assertIn("mean final floor differs", " ".join(validate_map_counterfactual_record(record)))

    def test_corpus_validator_accepts_a_complete_manifest(self) -> None:
        record = evaluate_map_counterfactuals(
            FakeMapEnvironment(),
            seed=123,
            decision_index=4,
            behavior_action=LEFT,
            rollout_policy_factory=FirstLegalPolicy,
        ).to_dict()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_path = root / "records.jsonl"
            records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "records": {"path": "records.jsonl", "sha256": digest},
                        "protocol": "map-counterfactual-rollouts",
                        "schema_version": 1,
                        "ascension": 20,
                        "seed_range": [123, 123],
                        "seed_range_name": "test",
                        "particles_per_action": 2,
                        "neow_history": "full",
                        "act1_boss_history": "all_seen",
                        "final_act_unlocked": True,
                        "counts": {"1": 1, "2": 0, "3": 0},
                        "errors": [],
                        "complete": True,
                    }
                ),
                encoding="utf-8",
            )
            result = validate_map_counterfactual_corpus(root)
        self.assertTrue(result["valid"])
        self.assertEqual(result["records"], 1)


if __name__ == "__main__":
    unittest.main()
