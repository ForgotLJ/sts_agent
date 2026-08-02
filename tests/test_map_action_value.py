from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn

from sts_env.training.map_action_value import (
    A20MapActionValuePolicy,
    MapActionFeatureEncoder,
    load_map_action_value_examples,
    load_map_action_value_model,
    save_map_action_value_checkpoint,
    split_map_action_value_examples,
    train_map_action_value_model,
)
from sts_env.training.map_counterfactual import MapCounterfactualConfig, evaluate_map_counterfactuals
from sts_env.types import Action, ActionKind, MapNodeView, Observation, Phase, PlayerView


LEFT = Action(
    ActionKind.CHOOSE_MAP_NODE,
    source_id="left",
    choice_index=0,
    option_type="M",
    target_x=0,
    target_y=1,
)
RIGHT = Action(
    ActionKind.CHOOSE_MAP_NODE,
    source_id="right",
    choice_index=1,
    option_type="R",
    target_x=1,
    target_y=1,
)


def map_observation() -> Observation:
    return Observation(
        phase=Phase.MAP,
        turn=0,
        player=PlayerView(hp=65, max_hp=80, block=0, energy=0, gold=99),
        hand=(),
        enemies=(),
        draw_pile=(),
        discard_pile=(),
        exhaust_pile=(),
        legal_actions=(LEFT, RIGHT),
        ascension=20,
        act=1,
        floor=3,
        map_x=0,
        map_y=0,
        deck=(("Strike_R", 5), ("Defend_R", 4), ("Bash", 1)),
        relics=(("Burning Blood", 1),),
        map_nodes=(
            MapNodeView(x=0, y=1, symbol="M", children=((0, 2),)),
            MapNodeView(x=1, y=1, symbol="R", children=((1, 2),)),
            MapNodeView(x=0, y=2, symbol="E", children=((0, 3),)),
            MapNodeView(x=1, y=2, symbol="$", children=((1, 3),)),
            MapNodeView(x=0, y=3, symbol="R"),
            MapNodeView(x=1, y=3, symbol="M"),
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
        floor = 15 if action is LEFT else 24
        self._observation = Observation(
            phase=Phase.TERMINAL,
            turn=0,
            player=PlayerView(hp=65, max_hp=80, block=0, energy=0, gold=99),
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


class RightRouteValueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        target_room_offset = 18 + 12 + len(("?", "M", "E", "R", "$", "T", "B"))
        return features[:, target_room_offset + 3] + self.anchor * 0.0


def write_corpus(root: Path, records: int = 120) -> None:
    payloads = []
    for seed in range(records):
        payloads.append(
            evaluate_map_counterfactuals(
                FakeMapEnvironment(),
                seed=seed,
                decision_index=0,
                behavior_action=LEFT,
                rollout_policy_factory=FirstLegalPolicy,
                config=MapCounterfactualConfig(particles_per_action=2, rollout_max_steps=4),
            ).to_dict()
        )
    records_path = root / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in payloads),
        encoding="utf-8",
    )
    digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "map-counterfactual-rollouts",
                "schema_version": 1,
                "ascension": 20,
                "seed_range": [0, records - 1],
                "particles_per_action": 2,
                "neow_history": "full",
                "act1_boss_history": "all_seen",
                "final_act_unlocked": True,
                "counts": {"1": records},
                "records": {"path": "records.jsonl", "sha256": digest},
                "errors": [],
                "complete": True,
            }
        ),
        encoding="utf-8",
    )


class MapActionValueTests(unittest.TestCase):
    def test_encoder_uses_candidate_route_topology(self) -> None:
        encoder = MapActionFeatureEncoder()
        left = encoder.encode(map_observation(), LEFT)
        right = encoder.encode(map_observation(), RIGHT)
        self.assertEqual(len(left), encoder.dimension)
        self.assertNotEqual(left, right)
        self.assertTrue(all(torch.isfinite(torch.tensor(left))))

    def test_loader_splits_by_root_seed_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_corpus(root)
            examples, _ = load_map_action_value_examples(root)
        splits = split_map_action_value_examples(examples)
        root_seeds = {name: {example.root_seed for example in values} for name, values in splits.items()}
        self.assertTrue(all(root_seeds.values()))
        self.assertFalse(root_seeds["train"] & root_seeds["validation"])
        self.assertFalse(root_seeds["train"] & root_seeds["test"])
        self.assertFalse(root_seeds["validation"] & root_seeds["test"])

    def test_multiple_decisions_from_one_root_seed_stay_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_corpus(root)
            examples, _ = load_map_action_value_examples(root)
        source = [example for example in examples if example.root_seed == 0]
        duplicated_decision = [replace(example, decision_index=1) for example in source]
        splits = split_map_action_value_examples(examples + duplicated_decision)
        matching_splits = [
            name
            for name, values in splits.items()
            if any(example.root_seed == 0 for example in values)
        ]
        self.assertEqual(len(matching_splits), 1)

    def test_policy_only_overrides_map_when_margin_is_met(self) -> None:
        policy = A20MapActionValuePolicy(
            RightRouteValueModel(),
            MapActionFeatureEncoder(),
            fallback=FirstLegalPolicy(),
            override_margin=0.5,
        )
        self.assertEqual(policy.select(FakeMapEnvironment()), RIGHT)
        self.assertEqual(policy.telemetry()["overrides"], 1)
        record_only = A20MapActionValuePolicy(
            RightRouteValueModel(),
            MapActionFeatureEncoder(),
            fallback=FirstLegalPolicy(),
            override_margin=0.5,
            record_only=True,
        )
        self.assertEqual(record_only.select(FakeMapEnvironment()), LEFT)
        self.assertEqual(record_only.telemetry()["overrides"], 0)

    def test_policy_never_scores_an_untrained_act(self) -> None:
        policy = A20MapActionValuePolicy(
            RightRouteValueModel(),
            MapActionFeatureEncoder(),
            fallback=FirstLegalPolicy(),
            override_margin=0.0,
            allowed_acts={1},
        )
        action = policy.select(FakeMapEnvironment(replace(map_observation(), act=2)))
        self.assertEqual(action, LEFT)
        self.assertEqual(policy.telemetry()["map_decisions"], 0)
        self.assertEqual(policy.telemetry()["untrained_act_map_decisions"], 1)
        self.assertEqual(policy.telemetry()["candidate_actions_scored"], 0)

    def test_small_training_and_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_corpus(root)
            examples, _ = load_map_action_value_examples(root)
            model, encoder, metrics = train_map_action_value_model(
                examples,
                epochs=2,
                groups_per_batch=16,
                seed=17,
            )
            checkpoint = root / "map-action-value.pt"
            save_map_action_value_checkpoint(
                checkpoint,
                model,
                encoder,
                metrics=metrics,
                metadata={"test": True},
            )
            restored, restored_encoder, metadata = load_map_action_value_model(checkpoint)
        self.assertEqual(encoder.dimension, restored_encoder.dimension)
        self.assertEqual(metadata["metadata"], {"test": True})
        features = torch.tensor([encoder.encode(map_observation(), RIGHT)], dtype=torch.float32)
        self.assertTrue(torch.allclose(model(features), restored(features)))
        self.assertGreater(metrics["test"]["groups"], 0.0)

    def test_training_cli_writes_frozen_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_corpus(root)
            checkpoint = root / "map-action-value.pt"
            frozen_evaluation = root / "frozen-evaluation.json"
            project = Path(__file__).resolve().parents[1]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(project / ".local_packages"), str(project / "src"))
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts" / "train-map-action-value.py"),
                    "--input",
                    str(root),
                    "--output",
                    str(checkpoint),
                    "--frozen-evaluation",
                    str(frozen_evaluation),
                    "--epochs",
                    "1",
                    "--groups-per-batch",
                    "16",
                    "--device",
                    "cpu",
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(checkpoint.is_file())
            payload = json.loads(frozen_evaluation.read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol"], "a20-map-action-value-offline-evaluation")
        self.assertEqual(payload["trained_acts"], [1])
        self.assertGreater(payload["metrics"]["test"]["examples"], 0.0)


if __name__ == "__main__":
    unittest.main()
