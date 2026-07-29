from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from sts_env import (
    Action,
    ActionKind,
    EpisodeTrace,
    Observation,
    Phase,
    PlayerView,
    TraceStep,
    observation_digest,
)
from sts_env.training import RecurrentPPOConfig, RecurrentPPOTrainer
from sts_env.training.m7c_dagger import (
    M7CDaggerLabel,
    M7C_DAGGER_TRACE_PROTOCOL,
    build_m7c_corpus_manifest,
    build_m7c_imitation_chunks,
    m7c_dagger_labels,
    record_m7c_dagger_trace,
    summarize_m7c_on_policy_labels,
    validate_m7c_dagger_trace,
    verify_m7c_corpus_manifest,
)


class BehaviorSensitiveEnvironment:
    card_behavior = Action(ActionKind.CHOOSE_CARD, source_id="behavior-card")
    card_teacher = Action(ActionKind.CHOOSE_CARD, source_id="teacher-card")
    event_behavior = Action(ActionKind.CHOOSE_OPTION, source_id="behavior-event")
    event_teacher = Action(ActionKind.CHOOSE_OPTION, source_id="teacher-event")
    map_action = Action(ActionKind.CHOOSE_MAP_NODE, source_id="teacher-map")

    def __init__(self) -> None:
        self._state = "card"
        self._observation = self._observation_for_state()

    @property
    def observation(self) -> Observation:
        return self._observation

    def reset(self, seed: int | None = None):
        self._state = "card"
        self._observation = self._observation_for_state()
        return self._observation, {"backend": "m7c-test", "seed": seed}

    def step(self, action: int | Action):
        resolved = (
            self._observation.legal_actions[action]
            if isinstance(action, int)
            else action
        )
        if resolved not in self._observation.legal_actions:
            raise ValueError("illegal M7-C fixture action")
        if self._state == "card":
            self._state = "event" if resolved == self.card_behavior else "map"
            terminated = False
        elif self._state == "event":
            self._state = "terminal"
            terminated = True
        elif self._state == "map":
            self._state = "terminal"
            terminated = True
        else:
            raise ValueError("cannot step terminal fixture")
        self._observation = self._observation_for_state()
        return self._observation, 1.0 if terminated else 0.0, terminated, False, {
            "floor": self._observation.floor
        }

    def _observation_for_state(self) -> Observation:
        if self._state == "card":
            phase, actions, floor = (
                Phase.CARD_REWARD,
                (self.card_behavior, self.card_teacher),
                1,
            )
        elif self._state == "event":
            phase, actions, floor = (
                Phase.EVENT,
                (self.event_behavior, self.event_teacher),
                2,
            )
        elif self._state == "map":
            phase, actions, floor = (Phase.MAP, (self.map_action,), 2)
        else:
            phase, actions, floor = (Phase.TERMINAL, (), 3)
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
            act=1,
            floor=floor,
        )


class LoopEnvironment:
    first_action = Action(ActionKind.CHOOSE_OPTION, source_id="loop-first")
    second_action = Action(ActionKind.CHOOSE_OPTION, source_id="loop-second")

    def __init__(self) -> None:
        self._observation = self._current_observation()

    @property
    def observation(self) -> Observation:
        return self._observation

    def reset(self, seed: int | None = None):
        self._observation = self._current_observation()
        return self._observation, {"backend": "m7c-loop-test", "seed": seed}

    def step(self, action: int | Action):
        resolved = (
            self._observation.legal_actions[action]
            if isinstance(action, int)
            else action
        )
        if resolved not in self._observation.legal_actions:
            raise ValueError("illegal M7-C loop fixture action")
        self._observation = self._current_observation()
        return self._observation, 0.0, False, False, {"floor": 1}

    def _current_observation(self) -> Observation:
        return Observation(
            phase=Phase.EVENT,
            turn=0,
            player=PlayerView(hp=80, max_hp=80, block=0, energy=0),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=(self.first_action, self.second_action),
            act=1,
            floor=1,
        )


def build_behavior_trace(seed: int = 77) -> EpisodeTrace:
    environment = BehaviorSensitiveEnvironment()
    observation, _ = environment.reset(seed=seed)
    initial_digest = observation_digest(observation)
    behaviors = (
        BehaviorSensitiveEnvironment.card_behavior,
        BehaviorSensitiveEnvironment.event_behavior,
    )
    teachers = (
        BehaviorSensitiveEnvironment.card_teacher,
        BehaviorSensitiveEnvironment.event_teacher,
    )
    steps = []
    labels = []
    for index, (behavior, teacher) in enumerate(zip(behaviors, teachers, strict=True)):
        labels.append(
            M7CDaggerLabel(
                step_index=index,
                teacher_action=teacher,
                behavior_action_index=0,
                student_action_index=0,
                phase=observation.phase,
                teacher_mixed=False,
                floor=observation.floor,
                act=observation.act,
                legal_action_count=len(observation.legal_actions),
                policy_entropy=0.5,
                policy_margin=0.25,
            )
        )
        observation, reward, terminated, truncated, info = environment.step(behavior)
        steps.append(
            TraceStep(
                action=behavior,
                observation_digest=observation_digest(observation),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
    return EpisodeTrace(
        seed=seed,
        initial_observation_digest=initial_digest,
        steps=tuple(steps),
        backend="m7c-test",
        metadata={
            "protocol": M7C_DAGGER_TRACE_PROTOCOL,
            "round_index": 0,
            "teacher_mix_probability": 0.0,
            "mixing_seed": seed,
            "behavior_policy": {"checkpoint_sha256": "fixture"},
            "teacher_identity": "fixture-teacher",
            "phase_supervision_counts": {
                "card_reward": 1,
                "event": 1,
                "map": 0,
                "rest_site": 0,
                "shop": 0,
            },
            "student_noncombat_steps": 2,
            "mixed_noncombat_steps": 0,
            "final_act": 1,
            "final_floor": 3,
            "won": True,
            "environment_return": 1.0,
            "dagger_labels": [label.to_dict() for label in labels],
        },
    )


class M7CDaggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = RecurrentPPOTrainer(
            RecurrentPPOConfig(
                recurrent_size=8,
                state_embedding_size=8,
                action_embedding_size=8,
                value_loss_weight=0.0,
                entropy_weight=0.0,
            ),
            seed=17,
        )

    def test_replay_uses_behavior_but_cross_entropy_uses_teacher(self) -> None:
        trace = build_behavior_trace()
        chunks = build_m7c_imitation_chunks(
            BehaviorSensitiveEnvironment,
            self.trainer,
            (trace,),
            chunk_length=8,
            burn_in_steps=0,
        )
        self.assertEqual(len(chunks), 1)
        self.assertTrue(torch.equal(chunks[0].chosen_actions, torch.ones(2, dtype=torch.long)))
        self.assertTrue(torch.equal(chunks[0].supervision_weights, torch.ones(2)))

    def test_illegal_teacher_label_is_rejected(self) -> None:
        trace = build_behavior_trace()
        metadata = dict(trace.metadata or {})
        labels = list(metadata["dagger_labels"])
        labels[0] = {
            **labels[0],
            "teacher_action": Action(ActionKind.CHOOSE_CARD, source_id="stale").to_dict(),
        }
        invalid = replace(trace, metadata={**metadata, "dagger_labels": labels})
        with self.assertRaisesRegex(ValueError, "teacher action is stale"):
            build_m7c_imitation_chunks(
                BehaviorSensitiveEnvironment,
                self.trainer,
                (invalid,),
                chunk_length=8,
                burn_in_steps=0,
            )

    def test_manifest_round_trip_and_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = root / "traces"
            traces.mkdir()
            for seed in (100, 101):
                build_behavior_trace(seed).write_jsonl(traces / f"seed-{seed:08d}.jsonl")
            manifest = build_m7c_corpus_manifest(
                root,
                seed_start=100,
                seed_count=2,
                round_index=0,
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            verified = verify_m7c_corpus_manifest(
                manifest_path,
                expected_seed_start=100,
                expected_seed_count=2,
                expected_round_index=0,
            )
            self.assertEqual(verified["phase_supervision_counts"]["event"], 2)
            diagnostic = summarize_m7c_on_policy_labels(
                tuple(
                    build_behavior_trace(seed)
                    for seed in (100, 101)
                )
            )
            self.assertEqual(diagnostic["student_behavior"]["agreement"], 0.0)
            trace_path = traces / "seed-00000100.jsonl"
            trace_path.write_bytes(
                trace_path.read_bytes().replace(
                    b"behavior-card",
                    b"behavior-xard",
                    1,
                )
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_m7c_corpus_manifest(manifest_path)
        self.assertEqual(len(m7c_dagger_labels(build_behavior_trace())), 2)

    def test_horizon_truncation_is_preserved_and_replayable(self) -> None:
        trace = record_m7c_dagger_trace(
            LoopEnvironment(),
            self.trainer,
            lambda observation: observation.legal_actions[-1],
            seed=11,
            max_steps=3,
            teacher_mix_probability=0.0,
            mixing_seed=29,
            behavior_policy={"checkpoint_sha256": "fixture"},
            teacher_identity="fixture-teacher",
            round_index=0,
        )
        self.assertTrue((trace.metadata or {})["horizon_truncated"])
        self.assertTrue(trace.steps[-1].truncated)
        self.assertTrue(trace.steps[-1].info["m7c_horizon_truncated"])
        validate_m7c_dagger_trace(trace, LoopEnvironment)
        chunks = build_m7c_imitation_chunks(
            LoopEnvironment,
            self.trainer,
            (trace,),
            chunk_length=8,
            burn_in_steps=0,
        )
        self.assertEqual(sum(int(chunk.supervision_weights.count_nonzero()) for chunk in chunks), 3)


if __name__ == "__main__":
    unittest.main()
