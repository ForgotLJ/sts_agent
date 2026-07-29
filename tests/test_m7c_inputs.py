from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from sts_env import Action, ActionKind, Observation, Phase, PlayerView
from sts_env.training import (
    M7B_SUPERVISED_PHASES,
    build_m7b_corpus_manifest,
    record_m7b_teacher_trace,
)
from sts_env.training.m7c_inputs import (
    M7C_FROZEN_CHECKPOINT_NAME,
    M7C_FROZEN_M6_BASELINE_NAME,
    M7C_FROZEN_TEACHER_DIRECTORY,
    build_m7c_frozen_inputs_manifest,
    checkpoint_identity,
    safe_relative_path,
    verify_m7c_frozen_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "package_m7c_frozen_inputs_test",
    PROJECT_ROOT / "scripts" / "package-m7c-frozen-inputs.py",
)
if PACKAGE_SPEC is None or PACKAGE_SPEC.loader is None:
    raise RuntimeError("unable to load M7-C packaging script")
PACKAGE_M7C = importlib.util.module_from_spec(PACKAGE_SPEC)
PACKAGE_SPEC.loader.exec_module(PACKAGE_M7C)


class FivePhaseEnvironment:
    behavior_action = Action(ActionKind.CHOOSE_OPTION, source_id="behavior")
    teacher_action = Action(ActionKind.CHOOSE_OPTION, source_id="teacher")

    def __init__(self) -> None:
        self.index = 0
        self._observation = self._active_observation()

    @property
    def observation(self) -> Observation:
        return self._observation

    def reset(self, seed: int | None = None):
        self.index = 0
        self._observation = self._active_observation()
        return self._observation, {"seed": seed}

    def step(self, action: int | Action):
        resolved = (
            self._observation.legal_actions[action]
            if isinstance(action, int)
            else action
        )
        if resolved not in self._observation.legal_actions:
            raise ValueError("illegal fixture action")
        self.index += 1
        terminal = self.index == len(M7B_SUPERVISED_PHASES)
        self._observation = (
            Observation(
                phase=Phase.TERMINAL,
                turn=0,
                player=PlayerView(hp=80, max_hp=80, block=0, energy=0),
                hand=(),
                enemies=(),
                draw_pile=(),
                discard_pile=(),
                exhaust_pile=(),
                legal_actions=(),
                act=1,
                floor=self.index + 1,
            )
            if terminal
            else self._active_observation()
        )
        return self._observation, float(terminal), terminal, False, {"floor": self.index + 1}

    def _active_observation(self) -> Observation:
        return Observation(
            phase=M7B_SUPERVISED_PHASES[self.index],
            turn=0,
            player=PlayerView(hp=80, max_hp=80, block=0, energy=0),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=(self.behavior_action, self.teacher_action),
            act=1,
            floor=self.index + 1,
        )


class M7CFrozenInputsTests(unittest.TestCase):
    def test_checksum_sidecar_uses_linux_compatible_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "inputs.tar.gz"
            archive.write_bytes(b"fixture")
            sidecar = PACKAGE_M7C.write_checksum_sidecar(archive, "a" * 64)

            payload = sidecar.read_bytes()

        self.assertEqual(payload, ("a" * 64 + "  inputs.tar.gz\n").encode("ascii"))
        self.assertNotIn(b"\r", payload)

    def test_build_and_verify_relocated_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_root = root / "teacher-source"
            traces = teacher_root / "traces"
            traces.mkdir(parents=True)
            for seed in (7, 8):
                record_m7b_teacher_trace(
                    FivePhaseEnvironment(),
                    lambda _: FivePhaseEnvironment.teacher_action,
                    seed=seed,
                    max_steps=8,
                ).write_jsonl(traces / f"seed-{seed:08d}.jsonl")
            teacher_manifest = build_m7b_corpus_manifest(
                teacher_root,
                seed_start=7,
                seed_count=2,
            )
            teacher_manifest_path = teacher_root / "manifest.json"
            teacher_manifest_path.write_text(
                json.dumps(teacher_manifest),
                encoding="utf-8",
            )
            m7b_checkpoint = root / "m7b.pt"
            m6_checkpoint = root / "m6.pt"
            torch.save(
                {
                    "protocol": "m7b",
                    "config": {"run_seed": 17},
                    "manifest": {"evaluation_only": True},
                },
                m7b_checkpoint,
            )
            torch.save(
                {
                    "config": {"run_seed": 17},
                    "manifest": {"evaluation_only": True},
                },
                m6_checkpoint,
            )
            identity = {
                "teacher_corpus": {
                    "seed_start": 7,
                    "seed_count": 2,
                    "aggregate_sha256": teacher_manifest["aggregate_sha256"],
                },
                "initial_checkpoint": checkpoint_identity(
                    m7b_checkpoint,
                    expected_protocol="m7b",
                ),
                "m6_baseline_checkpoint": checkpoint_identity(
                    m6_checkpoint,
                    expected_protocol="m6",
                ),
            }
            manifest = build_m7c_frozen_inputs_manifest(
                teacher_corpus_manifest=teacher_manifest_path,
                initialization_checkpoint=m7b_checkpoint,
                m6_baseline_checkpoint=m6_checkpoint,
                identity=identity,
            )
            imported = root / "imported"
            imported.mkdir()
            shutil.copytree(teacher_root, imported / M7C_FROZEN_TEACHER_DIRECTORY)
            shutil.copy2(m7b_checkpoint, imported / M7C_FROZEN_CHECKPOINT_NAME)
            shutil.copy2(m6_checkpoint, imported / M7C_FROZEN_M6_BASELINE_NAME)
            (imported / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            shutil.rmtree(teacher_root)
            verified = verify_m7c_frozen_inputs(imported, identity=identity)
        self.assertEqual(verified["identity"], identity)
        self.assertTrue(Path(verified["teacher_corpus_manifest"]).is_absolute())

    def test_unsafe_relative_paths_are_rejected(self) -> None:
        for path in ("", "../escape", "/absolute", "teacher/../../escape"):
            with self.assertRaises(ValueError):
                safe_relative_path(path)


if __name__ == "__main__":
    unittest.main()
