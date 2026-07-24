from __future__ import annotations

from pathlib import Path
import importlib.util
import tempfile
import unittest

from sts_env import Action, ActionKind, EpisodeTrace, TraceStep
from sts_env.training.teacher_corpus import (
    build_teacher_corpus_manifest,
    verify_teacher_corpus_manifest,
)


TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train-m6.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("train_m6_teacher_test", TRAIN_SCRIPT)
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise RuntimeError("could not load train-m6.py")
TRAIN_M6 = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN_M6)


class TeacherCorpusTests(unittest.TestCase):
    @staticmethod
    def write_trace(path: Path, seed: int) -> None:
        EpisodeTrace(
            seed=seed,
            initial_observation_digest="0" * 64,
            steps=(
                TraceStep(
                    action=Action(
                        kind=ActionKind.CHOOSE_OPTION,
                        source_id="neow:test",
                        choice_index=0,
                        label="Neow test",
                    ),
                    observation_digest="1" * 64,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={"floor": 0},
                ),
            ),
        ).write_jsonl(path)

    def test_manifest_verifies_exact_shared_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "act1_clear").mkdir()
            (root / "act2_clear").mkdir()
            self.write_trace(root / "act1_clear" / "one.jsonl", 60_000)
            self.write_trace(root / "act2_clear" / "two.jsonl", 60_001)

            manifest = build_teacher_corpus_manifest(
                root,
                {"act1_clear": 1, "act2_clear": 1},
            )
            verified = verify_teacher_corpus_manifest(manifest)

            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["validated_trace_count"], 2)
            self.assertEqual(verified["aggregate_sha256"], manifest["aggregate_sha256"])

            (root / "act1_clear" / "one.jsonl").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_teacher_corpus_manifest(manifest)

    def test_formal_run_reads_shared_teacher_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "act1_clear").mkdir()
            (root / "act2_clear").mkdir()
            self.write_trace(root / "act1_clear" / "one.jsonl", 60_000)
            self.write_trace(root / "act2_clear" / "two.jsonl", 60_001)

            act1 = TRAIN_M6.teacher_trace_paths(root / "run", "act1_clear", root)
            full_run = TRAIN_M6.teacher_trace_paths(root / "run", "full_run", root)

            self.assertEqual([path.name for path in act1], ["one.jsonl"])
            self.assertEqual([path.name for path in full_run], ["two.jsonl"])


if __name__ == "__main__":
    unittest.main()
