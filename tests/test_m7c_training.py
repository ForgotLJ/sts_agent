from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from unittest import mock

from sts_env.training import RecurrentPPOConfig, RecurrentPPOTrainer
from sts_env.training.m7c_training import (
    M7CDaggerTrainingConfig,
    M7CDaggerTrainingProgress,
    load_m7c_checkpoint,
    m7c_validation_selection_key,
    save_m7c_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    specification = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT / relative_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {relative_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def validation(accuracy: float, cross_entropy: float) -> dict[str, object]:
    phases = {
        phase: {
            "count": 2,
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
        }
        for phase in ("card_reward", "event", "map", "rest_site", "shop")
    }
    return {"phases": phases}


class M7CDaggerTrainingTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_round_progress(self) -> None:
        trainer = RecurrentPPOTrainer(
            RecurrentPPOConfig(
                recurrent_size=8,
                state_embedding_size=8,
                action_embedding_size=8,
                value_loss_weight=0.0,
                entropy_weight=0.0,
            ),
            seed=17,
        )
        config = M7CDaggerTrainingConfig(run_seed=17, round_index=1)
        progress = M7CDaggerTrainingProgress(
            next_epoch=1,
            next_trace_batch=3,
            total_trace_batches_completed=7,
            epochs_without_improvement=1,
            best_validation_key=(0.5, 0.5, 0.6, 0.6, -1.0, -1.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_m7c_checkpoint(
                path,
                trainer=trainer,
                config=config,
                progress=progress,
                metrics=({"epoch": 1},),
                manifest={"training_corpus_sha256": "fixture"},
            )
            loaded = load_m7c_checkpoint(path)
        self.assertEqual(loaded.config, config)
        self.assertEqual(loaded.progress, progress)
        self.assertEqual(loaded.metrics, ({"epoch": 1},))

    def test_selection_requires_anchor_and_on_policy_signal(self) -> None:
        anchor = validation(0.75, 0.4)
        on_policy = validation(0.5, 0.7)
        self.assertEqual(
            m7c_validation_selection_key(anchor, on_policy),
            (0.5, 0.75, 0.5, 0.75, -0.7, -0.4),
        )
        incomplete = {
            "phases": {
                "card_reward": {"count": 2, "accuracy": 0.5, "cross_entropy": 0.7},
                "event": {"count": 0, "accuracy": 0.0, "cross_entropy": 0.0},
            }
        }
        with self.assertRaisesRegex(ValueError, "cover every"):
            m7c_validation_selection_key(None, incomplete)
        self.assertEqual(
            m7c_validation_selection_key(
                None,
                incomplete,
                require_all_phases=False,
            ),
            (0.5, 0.5, -0.7),
        )

    def test_persistent_aggregate_keeps_teacher_and_repeats_dagger(self) -> None:
        train_script = load_script(
            "train_m7c_persistent_aggregate_test",
            "scripts/train-m7c-dagger.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_root = root / "teacher"
            dagger_root = root / "dagger"
            teacher_manifest = {
                "root": str(teacher_root),
                "files": [{"path": "traces/teacher.jsonl"}],
            }
            dagger_manifest = {
                "root": str(dagger_root),
                "files": [
                    {"path": "traces/dagger-a.jsonl"},
                    {"path": "traces/dagger-b.jsonl"},
                ],
            }
            units = train_script.persistent_training_units(
                teacher_manifest=teacher_manifest,
                train_entries=(("dagger_round_0", root / "manifest.json", dagger_manifest),),
                dagger_batch_repeat=4,
                maximum_teacher_traces=1,
            )
        self.assertEqual(units[0], ("teacher", teacher_root / "traces/teacher.jsonl"))
        self.assertEqual(len(units), 9)
        self.assertEqual([source for source, _ in units].count("teacher"), 1)
        self.assertEqual([source for source, _ in units].count("dagger"), 8)
        self.assertEqual(
            [path.name for source, path in units if source == "dagger"],
            [
                "dagger-a.jsonl",
                "dagger-b.jsonl",
                "dagger-a.jsonl",
                "dagger-b.jsonl",
                "dagger-a.jsonl",
                "dagger-b.jsonl",
                "dagger-a.jsonl",
                "dagger-b.jsonl",
            ],
        )

    def test_frozen_teacher_identity_rejects_replacement(self) -> None:
        train_script = load_script(
            "train_m7c_frozen_teacher_test",
            "scripts/train-m7c-dagger.py",
        )
        expected = {
            "seed_start": 400_000,
            "seed_count": 4_096,
            "aggregate_sha256": "a" * 64,
        }
        manifest = {
            "seed_range": [400_000, 404_095],
            "trace_count": 4_096,
            "aggregate_sha256": "a" * 64,
        }
        identity = train_script.validate_frozen_teacher_manifest(
            manifest,
            expected=expected,
        )
        self.assertEqual(identity["seed_range"], [400_000, 404_095])
        with self.assertRaisesRegex(ValueError, "hash differs"):
            train_script.validate_frozen_teacher_manifest(
                {**manifest, "aggregate_sha256": "b" * 64},
                expected=expected,
            )
        with self.assertRaisesRegex(ValueError, "unexpected seed range"):
            train_script.validate_frozen_teacher_manifest(
                {**manifest, "seed_range": [400_001, 404_096]},
                expected=expected,
            )

    def test_configuration_rejects_zero_dagger_repeat(self) -> None:
        train_script = load_script(
            "train_m7c_configuration_test",
            "scripts/train-m7c-dagger.py",
        )
        configuration_path = PROJECT_ROOT / "config" / "m7c_dagger_control.json"
        payload = __import__("json").loads(configuration_path.read_text(encoding="utf-8"))
        payload["experiment"]["dagger_batch_repeat"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-config.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch repeat"):
                train_script.load_configuration(
                    path,
                    run_seed=17,
                    round_index=0,
                    smoke=True,
                )

    def test_configuration_rejects_zero_smoke_teacher_limit(self) -> None:
        train_script = load_script(
            "train_m7c_smoke_limit_configuration_test",
            "scripts/train-m7c-dagger.py",
        )
        configuration_path = PROJECT_ROOT / "config" / "m7c_dagger_control.json"
        payload = __import__("json").loads(configuration_path.read_text(encoding="utf-8"))
        payload["experiment"]["smoke_teacher_trace_limit"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-config.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke teacher limit"):
                train_script.load_configuration(
                    path,
                    run_seed=17,
                    round_index=0,
                    smoke=True,
                )

    def test_configuration_rejects_wrong_teacher_anchor_horizon(self) -> None:
        train_script = load_script(
            "train_m7c_anchor_horizon_configuration_test",
            "scripts/train-m7c-dagger.py",
        )
        configuration_path = PROJECT_ROOT / "config" / "m7c_dagger_control.json"
        payload = __import__("json").loads(configuration_path.read_text(encoding="utf-8"))
        payload["experiment"]["teacher_anchor_max_steps"] = 1_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-config.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "teacher anchor horizon"):
                train_script.load_configuration(
                    path,
                    run_seed=17,
                    round_index=0,
                    smoke=True,
                )

    def test_configuration_rejects_wrong_anchor_truncation_policy(self) -> None:
        train_script = load_script(
            "train_m7c_anchor_truncation_configuration_test",
            "scripts/train-m7c-dagger.py",
        )
        configuration_path = PROJECT_ROOT / "config" / "m7c_dagger_control.json"
        payload = __import__("json").loads(configuration_path.read_text(encoding="utf-8"))
        payload["experiment"]["teacher_anchor_allow_horizon_truncation"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-config.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncation policy"):
                train_script.load_configuration(
                    path,
                    run_seed=17,
                    round_index=0,
                    smoke=True,
                )

    def test_teacher_anchor_manifest_requires_configured_horizon(self) -> None:
        train_script = load_script(
            "train_m7c_anchor_manifest_test",
            "scripts/train-m7c-dagger.py",
        )
        with mock.patch.object(
            train_script,
            "verify_m7b_corpus_manifest",
            return_value={"collection_max_steps": 5_000},
        ) as verify:
            train_script.resolve_teacher_anchor_manifest(
                "teacher_anchor=/tmp/anchor-manifest.json",
                expected_collection_max_steps=5_000,
                expected_allow_horizon_truncation=True,
            )
        self.assertEqual(
            verify.call_args.kwargs["expected_collection_max_steps"],
            5_000,
        )
        self.assertIs(
            verify.call_args.kwargs["expected_allow_horizon_truncation"],
            True,
        )

    def test_formal_round_initialization_is_frozen(self) -> None:
        train_script = load_script(
            "train_m7c_initialization_test",
            "scripts/train-m7c-dagger.py",
        )
        checkpoint = {
            "protocol": "m7b",
            "run_seed": 17,
            "sha256": "a" * 64,
        }
        train_script.validate_round_initialization(
            {**checkpoint, "round_index": None, "completed": None},
            round_index=0,
            expected_initial_checkpoint=checkpoint,
            smoke=False,
        )
        with self.assertRaisesRegex(ValueError, "frozen M7-B"):
            train_script.validate_round_initialization(
                {**checkpoint, "sha256": "b" * 64},
                round_index=0,
                expected_initial_checkpoint=checkpoint,
                smoke=False,
            )
        train_script.validate_round_initialization(
            {
                "protocol": "m7c-dagger",
                "run_seed": 17,
                "round_index": 0,
                "evaluation_only": True,
            },
            round_index=1,
            expected_initial_checkpoint=checkpoint,
            smoke=False,
        )
        with self.assertRaisesRegex(ValueError, "prior round"):
            train_script.validate_round_initialization(
                {
                    "protocol": "m7c-dagger",
                    "run_seed": 17,
                    "round_index": 0,
                    "evaluation_only": False,
                },
                round_index=1,
                expected_initial_checkpoint=checkpoint,
                smoke=False,
            )

    def test_round_corpus_requires_matching_behavior_checkpoint(self) -> None:
        train_script = load_script(
            "train_m7c_behavior_provenance_test",
            "scripts/train-m7c-dagger.py",
        )
        initialization = {"sha256": "a" * 64}
        train_script.validate_behavior_checkpoint(
            {"behavior_policy": {"checkpoint_sha256": "a" * 64}},
            initialization=initialization,
            label="dagger_round_0",
        )
        with self.assertRaisesRegex(ValueError, "not collected"):
            train_script.validate_behavior_checkpoint(
                {"behavior_policy": {"checkpoint_sha256": "b" * 64}},
                initialization=initialization,
                label="dagger_round_0",
            )

    def test_promotion_audit_requires_both_baselines(self) -> None:
        audit_script = load_script("audit_m7c_test", "scripts/audit-m7c.py")
        safety = {
            name: {"mean": 0.0}
            for name in (
                "errors",
                "crashes",
                "illegal_actions",
                "recovery_failures",
                "timeouts",
                "cycles",
            )
        }
        aggregate = {
            method: {"unique_environment_seed_count": 512, "metrics": safety}
            for method in ("m7c-dagger", "m6-initial", "heuristic")
        }
        comparison = {
            "environment_seed_count": 512,
            "seed_range": [2_220_000, 2_220_511],
            "metrics": {
                "final_floor": {
                    "mean_difference": 1.0,
                    "hierarchical_bootstrap_ci95": [0.1, 1.5],
                },
                "act1_clear": {"mean_difference": 0.01},
            },
        }
        result = audit_script.audit(
            {
                "protocol": "m7",
                "aggregate": aggregate,
                "paired_comparisons": {
                    "m7c-dagger_minus_m6-initial": comparison,
                    "m7c-dagger_minus_heuristic": comparison,
                },
            },
            candidate_method="m7c-dagger",
            m6_baseline_method="m6-initial",
            heuristic_baseline_method="heuristic",
            gate_seed_start=2_220_000,
            gate_seed_count=512,
        )
        self.assertEqual(result["verdict"], "PASS")
        failing = audit_script.audit(
            {
                "protocol": "m7",
                "aggregate": aggregate,
                "paired_comparisons": {
                    "m7c-dagger_minus_m6-initial": comparison,
                    "m7c-dagger_minus_heuristic": {
                        **comparison,
                        "metrics": {
                            **comparison["metrics"],
                            "act1_clear": {"mean_difference": -0.03},
                        },
                    },
                },
            },
            candidate_method="m7c-dagger",
            m6_baseline_method="m6-initial",
            heuristic_baseline_method="heuristic",
            gate_seed_start=2_220_000,
            gate_seed_count=512,
        )
        self.assertEqual(failing["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
