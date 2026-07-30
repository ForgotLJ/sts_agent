from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

from sts_env import Phase
from sts_env.training import (
    CurriculumScheduler,
    CurriculumSpec,
    ImitationChunk,
    M7TrainingConfig,
    M7TrainingProgress,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    balance_imitation_phase_weights,
    imitation_phase_coverage,
    load_m7_checkpoint,
    m7_validation_selection_key,
    save_m7_checkpoint,
    summarize_m7_evaluations,
    validate_m7_fixed_budget_progress,
    validate_m7_evaluation_seed_range,
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


def evaluation(
    method: str,
    *,
    policy_seed: int,
    run_seed: int | None,
    floor_delta: int,
    won_seed: int | None = None,
) -> dict[str, object]:
    episodes = [
        {
            "seed": seed,
            "won": seed == won_seed,
            "final_act": 2,
            "final_floor": 10 + seed - 7_000 + floor_delta,
            "final_hp": 20 + floor_delta,
            "proxy_score": 1.0 + floor_delta,
            "decisions": 50 + floor_delta,
            "simulator_calls": 0,
            "wall_seconds": 1.0,
        }
        for seed in range(7_000, 7_004)
    ]
    return {
        "method": method,
        "run_seed": run_seed,
        "policy_seed": policy_seed,
        "summary": {
            "episodes": episodes,
            "win_rate": sum(episode["won"] for episode in episodes) / len(episodes),
            "act1_clear_rate": 0.5,
            "act2_clear_rate": 0.25,
            "act3_clear_rate": 0.0,
            "mean_floor": sum(episode["final_floor"] for episode in episodes)
            / len(episodes),
            "median_floor": 11.5 + floor_delta,
            "mean_final_hp": 20.0 + floor_delta,
            "mean_proxy_score": 1.0 + floor_delta,
            "mean_decisions": 50.0 + floor_delta,
            "total_simulator_calls": 0,
            "total_wall_seconds": 4.0,
            "errors": 0,
            "crashes": 0,
            "illegal_actions": 0,
            "recovery_failures": 0,
            "truncations": 0,
            "timeouts": 0,
            "cycles": 0,
        },
    }


class M7PipelineTests(unittest.TestCase):
    def test_seed_protocol_rotates_screening_and_protects_final_range(self) -> None:
        config = M7TrainingConfig(
            run_seed=17,
            screening_seed_count=8,
            screening_batch_size=4,
        )

        self.assertEqual(
            config.screening_seeds(0),
            (1_210_000, 1_210_001, 1_210_002, 1_210_003),
        )
        self.assertEqual(
            config.screening_seeds(1),
            (1_210_004, 1_210_005, 1_210_006, 1_210_007),
        )
        self.assertEqual(config.screening_seeds(2), config.screening_seeds(0))
        self.assertEqual(
            config.pilot_gate_seeds()[:2],
            (1_400_000, 1_400_001),
        )
        self.assertEqual(
            validate_m7_evaluation_seed_range(3_000_000, 2_048, final=True),
            (3_000_000, 3_002_047),
        )
        with self.assertRaises(ValueError):
            validate_m7_evaluation_seed_range(3_000_000, 2_048, final=False)
        with self.assertRaises(ValueError):
            M7TrainingConfig(
                run_seed=17,
                promotion_seed_start=1_210_000,
            )

    def test_selection_key_prioritizes_progress_before_mean_floor(self) -> None:
        shallow_win = {
            "win_rate": 0.01,
            "act3_clear_rate": 0.01,
            "act2_clear_rate": 0.01,
            "act1_clear_rate": 0.01,
            "mean_floor": 1.0,
            "mean_proxy_score": -1.0,
        }
        deep_loss = {
            "win_rate": 0.0,
            "act3_clear_rate": 0.5,
            "act2_clear_rate": 0.5,
            "act1_clear_rate": 0.5,
            "mean_floor": 40.0,
            "mean_proxy_score": 1.0,
        }

        self.assertGreater(
            m7_validation_selection_key(shallow_win),
            m7_validation_selection_key(deep_loss),
        )

    def test_m7_checkpoint_round_trip_preserves_fixed_budget_progress(self) -> None:
        trainer = RecurrentPPOTrainer(
            config=RecurrentPPOConfig(
                recurrent_size=16,
                state_embedding_size=16,
                action_embedding_size=16,
                update_epochs=1,
                minibatch_environments=1,
            ),
            seed=17,
        )
        scheduler = CurriculumScheduler(
            stages=(
                CurriculumSpec(
                    "full_run",
                    completion_reward=0.0,
                    potential_scale=0.0,
                    progress_reward_per_floor=0.0,
                ),
            )
        )
        config = M7TrainingConfig(run_seed=17, full_run_updates=10)
        progress = M7TrainingProgress(
            full_run_entry_update=50,
            full_run_updates_completed=7,
            screening_batches_completed=2,
            selection_evaluations_completed=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_m7_checkpoint(
                path,
                trainer=trainer,
                collector_state={"fixture": True},
                scheduler=scheduler,
                config=config,
                progress=progress,
                update_index=57,
                metrics=({"update": 57},),
                manifest={"protocol": "m7"},
            )
            loaded = load_m7_checkpoint(path)

        self.assertEqual(loaded.progress, progress)
        self.assertEqual(loaded.config.full_run_updates, 10)
        self.assertEqual(loaded.update_index, 57)
        self.assertEqual(loaded.collector_state, {"fixture": True})

    def test_phase_balancing_equalizes_supervision_without_dropping_steps(self) -> None:
        phases = tuple(Phase)
        chunk = ImitationChunk(
            states=torch.zeros(3, 2),
            actions=torch.zeros(3, 2, 2),
            action_masks=torch.ones(3, 2, dtype=torch.bool),
            chosen_actions=torch.zeros(3, dtype=torch.long),
            supervision_weights=torch.ones(3),
            supervision_phases=torch.tensor(
                [
                    phases.index(Phase.EVENT),
                    phases.index(Phase.EVENT),
                    phases.index(Phase.SHOP),
                ]
            ),
        )

        balanced = balance_imitation_phase_weights((chunk,), maximum_multiplier=4.0)
        coverage = imitation_phase_coverage(balanced)

        self.assertEqual(coverage[Phase.EVENT.value]["steps"], 2.0)
        self.assertEqual(coverage[Phase.SHOP.value]["steps"], 1.0)
        self.assertAlmostEqual(coverage[Phase.EVENT.value]["weight"], 1.5)
        self.assertAlmostEqual(coverage[Phase.SHOP.value]["weight"], 1.5)

    def test_m7_reporting_uses_unique_seeds_and_hierarchical_pairing(self) -> None:
        evaluations = (
            evaluation("heuristic", policy_seed=17, run_seed=None, floor_delta=0),
            evaluation("heuristic", policy_seed=29, run_seed=None, floor_delta=0),
            evaluation("heuristic-search", policy_seed=17, run_seed=None, floor_delta=-1),
            evaluation("heuristic-search", policy_seed=29, run_seed=None, floor_delta=-1),
            evaluation("learned-heuristic", policy_seed=17, run_seed=17, floor_delta=1),
            evaluation(
                "learned-heuristic",
                policy_seed=29,
                run_seed=29,
                floor_delta=2,
                won_seed=7_003,
            ),
        )

        summary = summarize_m7_evaluations(
            evaluations,
            bootstrap_samples=200,
        )

        heuristic = summary["aggregate"]["heuristic"]
        learned = summary["aggregate"]["learned-heuristic"]
        comparison = summary["paired_comparisons"][
            "learned-heuristic_minus_heuristic"
        ]
        self.assertEqual(heuristic["episode_record_count"], 8)
        self.assertEqual(heuristic["unique_environment_seed_count"], 4)
        self.assertEqual(learned["record_wins"], 1)
        self.assertEqual(learned["unique_environment_wins"], 1)
        self.assertEqual(comparison["training_run_count"], 2)
        self.assertEqual(comparison["environment_seed_count"], 4)
        self.assertEqual(
            comparison["metrics"]["final_floor"]["mean_difference"],
            1.5,
        )
        self.assertGreater(
            comparison["metrics"]["final_floor"]["hierarchical_bootstrap_ci95"][0],
            0,
        )
        self.assertEqual(len(summary["warnings"]), 2)
        self.assertEqual(
            summary["paired_comparisons"]["heuristic-search_minus_heuristic"]
            ["metrics"]["final_floor"]["mean_difference"],
            -1.0,
        )

    def test_m7_reporting_supports_multiple_reference_methods(self) -> None:
        evaluations = (
            evaluation("heuristic", policy_seed=17, run_seed=None, floor_delta=0),
            evaluation("m6-initial", policy_seed=17, run_seed=17, floor_delta=-1),
            evaluation("m7c-dagger", policy_seed=17, run_seed=17, floor_delta=1),
        )

        summary = summarize_m7_evaluations(
            evaluations,
            reference_methods=("m6-initial", "heuristic"),
            bootstrap_samples=100,
        )

        self.assertEqual(summary["reference_method"], "m6-initial")
        self.assertEqual(
            summary["reference_methods"],
            ["m6-initial", "heuristic"],
        )
        self.assertIn(
            "m7c-dagger_minus_m6-initial",
            summary["paired_comparisons"],
        )
        self.assertIn(
            "m7c-dagger_minus_heuristic",
            summary["paired_comparisons"],
        )

        with self.assertRaisesRegex(ValueError, "configuration is invalid"):
            summarize_m7_evaluations(
                evaluations,
                reference_methods=("heuristic", "heuristic"),
                bootstrap_samples=100,
            )

    def test_training_entry_resolves_smoke_budget_without_changing_source_config(self) -> None:
        trainer_script = load_script("train_m7_test", "scripts/train-m7.py")

        resolved = trainer_script.load_configuration(
            PROJECT_ROOT / "config" / "m7_recurrent_ppo_pilot.json",
            17,
            True,
        )
        experiment = resolved[0]
        phase_balance = resolved[6]

        self.assertEqual(experiment.device, "cpu")
        self.assertEqual(experiment.full_run_updates, 2)
        self.assertEqual(experiment.selection_interval, 1)
        self.assertTrue(phase_balance["enabled"])

    def test_training_entry_honors_direct_full_run_start_stage(self) -> None:
        trainer_script = load_script("train_m7_stage_test", "scripts/train-m7.py")
        pilot = trainer_script.load_configuration(
            PROJECT_ROOT / "config" / "m7_recurrent_ppo_pilot.json",
            17,
            False,
        )
        formal = trainer_script.load_configuration(
            PROJECT_ROOT / "config" / "m7_recurrent_ppo.json",
            17,
            False,
        )

        pilot_stages = trainer_script.resolve_curriculum_stages(pilot[2], smoke=False)
        formal_stages = trainer_script.resolve_curriculum_stages(formal[2], smoke=False)

        self.assertEqual(tuple(stage.name for stage in pilot_stages), ("full_run",))
        self.assertEqual(formal_stages[0].name, "act1_floor6")
        self.assertEqual(formal_stages[-1].name, "full_run")

    def test_evaluation_report_label_does_not_change_policy_method(self) -> None:
        evaluation_script = load_script("evaluate_m7_label_test", "scripts/evaluate-m7.py")

        self.assertEqual(
            evaluation_script.report_method_name("learned-heuristic", "m7-balanced"),
            "m7-balanced",
        )
        self.assertEqual(
            evaluation_script.report_method_name("learned-heuristic", None),
            "learned-heuristic",
        )

    def test_component_diagnostic_cache_requires_matching_successful_result(self) -> None:
        diagnostic_script = load_script(
            "m7_component_diagnostic_test",
            "scripts/run-m7-component-diagnostics.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learned-heuristic.json"
            self.assertFalse(
                diagnostic_script.cached_evaluation_is_valid(
                    path,
                    run_seed=17,
                    seed_start=2_000_000,
                    seed_count=16,
                )
            )
            path.write_text("{not json", encoding="utf-8")
            self.assertFalse(
                diagnostic_script.cached_evaluation_is_valid(
                    path,
                    run_seed=17,
                    seed_start=2_000_000,
                    seed_count=16,
                )
            )
            path.write_text(
                '{"method":"learned-heuristic","run_seed":17,'
                '"seed_range":[2000000,2000015],"summary":{"errors":0}}',
                encoding="utf-8",
            )
            self.assertTrue(
                diagnostic_script.cached_evaluation_is_valid(
                    path,
                    run_seed=17,
                    seed_start=2_000_000,
                    seed_count=16,
                )
            )
            path.write_text(
                '{"method":"learned-heuristic","run_seed":17,'
                '"seed_range":[2000000,2000015],"summary":{"errors":1}}',
                encoding="utf-8",
            )
            self.assertFalse(
                diagnostic_script.cached_evaluation_is_valid(
                    path,
                    run_seed=17,
                    seed_start=2_000_000,
                    seed_count=16,
                )
            )

    def test_freeze_rejects_completion_checkpoint_before_fixed_budget(self) -> None:
        freeze_script = load_script("freeze_m7_test", "scripts/freeze-m7.py")
        config = M7TrainingConfig(run_seed=17, full_run_updates=10)
        completed = SimpleNamespace(
            config=config,
            manifest={"source_sha256": "source"},
            scheduler=SimpleNamespace(current=SimpleNamespace(name="full_run")),
            progress=M7TrainingProgress(
                full_run_entry_update=20,
                full_run_updates_completed=9,
            ),
            update_index=29,
        )

        with self.assertRaisesRegex(ValueError, "exhausted its fixed budget"):
            freeze_script.validate_completion_checkpoint(
                completed,
                run_seed=17,
                source_sha256="source",
            )

        completed.progress = M7TrainingProgress(
            full_run_entry_update=20,
            full_run_updates_completed=10,
        )
        completed.update_index = 30
        freeze_script.validate_completion_checkpoint(
            completed,
            run_seed=17,
            source_sha256="source",
        )

    def test_fixed_budget_progress_rejects_inconsistent_global_update(self) -> None:
        config = M7TrainingConfig(run_seed=17, full_run_updates=10)
        progress = M7TrainingProgress(
            full_run_entry_update=20,
            full_run_updates_completed=10,
        )
        with self.assertRaisesRegex(ValueError, "update count disagrees"):
            validate_m7_fixed_budget_progress(config, progress, 29)


if __name__ == "__main__":
    unittest.main()
