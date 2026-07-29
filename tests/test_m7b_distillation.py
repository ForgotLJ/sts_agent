from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys
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
from sts_env.training import (
    ImitationChunk,
    M7B_SUPERVISED_PHASES,
    M7BDistillationConfig,
    M7BDistillationProgress,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    build_imitation_chunks,
    build_m7b_corpus_manifest,
    build_m7b_replay_manifest,
    evaluate_m7b_imitation,
    load_m7b_checkpoint,
    load_m7b_replay_batch,
    phase_stratified_imitation_chunks,
    record_m7b_teacher_trace,
    save_m7b_checkpoint,
    save_m7b_replay_batch,
    train_m7b_chunk_batch,
    validate_m7b_training_objective,
    verify_m7b_corpus_manifest,
    verify_m7b_replay_manifest,
)
from sts_env.training.m7b_replay import sha256_file as replay_sha256_file


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


class MultiPhaseEnvironment:
    first_action = Action(
        ActionKind.CHOOSE_OPTION,
        source_id="first",
        option_type="first",
    )
    teacher_action = Action(
        ActionKind.CHOOSE_OPTION,
        source_id="teacher",
        option_type="teacher",
    )

    def __init__(self) -> None:
        self._index = 0
        self._observation = self._active_observation()

    @property
    def observation(self) -> Observation:
        return self._observation

    def reset(self, seed: int | None = None):
        self._index = 0
        self._observation = self._active_observation()
        return self._observation, {"backend": "m7b-test", "seed": seed}

    def step(self, action: int | Action):
        resolved = (
            self._observation.legal_actions[action]
            if isinstance(action, int)
            else action
        )
        if resolved not in self._observation.legal_actions:
            raise ValueError("illegal M7-B fixture action")
        self._index += 1
        terminated = self._index == len(M7B_SUPERVISED_PHASES)
        if terminated:
            self._observation = replace(
                self._observation,
                phase=Phase.TERMINAL,
                legal_actions=(),
                floor=self._index + 1,
            )
        else:
            self._observation = self._active_observation()
        return (
            self._observation,
            1.0 if terminated else 0.0,
            terminated,
            False,
            {"floor": self._index + 1},
        )

    def _active_observation(self) -> Observation:
        return Observation(
            phase=M7B_SUPERVISED_PHASES[self._index],
            turn=0,
            player=PlayerView(hp=80, max_hp=80, block=0, energy=0),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=(self.first_action, self.teacher_action),
            act=1,
            floor=self._index + 1,
        )


class CombatContextEnvironment:
    combat_actions = tuple(
        Action(
            ActionKind.CHOOSE_OPTION,
            source_id=f"combat-{index}",
            option_type=f"combat-{index}",
        )
        for index in range(6)
    )
    reward_actions = (
        Action(ActionKind.CHOOSE_CARD, source_id="card-a"),
        Action(ActionKind.CHOOSE_CARD, source_id="card-b"),
    )

    def __init__(self) -> None:
        self._index = 0
        self._observation = self._combat_observation()

    @property
    def observation(self) -> Observation:
        return self._observation

    def reset(self, seed: int | None = None):
        self._index = 0
        self._observation = self._combat_observation()
        return self._observation, {"backend": "m7b-combat-context", "seed": seed}

    def step(self, action: int | Action):
        resolved = (
            self._observation.legal_actions[action]
            if isinstance(action, int)
            else action
        )
        if resolved not in self._observation.legal_actions:
            raise ValueError("illegal combat-context fixture action")
        self._index += 1
        terminated = self._index == 2
        self._observation = Observation(
            phase=Phase.TERMINAL if terminated else Phase.CARD_REWARD,
            turn=0,
            player=PlayerView(hp=70, max_hp=80, block=0, energy=0),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=() if terminated else self.reward_actions,
            act=1,
            floor=2,
        )
        return self._observation, 0.0, terminated, False, {"floor": 2}

    @classmethod
    def trace(cls):
        environment = cls()
        observation, _ = environment.reset(seed=23)
        initial_digest = observation_digest(observation)
        steps = []
        for action in (cls.combat_actions[-1], cls.reward_actions[-1]):
            observation, reward, terminated, truncated, info = environment.step(action)
            steps.append(
                TraceStep(
                    action=action,
                    observation_digest=observation_digest(observation),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
        return EpisodeTrace(
            seed=23,
            initial_observation_digest=initial_digest,
            steps=tuple(steps),
            backend="m7b-combat-context",
        )

    @classmethod
    def _combat_observation(cls) -> Observation:
        return Observation(
            phase=Phase.COMBAT,
            turn=1,
            player=PlayerView(hp=80, max_hp=80, block=0, energy=3),
            hand=(),
            enemies=(),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
            legal_actions=cls.combat_actions,
            act=1,
            floor=1,
        )


def teacher_policy(_: Observation) -> Action:
    return MultiPhaseEnvironment.teacher_action


def make_chunk(
    trainer: RecurrentPPOTrainer,
    phase: Phase,
    *,
    length: int = 2,
) -> ImitationChunk:
    return ImitationChunk(
        states=torch.zeros(length, trainer.encoder.state_dimension),
        actions=torch.zeros(length, 2, trainer.encoder.action_dimension),
        action_masks=torch.ones(length, 2, dtype=torch.bool),
        chosen_actions=torch.ones(length, dtype=torch.long),
        supervision_weights=torch.ones(length),
        supervision_phases=torch.full(
            (length,),
            tuple(Phase).index(phase),
            dtype=torch.long,
        ),
    )


def write_corpus(root: Path, seed_start: int, seed_count: int) -> Path:
    traces = root / "traces"
    traces.mkdir(parents=True)
    for seed in range(seed_start, seed_start + seed_count):
        record_m7b_teacher_trace(
            MultiPhaseEnvironment(),
            teacher_policy,
            seed=seed,
            max_steps=10,
        ).write_jsonl(traces / f"seed-{seed:08d}.jsonl")
    manifest = build_m7b_corpus_manifest(
        root,
        seed_start=seed_start,
        seed_count=seed_count,
    )
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def imitation_evaluation(method: str, accuracy: float, cross_entropy: float):
    phases = {
        phase.value: {
            "count": 10,
            "correct": round(10 * accuracy),
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
            "cross_entropy_sum": cross_entropy * 10,
        }
        for phase in M7B_SUPERVISED_PHASES
    }
    return {
        "protocol": "m7b-imitation-evaluation",
        "method": method,
        "corpus_sha256": "same",
        "seed_range": [1_500_000, 1_500_511],
        "metrics": {
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
            "phases": phases,
        },
    }


def aggregate_method(seed_count: int) -> dict[str, object]:
    safety = {
        name: {"mean": 0.0, "standard_deviation": 0.0, "values": [0.0]}
        for name in (
            "errors",
            "crashes",
            "illegal_actions",
            "recovery_failures",
            "timeouts",
            "cycles",
        )
    }
    return {
        "unique_environment_seed_count": seed_count,
        "metrics": safety,
    }


class M7BDistillationTests(unittest.TestCase):
    def test_seed_ranges_are_disjoint_and_protect_final_test(self) -> None:
        config = M7BDistillationConfig(run_seed=17)
        self.assertEqual(config.seed_ranges()["training"].start, 400_000)
        self.assertEqual(config.seed_ranges()["final"].stop, 3_002_048)
        with self.assertRaisesRegex(ValueError, "overlap"):
            M7BDistillationConfig(
                run_seed=17,
                validation_seed_start=400_100,
            )

    def test_corpus_manifest_round_trip_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_corpus(root, 10, 2)
            verified = verify_m7b_corpus_manifest(
                manifest_path,
                expected_seed_start=10,
                expected_seed_count=2,
            )
            self.assertEqual(verified["trace_count"], 2)
            self.assertTrue(
                all(
                    verified["phase_supervision_counts"][phase.value] == 2
                    for phase in M7B_SUPERVISED_PHASES
                )
            )
            trace_path = root / "traces" / "seed-00000010.jsonl"
            trace_path.write_text(
                trace_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_m7b_corpus_manifest(manifest_path)

    def test_corpus_manifest_relocates_with_its_trace_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            manifest_path = write_corpus(source, 44, 3)
            portable = Path(directory) / "portable"
            portable.mkdir()
            shutil.copytree(source / "traces", portable / "traces")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["root"] = str(Path(directory) / "unavailable-original-root")
            (portable / "manifest.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            verified = verify_m7b_corpus_manifest(
                portable / "manifest.json",
                expected_seed_start=44,
                expected_seed_count=3,
            )
        self.assertEqual(Path(verified["root"]), portable)

    def test_phase_stratified_sampling_is_bounded(self) -> None:
        trainer = RecurrentPPOTrainer(
            RecurrentPPOConfig(
                recurrent_size=8,
                state_embedding_size=8,
                action_embedding_size=8,
                value_loss_weight=0.0,
                entropy_weight=0.0,
            )
        )
        chunks = (
            make_chunk(trainer, Phase.EVENT),
            make_chunk(trainer, Phase.EVENT),
            make_chunk(trainer, Phase.EVENT),
            make_chunk(trainer, Phase.SHOP),
        )
        sampled = phase_stratified_imitation_chunks(
            chunks,
            maximum_multiplier=2,
            seed=17,
        )
        event_index = tuple(Phase).index(Phase.EVENT)
        shop_index = tuple(Phase).index(Phase.SHOP)
        event_chunks = sum(
            bool(((chunk.supervision_phases == event_index) & (chunk.supervision_weights > 0)).any())
            for chunk in sampled
        )
        shop_chunks = sum(
            bool(((chunk.supervision_phases == shop_index) & (chunk.supervision_weights > 0)).any())
            for chunk in sampled
        )
        self.assertEqual(event_chunks, 3)
        self.assertEqual(shop_chunks, 2)

    def test_distillation_batch_updates_network_with_pure_cross_entropy(self) -> None:
        config = RecurrentPPOConfig(
            recurrent_size=8,
            state_embedding_size=8,
            action_embedding_size=8,
            value_loss_weight=0.0,
            entropy_weight=0.0,
            uniform_exploration_weight=0.0,
        )
        trainer = RecurrentPPOTrainer(config=config, seed=17)
        before = {
            name: value.detach().clone()
            for name, value in trainer.network.state_dict().items()
        }
        metric = train_m7b_chunk_batch(
            trainer,
            (make_chunk(trainer, Phase.EVENT),),
        )
        self.assertEqual(metric["phases"][Phase.EVENT.value]["count"], 2)
        self.assertEqual(trainer.gradient_steps, 1)
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in trainer.network.state_dict().items()
            )
        )
        with self.assertRaisesRegex(ValueError, "pure teacher cross-entropy"):
            validate_m7b_training_objective(replace(config, entropy_weight=0.01))

    def test_imitation_evaluation_reports_every_phase(self) -> None:
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            record_m7b_teacher_trace(
                MultiPhaseEnvironment(),
                teacher_policy,
                seed=11,
                max_steps=10,
            ).write_jsonl(path)
            metrics = evaluate_m7b_imitation(
                trainer,
                MultiPhaseEnvironment,
                (path,),
                trace_batch_size=1,
                optimizer_batch_chunks=2,
                chunk_length=8,
                burn_in_steps=0,
            )
        self.assertEqual(metrics["supervised_steps"], 5)
        self.assertTrue(
            all(metrics["phases"][phase.value]["count"] == 1 for phase in M7B_SUPERVISED_PHASES)
        )

    def test_sparse_context_keeps_supervision_and_reduces_action_padding(self) -> None:
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
        dense = build_imitation_chunks(
            CombatContextEnvironment,
            trainer,
            (CombatContextEnvironment.trace(),),
            chunk_length=8,
            burn_in_steps=0,
        )
        sparse = build_imitation_chunks(
            CombatContextEnvironment,
            trainer,
            (CombatContextEnvironment.trace(),),
            chunk_length=8,
            burn_in_steps=0,
            sparse_unsupervised_actions=True,
        )
        self.assertEqual(len(dense), 1)
        self.assertEqual(len(sparse), 1)
        self.assertTrue(torch.equal(dense[0].states, sparse[0].states))
        self.assertTrue(
            torch.equal(dense[0].supervision_weights, sparse[0].supervision_weights)
        )
        self.assertEqual(dense[0].actions.shape[1], 6)
        self.assertEqual(sparse[0].actions.shape[1], 2)
        self.assertEqual(int(sparse[0].chosen_actions[-1]), 1)

    def test_checkpoint_round_trip_preserves_progress(self) -> None:
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
        config = M7BDistillationConfig(run_seed=17)
        progress = M7BDistillationProgress(
            next_epoch=2,
            next_trace_batch=3,
            total_trace_batches_completed=9,
            epochs_without_improvement=1,
            best_validation_key=(0.5, 0.6, -1.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_m7b_checkpoint(
                path,
                trainer=trainer,
                config=config,
                progress=progress,
                metrics=({"epoch": 2},),
                manifest={"corpora": {"training": "a", "validation": "b"}},
            )
            loaded = load_m7b_checkpoint(path)
        self.assertEqual(loaded.progress, progress)
        self.assertEqual(loaded.config, config)
        self.assertEqual(loaded.metrics, ({"epoch": 2},))

    def test_replay_cache_round_trip_and_evaluation_match_raw_trace(self) -> None:
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
        trace = record_m7b_teacher_trace(
            MultiPhaseEnvironment(),
            teacher_policy,
            seed=11,
            max_steps=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.jsonl"
            trace.write_jsonl(trace_path)
            chunks = build_imitation_chunks(
                MultiPhaseEnvironment,
                trainer,
                (trace,),
                chunk_length=8,
                burn_in_steps=0,
                sparse_unsupervised_actions=True,
            )
            batch_path = root / "batch-0000.pt"
            encoder_config = trainer.encoder.config.to_dict()
            save_m7b_replay_batch(
                batch_path,
                chunks=chunks,
                corpus_sha256="corpus",
                trace_seeds=(11,),
                encoder_config=encoder_config,
                chunk_length=8,
                burn_in_steps=0,
            )
            loaded = load_m7b_replay_batch(
                batch_path,
                expected_corpus_sha256="corpus",
                expected_encoder_config=encoder_config,
                expected_chunk_length=8,
                expected_burn_in_steps=0,
            )
            self.assertTrue(torch.equal(loaded[0].states, chunks[0].states))
            entry = {
                "index": 0,
                "path": batch_path.name,
                "sha256": replay_sha256_file(batch_path),
                "size": batch_path.stat().st_size,
                "trace_count": 1,
                "trace_seeds": [11],
                "chunk_count": len(chunks),
                "supervised_steps": 5,
            }
            manifest = build_m7b_replay_manifest(
                root,
                corpus_manifest={
                    "aggregate_sha256": "corpus",
                    "seed_range": [11, 11],
                    "trace_count": 1,
                },
                entries=(entry,),
                trace_batch_size=1,
                encoder_config=encoder_config,
                chunk_length=8,
                burn_in_steps=0,
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            verify_m7b_replay_manifest(
                manifest_path,
                expected_corpus_sha256="corpus",
                expected_encoder_config=encoder_config,
                expected_chunk_length=8,
                expected_burn_in_steps=0,
            )
            raw_metrics = evaluate_m7b_imitation(
                trainer,
                MultiPhaseEnvironment,
                (trace_path,),
                trace_batch_size=1,
                optimizer_batch_chunks=2,
                chunk_length=8,
                burn_in_steps=0,
            )
            cached_metrics = evaluate_m7b_imitation(
                trainer,
                MultiPhaseEnvironment,
                (),
                trace_batch_size=1,
                optimizer_batch_chunks=2,
                chunk_length=8,
                burn_in_steps=0,
                replay_batch_paths=(batch_path,),
            )
        self.assertEqual(raw_metrics["supervised_steps"], cached_metrics["supervised_steps"])
        self.assertEqual(raw_metrics["accuracy"], cached_metrics["accuracy"])
        self.assertAlmostEqual(
            raw_metrics["cross_entropy"],
            cached_metrics["cross_entropy"],
            places=7,
        )

    def test_smoke_configuration_keeps_checkpoint_architecture(self) -> None:
        trainer_script = load_script(
            "train_m7b_test",
            "scripts/train-m7b-distillation.py",
        )
        experiment, ppo, _ = trainer_script.load_configuration(
            PROJECT_ROOT / "config" / "m7b_noncombat_distillation.json",
            17,
            True,
        )
        self.assertEqual(experiment.device, "cpu")
        self.assertEqual(experiment.max_epochs, 2)
        self.assertEqual(experiment.training_seed_count, 4096)
        self.assertEqual(ppo.recurrent_size, 128)
        self.assertEqual(ppo.state_embedding_size, 128)
        self.assertEqual(ppo.action_embedding_size, 128)

    def test_imitation_and_end_to_end_audits_enforce_both_gates(self) -> None:
        summary_script = load_script(
            "summarize_m7b_test",
            "scripts/summarize-m7b-imitation.py",
        )
        audit_script = load_script("audit_m7b_test", "scripts/audit-m7b.py")
        imitation = summary_script.summarize(
            imitation_evaluation("m6-initial", 0.4, 1.2),
            imitation_evaluation("m7b", 0.6, 0.8),
        )
        end_to_end = {
            "protocol": "m7",
            "aggregate": {
                "m6-initial": aggregate_method(512),
                "m7b": aggregate_method(512),
            },
            "paired_comparisons": {
                "m7b_minus_m6-initial": {
                    "environment_seed_count": 512,
                    "seed_range": [1_600_000, 1_600_511],
                    "metrics": {
                        "final_floor": {
                            "mean_difference": 1.0,
                            "hierarchical_bootstrap_ci95": [0.2, 1.8],
                        },
                        "act1_clear": {"mean_difference": 0.01},
                    },
                }
            },
        }
        result = audit_script.audit(
            imitation,
            end_to_end,
            candidate_method="m7b",
            baseline_method="m6-initial",
            gate_seed_start=1_600_000,
            gate_seed_count=512,
        )
        self.assertEqual(result["verdict"], "PASS")
        failed = audit_script.audit(
            imitation,
            {
                **end_to_end,
                "paired_comparisons": {
                    "m7b_minus_m6-initial": {
                        **end_to_end["paired_comparisons"]["m7b_minus_m6-initial"],
                        "metrics": {
                            "final_floor": {
                                "mean_difference": 0.1,
                                "hierarchical_bootstrap_ci95": [-0.5, 0.7],
                            },
                            "act1_clear": {"mean_difference": 0.01},
                        },
                    }
                },
            },
            candidate_method="m7b",
            baseline_method="m6-initial",
            gate_seed_start=1_600_000,
            gate_seed_count=512,
        )
        self.assertEqual(failed["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
