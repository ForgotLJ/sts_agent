from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.tensorboard import SummaryWriter

from sts_env import EpisodeTrace
from sts_env.training import (
    CurriculumEnvironmentFactory,
    CurriculumScheduler,
    DEFAULT_CURRICULUM,
    DaggerConfig,
    HeuristicPolicy,
    M7_FINAL_SEED_END,
    M7_FINAL_SEED_START,
    M7TrainingConfig,
    M7TrainingProgress,
    MultiprocessRecurrentRolloutCollector,
    ParameterEMA,
    ParameterEMAConfig,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    SelfImitationConfig,
    SubprocessVectorEnvironment,
    balance_imitation_phase_weights,
    build_imitation_chunks,
    collect_dagger_chunks,
    dagger_training_seeds,
    imitation_phase_coverage,
    is_self_imitation_candidate,
    load_m6_checkpoint,
    load_m7_checkpoint,
    m7_validation_selection_key,
    save_m7_checkpoint,
    train_self_imitation,
)
from sts_env.training.experiment import build_runtime_manifest
from sts_env.training.m7_corpus import (
    build_m7_prefix_corpus,
    load_m7_imitation_traces,
    m7_teacher_trace_paths,
    prune_m7_candidate_paths,
    stage_promotion_threshold,
)
from sts_env.training.parallel import LightspeedEnvironmentFactory
from sts_env.training.teacher_corpus import verify_teacher_corpus_manifest
from sts_env.training.m7_validation import (
    compact_full_run_summary,
    evaluate_m7_curriculum_stage,
    evaluate_m7_full_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the M7 recurrent full-run agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "m7_recurrent_ppo_pilot.json",
    )
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m7_recurrent_ppo_pilot",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="initialize network weights from an M6 or M7 checkpoint without optimizer state",
    )
    parser.add_argument("--source-freeze", type=Path)
    parser.add_argument(
        "--stop-after-update",
        type=int,
        help="stop after this global update while preserving the configured stage budgets",
    )
    parser.add_argument(
        "--reset-collector",
        action="store_true",
        help="reset worker environments when resuming after an intentional schema migration",
    )
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_freeze(path: Path, config_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "m7" or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported M7 source freeze")
    current = build_runtime_manifest(PROJECT_ROOT)
    frozen_source = dict(payload.get("runtime_manifest") or {}).get("source_sha256")
    if current["source_sha256"] != frozen_source:
        raise ValueError("current source tree differs from the M7 source freeze")
    frozen_config = dict(payload.get("config") or {})
    if str(config_path.resolve()) != frozen_config.get("path"):
        raise ValueError("training config path differs from the M7 source freeze")
    if sha256_file(config_path) != frozen_config.get("sha256"):
        raise ValueError("training config differs from the M7 source freeze")
    teacher_payload = dict(payload.get("teacher_corpus") or {})
    teacher_corpus = verify_teacher_corpus_manifest(teacher_payload)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_sha256": str(frozen_source),
        "config_sha256": str(frozen_config["sha256"]),
        "teacher_corpus_path": teacher_corpus["corpus_path"],
        "teacher_corpus_sha256": teacher_corpus["aggregate_sha256"],
    }


def load_configuration(path: Path, run_seed: int, smoke: bool) -> tuple[Any, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "m7":
        raise ValueError("M7 config must declare protocol=m7")
    experiment_payload = dict(payload["experiment"])
    run_seeds = tuple(int(seed) for seed in experiment_payload.pop("run_seeds"))
    if run_seed not in run_seeds:
        raise ValueError(f"run seed {run_seed} is not configured: {run_seeds}")
    experiment = M7TrainingConfig(run_seed=run_seed, **experiment_payload)
    ppo = RecurrentPPOConfig(**payload["ppo"])
    curriculum = dict(payload["curriculum"])
    start_stage = str(curriculum.get("start_stage", DEFAULT_CURRICULUM[0].name))
    stage_names = tuple(stage.name for stage in DEFAULT_CURRICULUM)
    if start_stage not in stage_names:
        raise ValueError(f"unsupported M7 curriculum start stage: {start_stage}")
    self_imitation = SelfImitationConfig(**payload.get("self_imitation", {}))
    parameter_ema = ParameterEMAConfig(**payload.get("parameter_ema", {}))
    dagger = DaggerConfig(**payload.get("dagger", {}))
    phase_balance = dict(payload.get("phase_balance", {}))
    phase_balance.setdefault("enabled", False)
    phase_balance.setdefault("maximum_multiplier", 4.0)
    if float(phase_balance["maximum_multiplier"]) < 1:
        raise ValueError("M7 phase-balance multiplier must be at least one")
    if smoke:
        experiment = replace(
            experiment,
            num_environments=2,
            rollout_steps=8,
            max_curriculum_updates=1,
            full_run_updates=2,
            checkpoint_interval=1,
            validation_interval=1,
            selection_interval=1,
            promotion_seed_count=4,
            screening_seed_count=8,
            screening_batch_size=4,
            selection_seed_count=4,
            device="cpu",
        )
        ppo = replace(
            ppo,
            recurrent_size=32,
            state_embedding_size=32,
            action_embedding_size=32,
            update_epochs=1,
            minibatch_environments=2,
        )
        curriculum["max_episode_steps"] = min(
            200,
            int(curriculum["max_episode_steps"]),
        )
        self_imitation = replace(
            self_imitation,
            minimum_traces=1,
            maximum_traces=2,
            maximum_candidate_traces=2,
            frontier_trace_repeats=1,
            corpus_capacity=max(2, self_imitation.corpus_capacity),
        )
        dagger = replace(
            dagger,
            interval_updates=1,
            rounds=1,
            stage_rounds={"full_run": 1},
            episodes=2,
            max_steps=min(100, dagger.max_steps),
            epochs=1,
        )
    return (
        experiment,
        ppo,
        curriculum,
        self_imitation,
        parameter_ema,
        dagger,
        phase_balance,
        run_seeds,
    )


def resolve_curriculum_stages(
    curriculum_config: dict[str, Any],
    *,
    smoke: bool,
) -> tuple[Any, ...]:
    start_stage = (
        "full_run"
        if smoke
        else str(curriculum_config.get("start_stage", DEFAULT_CURRICULUM[0].name))
    )
    stage_names = tuple(stage.name for stage in DEFAULT_CURRICULUM)
    if start_stage not in stage_names:
        raise ValueError(f"unsupported M7 curriculum start stage: {start_stage}")
    start_index = stage_names.index(start_stage)
    return tuple(
        replace(
            stage,
            max_episode_steps=int(curriculum_config["max_episode_steps"]),
        )
        for stage in DEFAULT_CURRICULUM[start_index:]
    )


def stage_factory(
    scheduler: CurriculumScheduler,
    run_directory: Path,
) -> CurriculumEnvironmentFactory:
    from sts_env.training import PrefixCorpus

    spec = scheduler.current
    corpus = None
    if spec.use_prefix_starts:
        corpus = PrefixCorpus.read(run_directory / "curriculum" / f"act-{spec.start_act}")
    return CurriculumEnvironmentFactory(spec=spec, prefix_corpus=corpus)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"unavailable: {result.stderr.strip()}"]
    return [line for line in result.stdout.splitlines() if line]


def initialize_network_from_checkpoint(
    trainer: RecurrentPPOTrainer,
    path: Path,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") == "m7":
        loaded = load_m7_checkpoint(path, device="cpu")
        protocol = "m7"
        update = loaded.update_index
        run_seed = loaded.config.run_seed
        source = loaded.trainer
    else:
        loaded_m6 = load_m6_checkpoint(path, device="cpu")
        protocol = "m6"
        update = loaded_m6.update_index
        run_seed = loaded_m6.config.run_seed
        source = loaded_m6.trainer
    structural_fields = (
        "recurrent_size",
        "state_embedding_size",
        "action_embedding_size",
    )
    if any(
        getattr(source.config, field) != getattr(trainer.config, field)
        for field in structural_fields
    ):
        raise ValueError("initialization checkpoint network architecture differs")
    trainer.network.load_state_dict(source.network.state_dict())
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "protocol": protocol,
        "run_seed": run_seed,
        "update": update,
        "optimizer_restored": False,
    }


def main() -> int:
    args = parse_args()
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("resume and initialize-from are mutually exclusive")
    if args.torch_threads <= 0 or args.torch_interop_threads <= 0:
        raise ValueError("PyTorch thread counts must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    source_freeze = (
        verify_source_freeze(args.source_freeze, args.config)
        if args.source_freeze is not None
        else None
    )
    shared_teacher_corpus = (
        Path(source_freeze["teacher_corpus_path"])
        if source_freeze is not None
        else None
    )
    (
        experiment,
        ppo_config,
        curriculum_config,
        self_imitation_config,
        parameter_ema_config,
        dagger_config,
        phase_balance_config,
        run_seeds,
    ) = load_configuration(args.config, args.run_seed, args.smoke)
    if experiment.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M7 CUDA training requested but PyTorch cannot access a GPU")

    run_directory = args.output / f"seed-{args.run_seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "checkpoint.pt"
    evaluation_checkpoint_path = run_directory / "evaluation-checkpoint.pt"
    best_evaluation_checkpoint_path = run_directory / "best-evaluation-checkpoint.pt"
    best_validation_path = run_directory / "best-validation.json"
    metrics_path = run_directory / "metrics.jsonl"
    configured_stages = resolve_curriculum_stages(
        curriculum_config,
        smoke=args.smoke,
    )

    initialization = None
    if args.resume is not None:
        loaded = load_m7_checkpoint(args.resume, device=experiment.device)
        if loaded.manifest.get("evaluation_only"):
            raise ValueError("evaluation-only M7 checkpoints cannot resume training")
        if loaded.config.run_seed != experiment.run_seed:
            raise ValueError("resume checkpoint belongs to a different run seed")
        trainer = loaded.trainer
        structural_fields = (
            "recurrent_size",
            "state_embedding_size",
            "action_embedding_size",
        )
        if any(
            getattr(trainer.config, field) != getattr(ppo_config, field)
            for field in structural_fields
        ):
            raise ValueError("resume configuration changes the M7 network architecture")
        trainer.config = ppo_config
        trainer.network.config = ppo_config
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = ppo_config.learning_rate
        scheduler = loaded.scheduler
        if tuple(stage.name for stage in scheduler.stages) != tuple(
            stage.name for stage in configured_stages
        ):
            raise ValueError("resume checkpoint curriculum stages differ")
        scheduler.stages = configured_stages
        scheduler.promotion_threshold = stage_promotion_threshold(
            curriculum_config,
            scheduler.current.name,
        )
        progress = loaded.progress
        update_start = loaded.update_index
        metric_history = list(loaded.metrics)
        metrics_path.write_text(
            "".join(
                json.dumps(metric, ensure_ascii=False, sort_keys=True) + "\n"
                for metric in metric_history
            ),
            encoding="utf-8",
        )
    else:
        trainer = RecurrentPPOTrainer(
            config=ppo_config,
            seed=args.run_seed,
            device=experiment.device,
        )
        if args.initialize_from is not None:
            initialization = initialize_network_from_checkpoint(trainer, args.initialize_from)
        scheduler = CurriculumScheduler(
            stages=configured_stages,
            promotion_threshold=float(curriculum_config["promotion_threshold"]),
        )
        progress = M7TrainingProgress(
            full_run_entry_update=0 if scheduler.current.name == "full_run" else None
        )
        update_start = 0
        metric_history: list[dict[str, Any]] = []

    resolved_config = {
        "protocol": "m7",
        "experiment": experiment.to_dict(),
        "ppo": ppo_config.to_dict(),
        "curriculum": curriculum_config,
        "dagger": dagger_config.to_dict(),
        "self_imitation": self_imitation_config.to_dict(),
        "parameter_ema": parameter_ema_config.to_dict(),
        "phase_balance": phase_balance_config,
        "training_combat_policy": "heuristic",
        "selection_combat_policy": experiment.selection_combat_policy,
        "collector_reset_on_resume": bool(args.reset_collector),
        "runtime_controls": {
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
        },
        "formal_run_seeds": run_seeds,
        "source_freeze": source_freeze,
        "initialization": initialization,
        "smoke": args.smoke,
    }
    (run_directory / "resolved-config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = build_runtime_manifest(PROJECT_ROOT)
    manifest.update(
        {
            "protocol": "m7",
            "m7_protocol": "docs/M7_IMPLEMENTATION_AND_EVALUATION_PLAN.md",
            "run_seed": args.run_seed,
            "final_test_seed_range": [M7_FINAL_SEED_START, M7_FINAL_SEED_END],
            "training_combat_policy": "heuristic",
            "selection_combat_policy": experiment.selection_combat_policy,
            "collector_reset_on_resume": bool(args.reset_collector),
            "runtime_controls": resolved_config["runtime_controls"],
            "source_freeze": source_freeze,
            "initialization": initialization,
            "git_status": git_status(),
        }
    )
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    parameter_ema = ParameterEMA(trainer.network, parameter_ema_config.decay)
    if args.resume is not None and loaded.parameter_ema_state is not None:
        parameter_ema.load_state_dict(trainer.network, loaded.parameter_ema_state)

    def parameter_ema_active() -> bool:
        return (
            parameter_ema_config.enabled
            and scheduler.current.name in parameter_ema_config.stages
        )

    training_seeds = experiment.training_seeds()
    factory = stage_factory(scheduler, run_directory)
    pool = SubprocessVectorEnvironment(factory, experiment.num_environments)
    heuristic = HeuristicPolicy()
    combat_selector = lambda observation: heuristic(observation)
    if args.resume is not None and not args.reset_collector:
        collector = MultiprocessRecurrentRolloutCollector.from_state_dict(
            pool,
            trainer,
            loaded.collector_state,
            combat_selector=combat_selector,
        )
    else:
        collector = MultiprocessRecurrentRolloutCollector(
            pool,
            trainer,
            training_seeds,
            combat_selector=combat_selector,
        )
    writer = SummaryWriter(log_dir=str(run_directory / "tensorboard"))
    candidate_directory = (
        run_directory / "curriculum" / "candidates" / scheduler.current.name
    )
    candidate_directory.mkdir(parents=True, exist_ok=True)
    candidate_prefix_paths = sorted(candidate_directory.glob("*.jsonl"))
    teacher_paths = m7_teacher_trace_paths(
        run_directory,
        scheduler.current.name,
        shared_teacher_corpus,
    )
    best_validation_key: tuple[float, ...] | None = None
    if scheduler.current.name == "full_run" and best_validation_path.is_file():
        best_payload = json.loads(best_validation_path.read_text(encoding="utf-8"))
        best_validation_key = tuple(float(value) for value in best_payload["selection_key"])

    def save_evaluation_checkpoint(
        destination: Path,
        update_index: int,
        metrics: tuple[dict[str, Any], ...],
        *,
        selection: dict[str, Any] | None = None,
    ) -> None:
        parameter_ema_state = parameter_ema.state_dict()
        evaluation_manifest = {
            **manifest,
            "evaluation_only": True,
            "parameter_source": "ema" if parameter_ema_active() else "online",
            "parameter_ema": parameter_ema_config.to_dict(),
            "selection": selection,
        }
        collector_state = collector.state_dict()
        if parameter_ema_active():
            with parameter_ema.use_averaged_parameters(trainer.network):
                save_m7_checkpoint(
                    destination,
                    trainer=trainer,
                    collector_state=collector_state,
                    scheduler=scheduler,
                    config=experiment,
                    progress=progress,
                    update_index=update_index,
                    metrics=metrics,
                    manifest=evaluation_manifest,
                    parameter_ema_state=parameter_ema_state,
                )
        else:
            save_m7_checkpoint(
                destination,
                trainer=trainer,
                collector_state=collector_state,
                scheduler=scheduler,
                config=experiment,
                progress=progress,
                update_index=update_index,
                metrics=metrics,
                manifest=evaluation_manifest,
                parameter_ema_state=parameter_ema_state,
            )

    def save_checkpoint_pair(update_index: int) -> None:
        parameter_ema_state = parameter_ema.state_dict()
        collector_state = collector.state_dict()
        save_m7_checkpoint(
            checkpoint_path,
            trainer=trainer,
            collector_state=collector_state,
            scheduler=scheduler,
            config=experiment,
            progress=progress,
            update_index=update_index,
            metrics=tuple(metric_history),
            manifest=manifest,
            parameter_ema_state=parameter_ema_state,
        )
        save_evaluation_checkpoint(
            evaluation_checkpoint_path,
            update_index,
            tuple(metric_history),
        )

    def maybe_balance(chunks: tuple[Any, ...]) -> tuple[Any, ...]:
        if not bool(phase_balance_config["enabled"]):
            return chunks
        return balance_imitation_phase_weights(
            chunks,
            maximum_multiplier=float(phase_balance_config["maximum_multiplier"]),
        )

    started = time.perf_counter()
    if args.stop_after_update is not None and args.stop_after_update <= update_start:
        raise ValueError("stop-after-update must be greater than the resume update")
    final_update = update_start
    completed_budget = False
    try:
        for update_index in range(
            update_start + 1,
            experiment.maximum_total_updates + 1,
        ):
            if args.stop_after_update is not None and update_index > args.stop_after_update:
                break
            stage_at_update = scheduler.current.name
            rollout = collector.collect(experiment.rollout_steps)
            train_metrics = trainer.update(rollout)
            if stage_at_update == "full_run":
                progress = replace(
                    progress,
                    full_run_updates_completed=progress.full_run_updates_completed + 1,
                )
            successful_episodes = 0
            candidate_episodes = 0
            for episode, trace in zip(
                rollout.completed_episodes,
                rollout.completed_traces,
                strict=True,
            ):
                target_act = scheduler.current.target_act
                succeeded = (
                    episode.final_act >= target_act
                    if target_act is not None
                    else scheduler.current.name in {"act3_clear", "full_run"} and episode.won
                )
                candidate = is_self_imitation_candidate(
                    scheduler.current.name,
                    target_act=target_act,
                    final_act=episode.final_act,
                    won=episode.won,
                )
                successful_episodes += int(succeeded)
                if not candidate:
                    continue
                candidate_episodes += 1
                candidate_trace = (
                    trace if episode.won else trace.prefix(max(0, len(trace.steps) - 1))
                )
                candidate_path = candidate_directory / (
                    f"act-{episode.final_act}-floor-{episode.final_floor:02d}"
                    f"-u{update_index:06d}"
                    f"-e{episode.environment_index:02d}-s{episode.seed}.jsonl"
                )
                candidate_trace.write_jsonl(candidate_path)
                candidate_prefix_paths.append(candidate_path)
            if len(candidate_prefix_paths) > self_imitation_config.corpus_capacity:
                candidate_prefix_paths = prune_m7_candidate_paths(
                    candidate_prefix_paths,
                    self_imitation_config.corpus_capacity,
                )

            record: dict[str, Any] = {
                "update": update_index,
                "environment_steps": trainer.environment_steps,
                "stage": stage_at_update,
                "stage_update": (
                    progress.full_run_updates_completed
                    if stage_at_update == "full_run"
                    else update_index
                ),
                "progress": progress.to_dict(),
                "wall_seconds": time.perf_counter() - started,
                "completed_episodes": len(rollout.completed_episodes),
                "successful_episodes": successful_episodes,
                "candidate_episodes": candidate_episodes,
                **train_metrics,
            }
            parameter_ema_updated = False
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, trainer.environment_steps)

            if (
                self_imitation_config.enabled
                and update_index % self_imitation_config.interval_updates == 0
                and len(candidate_prefix_paths) + len(teacher_paths)
                >= self_imitation_config.minimum_traces
            ):
                imitation_traces = load_m7_imitation_traces(
                    candidate_prefix_paths,
                    teacher_paths,
                    self_imitation_config.maximum_traces,
                    self_imitation_config.maximum_candidate_traces,
                    self_imitation_config.frontier_trace_repeats,
                )
                imitation_chunks = build_imitation_chunks(
                    LightspeedEnvironmentFactory(),
                    trainer,
                    imitation_traces,
                    chunk_length=self_imitation_config.chunk_length,
                    burn_in_steps=self_imitation_config.burn_in_steps,
                    recovery_environment_factory=factory,
                )
                if imitation_chunks:
                    raw_coverage = imitation_phase_coverage(imitation_chunks)
                    imitation_chunks = maybe_balance(imitation_chunks)
                    imitation_metrics = train_self_imitation(
                        trainer,
                        imitation_chunks,
                        epochs=self_imitation_config.epochs,
                        seed=args.run_seed * 1_000_003 + update_index,
                    )
                    record["self_imitation"] = {
                        **imitation_metrics,
                        "phase_coverage": raw_coverage,
                        "balanced_phase_coverage": imitation_phase_coverage(
                            imitation_chunks
                        ),
                    }
                    for key, value in imitation_metrics.items():
                        writer.add_scalar(
                            f"self_imitation/{key}",
                            value,
                            trainer.environment_steps,
                        )

            if update_index % experiment.validation_interval == 0:
                if dagger_config.enabled and update_index % dagger_config.interval_updates == 0:
                    dagger_rounds = []
                    for round_index in range(
                        dagger_config.rounds_for_stage(scheduler.current.name)
                    ):
                        dagger_seeds = dagger_training_seeds(
                            dagger_config,
                            training_seed_start=experiment.training_seed_start,
                            training_seed_count=experiment.training_seed_count,
                            update_index=update_index,
                            round_index=round_index,
                        )
                        dagger_chunks = collect_dagger_chunks(
                            factory,
                            trainer,
                            heuristic,
                            dagger_seeds,
                            max_steps=dagger_config.max_steps,
                            chunk_length=dagger_config.chunk_length,
                            burn_in_steps=dagger_config.burn_in_steps,
                            phase_weights=dagger_config.phase_weights(),
                        )
                        if not dagger_chunks:
                            dagger_rounds.append(
                                {
                                    "round": round_index + 1,
                                    "seed_start": dagger_seeds[0],
                                    "seed_end": dagger_seeds[-1],
                                    "skipped": "no supervised non-combat decisions",
                                }
                            )
                            continue
                        raw_coverage = imitation_phase_coverage(dagger_chunks)
                        dagger_chunks = maybe_balance(dagger_chunks)
                        dagger_metrics = train_self_imitation(
                            trainer,
                            dagger_chunks,
                            epochs=dagger_config.epochs,
                            seed=(
                                args.run_seed * 1_000_003
                                + update_index * dagger_config.rounds
                                + round_index
                            ),
                        )
                        dagger_rounds.append(
                            {
                                "round": round_index + 1,
                                **dagger_metrics,
                                "phase_coverage": raw_coverage,
                                "balanced_phase_coverage": imitation_phase_coverage(
                                    dagger_chunks
                                ),
                                "seed_start": dagger_seeds[0],
                                "seed_end": dagger_seeds[-1],
                            }
                        )
                    record["dagger"] = {
                        "configured_rounds": dagger_config.rounds_for_stage(
                            scheduler.current.name
                        ),
                        "rounds": dagger_rounds,
                    }
                    completed_rounds = [
                        dagger_round
                        for dagger_round in dagger_rounds
                        if "skipped" not in dagger_round
                    ]
                    if completed_rounds:
                        record["dagger"].update(
                            {
                                key: value
                                for key, value in completed_rounds[-1].items()
                                if key not in {
                                    "round",
                                    "phase_coverage",
                                    "balanced_phase_coverage",
                                }
                            }
                        )

                if parameter_ema_active():
                    parameter_ema.update(trainer.network)
                else:
                    parameter_ema.reset(trainer.network)
                parameter_ema_updated = True
                record["parameter_ema"] = {
                    "active": parameter_ema_active(),
                    "decay": parameter_ema_config.decay,
                    "evaluation_model": (
                        "ema" if parameter_ema_active() else "online"
                    ),
                }
                if scheduler.current.name == "full_run":
                    screening_seeds = experiment.screening_seeds(
                        progress.screening_batches_completed
                    )
                    with parameter_ema.use_averaged_parameters(trainer.network):
                        screening_summary = evaluate_m7_full_run(
                            trainer,
                            factory,
                            screening_seeds,
                            policy_seed=args.run_seed,
                            combat_policy="heuristic",
                            search_budget=experiment.selection_search_budget,
                            max_steps=int(curriculum_config["max_episode_steps"]),
                            bootstrap_samples=500 if args.smoke else 2_000,
                        )
                    screening = compact_full_run_summary(screening_summary)
                    progress = replace(
                        progress,
                        screening_batches_completed=(
                            progress.screening_batches_completed + 1
                        ),
                    )
                    record["screening"] = {
                        **screening,
                        "seed_range": [screening_seeds[0], screening_seeds[-1]],
                        "seed_count": len(screening_seeds),
                        "combat_policy": "heuristic",
                    }
                    selection_due = (
                        progress.full_run_updates_completed
                        % experiment.selection_interval
                        == 0
                        or progress.full_run_updates_completed
                        >= experiment.full_run_updates
                    )
                    if selection_due:
                        selection_seeds = experiment.selection_seeds()
                        with parameter_ema.use_averaged_parameters(trainer.network):
                            selection_summary = evaluate_m7_full_run(
                                trainer,
                                factory,
                                selection_seeds,
                                policy_seed=args.run_seed,
                                combat_policy=experiment.selection_combat_policy,
                                search_budget=experiment.selection_search_budget,
                                max_steps=int(curriculum_config["max_episode_steps"]),
                                bootstrap_samples=500 if args.smoke else 2_000,
                            )
                        selection = compact_full_run_summary(selection_summary)
                        selection_key = m7_validation_selection_key(selection)
                        improved = (
                            best_validation_key is None
                            or selection_key > best_validation_key
                        )
                        record["selection"] = {
                            **selection,
                            "seed_range": [selection_seeds[0], selection_seeds[-1]],
                            "seed_count": len(selection_seeds),
                            "combat_policy": experiment.selection_combat_policy,
                            "selection_key": list(selection_key),
                            "best_before": (
                                None
                                if best_validation_key is None
                                else list(best_validation_key)
                            ),
                            "improved": improved,
                        }
                        progress = replace(
                            progress,
                            selection_evaluations_completed=(
                                progress.selection_evaluations_completed + 1
                            ),
                        )
                        if improved:
                            best_validation_key = selection_key
                            best_payload = {
                                "protocol": "m7",
                                "update": update_index,
                                "full_run_update": progress.full_run_updates_completed,
                                "stage": scheduler.current.name,
                                "selection_key": list(selection_key),
                                "selection": record["selection"],
                                "checkpoint": str(best_evaluation_checkpoint_path),
                            }
                            save_evaluation_checkpoint(
                                best_evaluation_checkpoint_path,
                                update_index,
                                (*metric_history, record),
                                selection=best_payload,
                            )
                            best_validation_path.write_text(
                                json.dumps(
                                    best_payload,
                                    ensure_ascii=False,
                                    indent=2,
                                    sort_keys=True,
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                else:
                    with parameter_ema.use_averaged_parameters(trainer.network):
                        validation = evaluate_m7_curriculum_stage(
                            trainer,
                            factory,
                            experiment.promotion_seeds(),
                            policy_seed=args.run_seed,
                            max_steps=int(curriculum_config["max_episode_steps"]),
                            combat_policy="heuristic",
                        )
                    record["promotion_validation"] = {
                        **validation,
                        "seed_range": [
                            experiment.promotion_seed_start,
                            experiment.promotion_seed_start
                            + experiment.promotion_seed_count
                            - 1,
                        ],
                        "seed_count": experiment.promotion_seed_count,
                    }
                    previous_stage = scheduler.current
                    scheduler.promotion_threshold = stage_promotion_threshold(
                        curriculum_config,
                        scheduler.current.name,
                    )
                    next_stage = (
                        scheduler.stages[scheduler.stage_index + 1]
                        if scheduler.stage_index + 1 < len(scheduler.stages)
                        else None
                    )
                    prefix_ready = (
                        next_stage is None
                        or not next_stage.use_prefix_starts
                        or bool(candidate_prefix_paths or teacher_paths)
                    )
                    promoted = prefix_ready and scheduler.observe_validation(
                        validation["completion_rate"]
                    )
                    if (
                        not prefix_ready
                        and validation["completion_rate"]
                        >= scheduler.promotion_threshold
                    ):
                        record["promotion_blocked"] = (
                            "no recoverable prefix reached the next act"
                        )
                    if promoted:
                        next_stage = scheduler.current
                        if next_stage.use_prefix_starts:
                            build_m7_prefix_corpus(
                                run_directory,
                                next_stage.start_act,
                                [*candidate_prefix_paths, *teacher_paths],
                                recovery_environment_factory=factory,
                            )
                        collector.close()
                        factory = stage_factory(scheduler, run_directory)
                        pool = SubprocessVectorEnvironment(
                            factory,
                            experiment.num_environments,
                        )
                        collector = MultiprocessRecurrentRolloutCollector(
                            pool,
                            trainer,
                            training_seeds,
                            combat_selector=combat_selector,
                        )
                        candidate_directory = (
                            run_directory
                            / "curriculum"
                            / "candidates"
                            / scheduler.current.name
                        )
                        candidate_directory.mkdir(parents=True, exist_ok=True)
                        candidate_prefix_paths = sorted(
                            candidate_directory.glob("*.jsonl")
                        )
                        teacher_paths = m7_teacher_trace_paths(
                            run_directory,
                            scheduler.current.name,
                            shared_teacher_corpus,
                        )
                        record["promotion"] = {
                            "from": previous_stage.name,
                            "to": next_stage.name,
                        }
                        parameter_ema.reset(trainer.network)
                        record["parameter_ema"]["reset_after_promotion"] = True
                        if next_stage.name == "full_run":
                            progress = replace(
                                progress,
                                full_run_entry_update=update_index,
                                full_run_updates_completed=0,
                                screening_batches_completed=0,
                                selection_evaluations_completed=0,
                            )
                            best_validation_key = None

            if not parameter_ema_updated:
                if parameter_ema_active():
                    parameter_ema.update(trainer.network)
                else:
                    parameter_ema.reset(trainer.network)
                record["parameter_ema"] = {
                    "active": parameter_ema_active(),
                    "decay": parameter_ema_config.decay,
                    "evaluation_model": (
                        "ema" if parameter_ema_active() else "online"
                    ),
                }

            record["progress"] = progress.to_dict()
            metric_history.append(record)
            append_jsonl(metrics_path, record)
            final_update = update_index
            if update_index % experiment.checkpoint_interval == 0:
                save_checkpoint_pair(update_index)
            writer.flush()

            if progress.full_run_updates_completed >= experiment.full_run_updates:
                completed_budget = True
                save_checkpoint_pair(update_index)
                break
            if (
                scheduler.current.name != "full_run"
                and update_index >= experiment.max_curriculum_updates
            ):
                save_checkpoint_pair(update_index)
                raise RuntimeError(
                    "M7 curriculum failed to reach full_run within max_curriculum_updates"
                )
        else:
            raise RuntimeError("M7 exhausted its maximum total update safety bound")
        if not completed_budget:
            save_checkpoint_pair(final_update)
    finally:
        collector.close()
        writer.close()

    state = "complete" if completed_budget else "stopped"
    print(
        json.dumps(
            {
                "protocol": "m7",
                "state": state,
                "checkpoint": str(checkpoint_path),
                "evaluation_checkpoint": str(evaluation_checkpoint_path),
                "best_evaluation_checkpoint": str(best_evaluation_checkpoint_path),
                "environment_steps": trainer.environment_steps,
                "stage": scheduler.current.name,
                "updates": final_update,
                "full_run_updates": progress.full_run_updates_completed,
                "full_run_update_target": experiment.full_run_updates,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
