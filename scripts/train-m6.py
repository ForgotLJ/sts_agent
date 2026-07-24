from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.tensorboard import SummaryWriter

from sts_env.training import (
    CurriculumEnvironmentFactory,
    CurriculumScheduler,
    DEFAULT_CURRICULUM,
    DaggerConfig,
    HierarchicalRecurrentPolicy,
    HeuristicPolicy,
    LightspeedEnvironmentFactory,
    M6TrainingConfig,
    MultiprocessRecurrentRolloutCollector,
    ParameterEMA,
    ParameterEMAConfig,
    PrefixCorpus,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    SelfImitationConfig,
    SubprocessVectorEnvironment,
    build_imitation_chunks,
    collect_dagger_chunks,
    dagger_training_seeds,
    imitation_trace_progress,
    is_self_imitation_candidate,
    load_m6_checkpoint,
    m6_validation_selection_key,
    materialize_recovery_trace,
    save_m6_checkpoint,
    rank_imitation_traces,
    select_weighted_frontier_traces,
    train_self_imitation,
)
from sts_env.training.experiment import build_runtime_manifest
from sts_env.training.teacher_corpus import verify_teacher_corpus_manifest
from sts_env import EpisodeTrace
from sts_env.differential import canonical_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the M6 recurrent full-run agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "m6_recurrent_ppo.json",
    )
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m6_recurrent_ppo",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--source-freeze", type=Path)
    parser.add_argument(
        "--stop-after-update",
        type=int,
        help="stop after this update while retaining the configured total update budget",
    )
    parser.add_argument(
        "--reset-collector",
        action="store_true",
        help="reset worker environments after an intentional observation schema migration",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=2,
        help="CPU threads used by PyTorch within this training process",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="PyTorch inter-op thread count",
    )
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
    if int(payload.get("schema_version", -1)) != 1 or payload.get("status") != "frozen":
        raise ValueError("invalid M6 source freeze manifest")
    current = build_runtime_manifest(PROJECT_ROOT)
    frozen_source = dict(payload.get("runtime_manifest") or {}).get("source_sha256")
    if current["source_sha256"] != frozen_source:
        raise ValueError("current source tree differs from the M6 source freeze")
    frozen_config = dict(payload.get("config") or {})
    if str(config_path.resolve()) != frozen_config.get("path"):
        raise ValueError("training config path differs from the M6 source freeze")
    if sha256_file(config_path) != frozen_config.get("sha256"):
        raise ValueError("training config differs from the M6 source freeze")
    teacher_gate = dict(dict(payload.get("gates") or {}).get("teacher-corpus") or {})
    teacher_payload = dict(teacher_gate.get("payload") or {})
    teacher_corpus = verify_teacher_corpus_manifest(teacher_payload)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_sha256": frozen_source,
        "config_sha256": frozen_config["sha256"],
        "teacher_corpus_path": teacher_corpus["corpus_path"],
        "teacher_corpus_sha256": teacher_corpus["aggregate_sha256"],
    }


def load_configuration(path: Path, run_seed: int, smoke: bool) -> tuple[Any, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment_payload = dict(payload["experiment"])
    run_seeds = tuple(int(seed) for seed in experiment_payload.pop("run_seeds"))
    if run_seed not in run_seeds:
        raise ValueError(f"run seed {run_seed} is not in configured run seeds {run_seeds}")
    experiment = M6TrainingConfig(run_seed=run_seed, **experiment_payload)
    ppo = RecurrentPPOConfig(**payload["ppo"])
    curriculum = dict(payload["curriculum"])
    self_imitation = SelfImitationConfig(**payload.get("self_imitation", {}))
    parameter_ema = ParameterEMAConfig(**payload.get("parameter_ema", {}))
    dagger = DaggerConfig(**payload.get("dagger", {}))
    if smoke:
        experiment = replace(
            experiment,
            num_environments=2,
            rollout_steps=8,
            total_updates=2,
            checkpoint_interval=1,
            validation_interval=1,
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
        curriculum["validation_episodes"] = 4
        curriculum["max_episode_steps"] = 1000
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
        run_seeds,
    )


def stage_factory(
    scheduler: CurriculumScheduler,
    run_directory: Path,
) -> CurriculumEnvironmentFactory:
    spec = scheduler.current
    corpus = None
    if spec.use_prefix_starts:
        corpus = PrefixCorpus.read(run_directory / "curriculum" / f"act-{spec.start_act}")
    return CurriculumEnvironmentFactory(spec=spec, prefix_corpus=corpus)


def evaluate_stage(
    trainer: RecurrentPPOTrainer,
    factory: CurriculumEnvironmentFactory,
    seeds: tuple[int, ...],
    max_steps: int,
    combat_selector: Any | None = None,
) -> dict[str, float]:
    completed = 0
    wins = 0
    floors: list[int] = []
    lengths: list[int] = []
    for seed in seeds:
        environment = factory()
        observation, _ = environment.reset(seed=seed)
        policy = HierarchicalRecurrentPolicy(
            trainer,
            combat_selector=combat_selector,
            deterministic=True,
        )
        episode_completed = False
        repeated_decisions: dict[str, int] = {}
        for step_index in range(max_steps):
            action = policy.select(environment)
            decision_key = json.dumps(
                {
                    "observation": canonical_observation(observation),
                    "action": action.to_dict(),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            repeated_decisions[decision_key] = repeated_decisions.get(decision_key, 0) + 1
            if repeated_decisions[decision_key] > 16:
                lengths.append(step_index)
                floors.append(observation.floor)
                break
            observation, _, terminated, truncated, info = environment.step(action)
            episode_completed = episode_completed or bool(info.get("curriculum_completed", False))
            if terminated or truncated:
                completed += int(episode_completed or float(info.get("raw_reward", 0.0)) > 0)
                wins += int(terminated and float(info.get("raw_reward", 0.0)) > 0)
                lengths.append(step_index + 1)
                floors.append(observation.floor)
                break
        else:
            lengths.append(max_steps)
            floors.append(observation.floor)
    return {
        "completion_rate": completed / len(seeds),
        "win_rate": wins / len(seeds),
        "mean_floor": sum(floors) / len(floors),
        "mean_length": sum(lengths) / len(lengths),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def build_prefix_corpus(
    run_directory: Path,
    target_act: int,
    candidate_paths: list[Path],
    recovery_environment_factory: Any | None = None,
) -> PrefixCorpus:
    traces = tuple(EpisodeTrace.read_jsonl(path) for path in candidate_paths)
    traces = tuple(trace for trace in traces if trace_uses_semantic_neow(trace))
    if not traces:
        raise ValueError("no candidate trace uses the current public action schema")
    progress: list[tuple[EpisodeTrace, int, bool]] = []
    for trace in traces:
        has_recovery_prefix = bool(
            (trace.metadata or {}).get("curriculum_source_trace")
        )
        if has_recovery_prefix:
            if recovery_environment_factory is None:
                raise ValueError(
                    "curriculum prefix extraction requires a recovery environment factory"
                )
            environment = recovery_environment_factory()
            observation = environment.replay_recovery_trace(trace.prefix(0))
        else:
            environment = LightspeedEnvironmentFactory()()
            observation, _ = environment.reset(seed=trace.seed)
        prefix = trace.prefix(0) if observation.act >= target_act else None
        for step_index, step in enumerate(trace.steps):
            if step.action not in observation.legal_actions:
                raise ValueError("prefix source trace contains a stale action")
            observation, _, _, _, _ = environment.step(step.action)
            if prefix is None and observation.act >= target_act:
                prefix = trace.prefix(step_index + 1)
        if prefix is not None:
            progress.append(
                (
                    prefix,
                    observation.act,
                    bool(trace.steps and trace.steps[-1].reward > 0),
                )
            )
    if not progress:
        raise ValueError(f"no supplied trace reaches Act {target_act}")
    if target_act < 3:
        farther = tuple(prefix for prefix, final_act, _ in progress if final_act > target_act)
        if farther:
            prefixes = farther
        else:
            prefixes = tuple(prefix for prefix, _, _ in progress)
    else:
        wins = tuple(prefix for prefix, _, won in progress if won)
        if wins:
            prefixes = wins
        else:
            prefixes = tuple(prefix for prefix, _, _ in progress)
    corpus = PrefixCorpus(prefixes, target_act)
    corpus.write(run_directory / "curriculum" / f"act-{target_act}")
    return corpus


def teacher_trace_paths(
    run_directory: Path,
    stage_name: str,
    shared_corpus: Path | None = None,
) -> list[Path]:
    stage_candidates = [stage_name]
    if stage_name in {"act3_clear", "full_run"}:
        stage_candidates = ["act2_clear", stage_name]
    if shared_corpus is not None:
        for candidate_stage in stage_candidates:
            paths = sorted((shared_corpus / candidate_stage).glob("*.jsonl"))
            if paths:
                return paths
    for candidate_stage in stage_candidates:
        for source in ("teacher-v4", "teacher-v3", "teacher-v2", "teacher"):
            paths = sorted(
                (run_directory / "curriculum" / source / candidate_stage).glob(
                    "*.jsonl"
                )
            )
            if paths:
                return paths
    return []


def stage_promotion_threshold(
    curriculum_config: dict[str, Any],
    stage_name: str,
) -> float:
    thresholds = dict(curriculum_config.get("stage_promotion_thresholds", {}))
    return float(thresholds.get(stage_name, curriculum_config["promotion_threshold"]))


def load_imitation_traces(
    candidate_paths: list[Path],
    teacher_paths: list[Path],
    maximum_traces: int,
    maximum_candidate_traces: int,
    frontier_trace_repeats: int,
) -> tuple[EpisodeTrace, ...]:
    teacher_limit = min(len(teacher_paths), maximum_traces * 3 // 4)
    candidate_limit = min(
        maximum_candidate_traces,
        maximum_traces - teacher_limit,
    )
    selected_teachers: list[Path] = []
    if teacher_limit:
        ordered_teachers = sorted(teacher_paths)
        selected_teachers = [
            ordered_teachers[index * len(ordered_teachers) // teacher_limit]
            for index in range(teacher_limit)
        ]
    loaded_candidates = tuple(EpisodeTrace.read_jsonl(path) for path in candidate_paths)
    loaded_candidates = tuple(
        trace
        for trace in loaded_candidates
        if trace_uses_semantic_neow(materialize_recovery_trace(trace))
    )
    selected_candidates = select_weighted_frontier_traces(
        loaded_candidates,
        candidate_limit,
        frontier_trace_repeats,
    )
    selected_teacher_traces = tuple(
        trace
        for path in selected_teachers
        if trace_uses_semantic_neow(trace := EpisodeTrace.read_jsonl(path))
    )
    return (*selected_candidates, *selected_teacher_traces)


def trace_uses_semantic_neow(trace: EpisodeTrace) -> bool:
    source_payload = dict((trace.metadata or {}).get("curriculum_source_trace") or {})
    while source_payload:
        trace = EpisodeTrace.from_dict(source_payload)
        source_payload = dict(
            (trace.metadata or {}).get("curriculum_source_trace") or {}
        )
    if not trace.steps:
        return False
    source_id = trace.steps[0].action.source_id
    return isinstance(source_id, str) and source_id.startswith("neow")


def prune_candidate_paths(candidate_paths: list[Path], capacity: int) -> list[Path]:
    traces = {path: EpisodeTrace.read_jsonl(path) for path in candidate_paths}
    ordered = sorted(
        candidate_paths,
        key=lambda path: (
            -imitation_trace_progress(traces[path])[0],
            -imitation_trace_progress(traces[path])[1],
            -imitation_trace_progress(traces[path])[2],
            path.name,
        ),
    )
    for path in ordered[capacity:]:
        path.unlink()
    return sorted(ordered[:capacity])


def main() -> int:
    args = parse_args()
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
        run_seeds,
    ) = load_configuration(args.config, args.run_seed, args.smoke)
    run_directory = args.output / f"seed-{args.run_seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "resolved-config.json").write_text(
        json.dumps(
            {
                "experiment": experiment.to_dict(),
                "ppo": ppo_config.to_dict(),
                "curriculum": curriculum_config,
                "dagger": dagger_config.to_dict(),
                "self_imitation": self_imitation_config.to_dict(),
                "parameter_ema": parameter_ema_config.to_dict(),
                "training_combat_policy": "heuristic",
                "collector_reset_on_resume": bool(args.reset_collector),
                "runtime_controls": {
                    "torch_threads": args.torch_threads,
                    "torch_interop_threads": args.torch_interop_threads,
                },
                "formal_run_seeds": run_seeds,
                "source_freeze": source_freeze,
                "smoke": args.smoke,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = build_runtime_manifest(PROJECT_ROOT)
    manifest.update(
        {
            "m6_protocol": "docs/M6_IMPLEMENTATION_AND_EVALUATION_PLAN.md",
            "run_seed": args.run_seed,
            "final_test_seed_range": [2000000, 2001023],
            "training_combat_policy": "heuristic",
            "collector_reset_on_resume": bool(args.reset_collector),
            "runtime_controls": {
                "torch_threads": args.torch_threads,
                "torch_interop_threads": args.torch_interop_threads,
            },
            "source_freeze": source_freeze,
        }
    )
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checkpoint_path = run_directory / "checkpoint.pt"
    evaluation_checkpoint_path = run_directory / "evaluation-checkpoint.pt"
    best_evaluation_checkpoint_path = run_directory / "best-evaluation-checkpoint.pt"
    best_validation_path = run_directory / "best-validation.json"
    metrics_path = run_directory / "metrics.jsonl"
    configured_stages = tuple(
        replace(
            stage,
            max_episode_steps=int(curriculum_config["max_episode_steps"]),
        )
        for stage in DEFAULT_CURRICULUM
    )
    if args.resume is not None:
        loaded = load_m6_checkpoint(args.resume, device=experiment.device)
        if loaded.manifest.get("evaluation_only"):
            raise ValueError("evaluation-only EMA checkpoints cannot resume training")
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
            raise ValueError("resume configuration changes the checkpoint network architecture")
        trainer.config = ppo_config
        trainer.network.config = ppo_config
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = ppo_config.learning_rate
        scheduler = loaded.scheduler
        if tuple(stage.name for stage in scheduler.stages) != tuple(
            stage.name for stage in configured_stages
        ):
            raise ValueError("resume checkpoint curriculum stages differ from configuration")
        scheduler.stages = configured_stages
        scheduler.promotion_threshold = stage_promotion_threshold(
            curriculum_config,
            scheduler.current.name,
        )
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
        scheduler = CurriculumScheduler(
            stages=configured_stages,
            promotion_threshold=float(curriculum_config["promotion_threshold"]),
        )
        update_start = 0
        metric_history: list[dict[str, Any]] = []

    parameter_ema = ParameterEMA(trainer.network, parameter_ema_config.decay)
    if args.resume is not None and loaded.parameter_ema_state is not None:
        parameter_ema.load_state_dict(trainer.network, loaded.parameter_ema_state)

    def parameter_ema_active() -> bool:
        return (
            parameter_ema_config.enabled
            and scheduler.current.name in parameter_ema_config.stages
        )

    training_seeds = tuple(
        range(
            experiment.training_seed_start,
            experiment.training_seed_start + experiment.training_seed_count,
        )
    )
    validation_count = int(curriculum_config["validation_episodes"])
    validation_seeds = tuple(
        range(experiment.validation_seed_start, experiment.validation_seed_start + validation_count)
    )
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
    candidate_prefix_paths: list[Path] = sorted(candidate_directory.glob("*.jsonl"))
    teacher_paths = teacher_trace_paths(
        run_directory,
        scheduler.current.name,
        shared_teacher_corpus,
    )
    best_validation_key: tuple[float, float] | None = None
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
                save_m6_checkpoint(
                    destination,
                    trainer=trainer,
                    collector_state=collector_state,
                    scheduler=scheduler,
                    config=experiment,
                    update_index=update_index,
                    metrics=metrics,
                    manifest=evaluation_manifest,
                    parameter_ema_state=parameter_ema_state,
                )
        else:
            save_m6_checkpoint(
                destination,
                trainer=trainer,
                collector_state=collector_state,
                scheduler=scheduler,
                config=experiment,
                update_index=update_index,
                metrics=metrics,
                manifest=evaluation_manifest,
                parameter_ema_state=parameter_ema_state,
            )

    def save_checkpoint_pair(update_index: int) -> None:
        parameter_ema_state = parameter_ema.state_dict()
        collector_state = collector.state_dict()
        save_m6_checkpoint(
            checkpoint_path,
            trainer=trainer,
            collector_state=collector_state,
            scheduler=scheduler,
            config=experiment,
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

    started = time.perf_counter()
    target_update = experiment.total_updates
    if args.stop_after_update is not None:
        if args.stop_after_update <= update_start:
            raise ValueError("stop-after-update must be greater than the resume update")
        target_update = min(target_update, args.stop_after_update)
    try:
        for update_index in range(update_start + 1, target_update + 1):
            rollout = collector.collect(experiment.rollout_steps)
            train_metrics = trainer.update(rollout)
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
                if succeeded:
                    successful_episodes += 1
                if not candidate:
                    continue
                candidate_episodes += 1
                candidate_trace = trace if episode.won else trace.prefix(max(0, len(trace.steps) - 1))
                path = candidate_directory / (
                    f"act-{episode.final_act}-floor-{episode.final_floor:02d}"
                    f"-u{update_index:06d}"
                    f"-e{episode.environment_index:02d}-s{episode.seed}.jsonl"
                )
                candidate_trace.write_jsonl(path)
                candidate_prefix_paths.append(path)
            if len(candidate_prefix_paths) > self_imitation_config.corpus_capacity:
                candidate_prefix_paths = prune_candidate_paths(
                    candidate_prefix_paths,
                    self_imitation_config.corpus_capacity,
                )

            record: dict[str, Any] = {
                "update": update_index,
                "environment_steps": trainer.environment_steps,
                "stage": scheduler.current.name,
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
                imitation_traces = load_imitation_traces(
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
                imitation_metrics = train_self_imitation(
                    trainer,
                    imitation_chunks,
                    epochs=self_imitation_config.epochs,
                    seed=args.run_seed * 1_000_003 + update_index,
                )
                record["self_imitation"] = imitation_metrics
                for key, value in imitation_metrics.items():
                    writer.add_scalar(
                        f"self_imitation/{key}",
                        value,
                        trainer.environment_steps,
                    )

            if update_index % experiment.validation_interval == 0:
                if (
                    dagger_config.enabled
                    and update_index % dagger_config.interval_updates == 0
                ):
                    dagger_rounds = []
                    stage_dagger_rounds = dagger_config.rounds_for_stage(
                        scheduler.current.name
                    )
                    for round_index in range(stage_dagger_rounds):
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
                                "seed_start": dagger_seeds[0],
                                "seed_end": dagger_seeds[-1],
                            }
                        )
                        for key, value in dagger_metrics.items():
                            writer.add_scalar(
                                f"dagger/round-{round_index + 1}/{key}",
                                value,
                                trainer.environment_steps,
                            )
                    record["dagger"] = {"rounds": dagger_rounds}
                    record["dagger"]["configured_rounds"] = stage_dagger_rounds
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
                                if key != "round"
                            }
                        )
                if parameter_ema_active():
                    parameter_ema.update(trainer.network)
                    with parameter_ema.use_averaged_parameters(trainer.network):
                        validation = evaluate_stage(
                            trainer,
                            factory,
                            validation_seeds,
                            int(curriculum_config["max_episode_steps"]),
                            combat_selector=lambda environment: heuristic(
                                environment.observation
                            ),
                        )
                else:
                    parameter_ema.reset(trainer.network)
                    validation = evaluate_stage(
                        trainer,
                        factory,
                        validation_seeds,
                        int(curriculum_config["max_episode_steps"]),
                        combat_selector=lambda environment: heuristic(
                            environment.observation
                        ),
                    )
                parameter_ema_updated = True
                record["parameter_ema"] = {
                    "active": parameter_ema_active(),
                    "decay": parameter_ema_config.decay,
                    "evaluation_model": (
                        "ema" if parameter_ema_active() else "online"
                    ),
                }
                record["validation"] = validation
                if scheduler.current.name == "full_run":
                    selection_key = m6_validation_selection_key(
                        scheduler.current.name,
                        validation,
                    )
                    improved = (
                        best_validation_key is None
                        or selection_key > best_validation_key
                    )
                    record["model_selection"] = {
                        "selection_key": list(selection_key),
                        "best_before": (
                            None
                            if best_validation_key is None
                            else list(best_validation_key)
                        ),
                        "improved": improved,
                    }
                    if improved:
                        best_validation_key = selection_key
                        best_payload = {
                            "update": update_index,
                            "stage": scheduler.current.name,
                            "selection_key": list(selection_key),
                            "validation": validation,
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
                for key, value in validation.items():
                    writer.add_scalar(f"validation/{key}", value, trainer.environment_steps)
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
                if not prefix_ready and validation["completion_rate"] >= scheduler.promotion_threshold:
                    record["promotion_blocked"] = "no recoverable prefix reached the next act"
                if promoted:
                    next_stage = scheduler.current
                    if next_stage.use_prefix_starts:
                        build_prefix_corpus(
                            run_directory,
                            next_stage.start_act,
                            [*candidate_prefix_paths, *teacher_paths],
                            recovery_environment_factory=factory,
                        )
                    collector.close()
                    factory = stage_factory(scheduler, run_directory)
                    pool = SubprocessVectorEnvironment(factory, experiment.num_environments)
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
                    candidate_prefix_paths = sorted(candidate_directory.glob("*.jsonl"))
                    teacher_paths = teacher_trace_paths(
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

            metric_history.append(record)
            append_jsonl(metrics_path, record)
            if update_index % experiment.checkpoint_interval == 0:
                save_checkpoint_pair(update_index)
            writer.flush()

        save_checkpoint_pair(target_update)
    finally:
        collector.close()
        writer.close()
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "evaluation_checkpoint": str(evaluation_checkpoint_path),
                "best_evaluation_checkpoint": str(best_evaluation_checkpoint_path),
                "environment_steps": trainer.environment_steps,
                "stage": scheduler.current.name,
                "updates": target_update,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
