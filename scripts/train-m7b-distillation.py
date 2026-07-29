from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from sts_env.training import (
    LightspeedEnvironmentFactory,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    load_m6_checkpoint,
    load_m7_checkpoint,
    replay_cache_batch_paths,
    validate_m7b_training_objective,
    verify_m7b_replay_manifest,
)
from sts_env.training.experiment import build_runtime_manifest
from sts_env.training.m7b_distillation import (
    M7BDistillationConfig,
    M7BDistillationProgress,
    corpus_trace_paths,
    evaluate_m7b_imitation,
    load_m7b_checkpoint,
    m7b_validation_selection_key,
    merge_m7b_metric_totals,
    phase_stratified_imitation_chunks,
    save_m7b_checkpoint,
    sha256_file,
    train_m7b_chunk_batch,
    verify_m7b_corpus_manifest,
)
from sts_env.training.m7b_replay import load_m7b_replay_batch
from sts_env.training.self_imitation import build_imitation_chunks
from sts_env.trace import EpisodeTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the persistent M7-B non-combat distillation policy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "m7b_noncombat_distillation.json",
    )
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--validation-corpus", type=Path, required=True)
    parser.add_argument("--train-replay-cache", type=Path)
    parser.add_argument("--validation-replay-cache", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after-batches", type=int)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--progress-interval-batches", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_configuration(
    path: Path,
    run_seed: int,
    smoke: bool,
) -> tuple[M7BDistillationConfig, RecurrentPPOConfig, tuple[int, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "m7b":
        raise ValueError("M7-B config must declare protocol=m7b")
    experiment_payload = dict(payload["experiment"])
    run_seeds = tuple(int(seed) for seed in experiment_payload.pop("run_seeds"))
    if run_seed not in run_seeds:
        raise ValueError(f"M7-B run seed is not configured: {run_seed}")
    experiment = M7BDistillationConfig(run_seed=run_seed, **experiment_payload)
    ppo = RecurrentPPOConfig(**payload["ppo"])
    validate_m7b_training_objective(ppo)
    if smoke:
        experiment = replace(
            experiment,
            device="cpu",
            max_epochs=2,
            early_stopping_patience=2,
            trace_batch_size=4,
            optimizer_batch_chunks=4,
        )
    return experiment, ppo, run_seeds


def resolve_corpora(
    experiment: M7BDistillationConfig,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    *,
    smoke: bool,
) -> tuple[M7BDistillationConfig, dict[str, Any], dict[str, Any]]:
    if smoke:
        train_manifest = verify_m7b_corpus_manifest(train_manifest_path)
        validation_manifest = verify_m7b_corpus_manifest(validation_manifest_path)
        train_start, train_end = (
            int(value) for value in train_manifest["seed_range"]
        )
        validation_start, validation_end = (
            int(value) for value in validation_manifest["seed_range"]
        )
        experiment = replace(
            experiment,
            training_seed_start=train_start,
            training_seed_count=train_end - train_start + 1,
            validation_seed_start=validation_start,
            validation_seed_count=validation_end - validation_start + 1,
        )
    else:
        train_manifest = verify_m7b_corpus_manifest(
            train_manifest_path,
            expected_seed_start=experiment.training_seed_start,
            expected_seed_count=experiment.training_seed_count,
        )
        validation_manifest = verify_m7b_corpus_manifest(
            validation_manifest_path,
            expected_seed_start=experiment.validation_seed_start,
            expected_seed_count=experiment.validation_seed_count,
        )
    if train_manifest["aggregate_sha256"] == validation_manifest["aggregate_sha256"]:
        raise ValueError("M7-B training and validation corpora must differ")
    return experiment, train_manifest, validation_manifest


def initialize_network(
    trainer: RecurrentPPOTrainer,
    checkpoint: Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    protocol = str(payload.get("protocol") or "m6")
    if protocol == "m7b":
        loaded = load_m7b_checkpoint(checkpoint, device="cpu")
    elif protocol == "m7":
        loaded = load_m7_checkpoint(checkpoint, device="cpu")
    else:
        loaded = load_m6_checkpoint(checkpoint, device="cpu")
        protocol = "m6"
    source = loaded.trainer
    structural_fields = (
        "recurrent_size",
        "state_embedding_size",
        "action_embedding_size",
    )
    if any(
        getattr(source.config, field) != getattr(trainer.config, field)
        for field in structural_fields
    ):
        raise ValueError("M7-B initialization checkpoint architecture differs")
    trainer.network.load_state_dict(source.network.state_dict())
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "protocol": protocol,
        "run_seed": loaded.config.run_seed,
        "optimizer_restored": False,
    }


def git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.splitlines() if result.returncode == 0 else [result.stderr.strip()]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if (args.resume is None) == (args.initialize_from is None):
        raise ValueError("M7-B requires exactly one of resume or initialize-from")
    if (args.train_replay_cache is None) != (args.validation_replay_cache is None):
        raise ValueError("M7-B requires both replay caches or neither")
    if args.stop_after_batches is not None and args.stop_after_batches <= 0:
        raise ValueError("stop-after-batches must be positive")
    if args.progress_interval_batches <= 0:
        raise ValueError("progress-interval-batches must be positive")
    if args.stop_file is not None and args.stop_file.exists():
        raise FileExistsError(f"M7-B stop file already exists: {args.stop_file}")
    if args.torch_threads <= 0 or args.torch_interop_threads <= 0:
        raise ValueError("M7-B PyTorch thread counts must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    experiment, ppo_config, run_seeds = load_configuration(
        args.config,
        args.run_seed,
        args.smoke,
    )
    if experiment.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M7-B CUDA training requested but PyTorch cannot access a GPU")
    experiment, train_manifest, validation_manifest = resolve_corpora(
        experiment,
        args.train_corpus,
        args.validation_corpus,
        smoke=args.smoke,
    )
    run_directory = args.output / f"seed-{args.run_seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "checkpoint.pt"
    best_checkpoint_path = run_directory / "best-evaluation-checkpoint.pt"
    metrics_path = run_directory / "metrics.jsonl"
    best_validation_path = run_directory / "best-validation.json"
    initialization = None
    if args.resume is not None:
        loaded = load_m7b_checkpoint(args.resume, device=experiment.device)
        if loaded.manifest.get("evaluation_only"):
            raise ValueError("M7-B evaluation checkpoint cannot resume training")
        if loaded.config != experiment:
            raise ValueError("M7-B resume configuration differs")
        trainer = loaded.trainer
        trainer.config = ppo_config
        trainer.network.config = ppo_config
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = ppo_config.learning_rate
        progress = loaded.progress
        metric_history = list(loaded.metrics)
        manifest = dict(loaded.manifest)
        expected_corpora = dict(manifest.get("corpora") or {})
        if expected_corpora.get("training") != train_manifest["aggregate_sha256"]:
            raise ValueError("M7-B training corpus changed since checkpoint")
        if expected_corpora.get("validation") != validation_manifest["aggregate_sha256"]:
            raise ValueError("M7-B validation corpus changed since checkpoint")
        metrics_path.write_text(
            "".join(
                json.dumps(metric, ensure_ascii=False, sort_keys=True) + "\n"
                for metric in metric_history
            ),
            encoding="utf-8",
        )
    else:
        occupied = tuple(
            path
            for path in (
                checkpoint_path,
                best_checkpoint_path,
                metrics_path,
                best_validation_path,
            )
            if path.exists()
        )
        if occupied:
            raise FileExistsError(
                "M7-B output already contains training artifacts; use --resume: "
                + ", ".join(str(path) for path in occupied)
            )
        trainer = RecurrentPPOTrainer(
            config=ppo_config,
            seed=args.run_seed,
            device=experiment.device,
        )
        assert args.initialize_from is not None
        initialization = initialize_network(trainer, args.initialize_from)
        progress = M7BDistillationProgress()
        metric_history: list[dict[str, Any]] = []
        manifest = build_runtime_manifest(PROJECT_ROOT)
        manifest.update(
            {
                "protocol": "m7b",
                "run_seed": args.run_seed,
                "formal_run_seeds": run_seeds,
                "git_status": git_status(),
                "config_path": str(args.config.resolve()),
                "config_sha256": sha256_file(args.config),
                "initialization": initialization,
                "corpora": {
                    "training": train_manifest["aggregate_sha256"],
                    "validation": validation_manifest["aggregate_sha256"],
                },
                "gate_seed_range": [
                    experiment.gate_seed_start,
                    experiment.gate_seed_start + experiment.gate_seed_count - 1,
                ],
                "final_seed_range": [3_000_000, 3_002_047],
                "training_objective": "noncombat_teacher_cross_entropy_only",
            }
        )
    train_replay = None
    validation_replay = None
    if args.train_replay_cache is not None:
        assert args.validation_replay_cache is not None
        encoder_config = trainer.encoder.config.to_dict()
        train_replay = verify_m7b_replay_manifest(
            args.train_replay_cache,
            expected_corpus_sha256=train_manifest["aggregate_sha256"],
            expected_encoder_config=encoder_config,
            expected_chunk_length=experiment.chunk_length,
            expected_burn_in_steps=experiment.burn_in_steps,
        )
        validation_replay = verify_m7b_replay_manifest(
            args.validation_replay_cache,
            expected_corpus_sha256=validation_manifest["aggregate_sha256"],
            expected_encoder_config=encoder_config,
            expected_chunk_length=experiment.chunk_length,
            expected_burn_in_steps=experiment.burn_in_steps,
        )
        replay_hashes = {
            "training": train_replay["aggregate_sha256"],
            "validation": validation_replay["aggregate_sha256"],
        }
        expected_replay = dict(manifest.get("replay_caches") or {})
        if args.resume is not None and expected_replay != replay_hashes:
            raise ValueError("M7-B replay caches changed since checkpoint")
        manifest["replay_caches"] = replay_hashes
    resolved = {
        "protocol": "m7b",
        "experiment": experiment.to_dict(),
        "ppo": ppo_config.to_dict(),
        "initialization": initialization or manifest.get("initialization"),
        "training_corpus": str(args.train_corpus.resolve()),
        "validation_corpus": str(args.validation_corpus.resolve()),
        "training_replay_cache": (
            None if args.train_replay_cache is None else str(args.train_replay_cache.resolve())
        ),
        "validation_replay_cache": (
            None
            if args.validation_replay_cache is None
            else str(args.validation_replay_cache.resolve())
        ),
        "smoke": args.smoke,
    }
    (run_directory / "resolved-config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_paths = corpus_trace_paths(train_manifest)
    validation_paths = corpus_trace_paths(validation_manifest)
    train_replay_batches = (
        None if train_replay is None else replay_cache_batch_paths(train_replay)
    )
    validation_replay_batches = (
        None
        if validation_replay is None
        else replay_cache_batch_paths(validation_replay)
    )
    environment_factory = LightspeedEnvironmentFactory()

    def save(destination: Path, *, evaluation_only: bool = False) -> None:
        save_m7b_checkpoint(
            destination,
            trainer=trainer,
            config=experiment,
            progress=progress,
            metrics=tuple(metric_history),
            manifest={**manifest, "evaluation_only": evaluation_only},
        )

    if args.stop_after_batches is not None and (
        args.stop_after_batches <= progress.total_trace_batches_completed
    ):
        raise ValueError("stop-after-batches must exceed restored progress")
    started = time.perf_counter()
    stopped = False
    stop_requested = False

    def request_stop(signum: int, _: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            json.dumps(
                {
                    "protocol": "m7b",
                    "state": "stop-requested",
                    "signal": signum,
                    "detail": "finishing the current trace batch before checkpointing",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for epoch in range(progress.next_epoch, experiment.max_epochs):
            if train_replay_batches is None:
                ordered_units = list(train_paths)
                trace_batch_count = math.ceil(
                    len(ordered_units) / experiment.trace_batch_size
                )
            else:
                ordered_units = list(train_replay_batches)
                trace_batch_count = len(ordered_units)
            random.Random(args.run_seed * 1_000_003 + epoch).shuffle(ordered_units)
            start_batch = progress.next_trace_batch if epoch == progress.next_epoch else 0
            for trace_batch_index in range(start_batch, trace_batch_count):
                if train_replay_batches is None:
                    start = trace_batch_index * experiment.trace_batch_size
                    selected_paths = ordered_units[
                        start : start + experiment.trace_batch_size
                    ]
                    traces = tuple(
                        EpisodeTrace.read_jsonl(path) for path in selected_paths
                    )
                    chunks = build_imitation_chunks(
                        environment_factory,
                        trainer,
                        traces,
                        chunk_length=experiment.chunk_length,
                        burn_in_steps=experiment.burn_in_steps,
                        sparse_unsupervised_actions=True,
                    )
                    selected_trace_count = len(selected_paths)
                else:
                    replay_path = ordered_units[trace_batch_index]
                    chunks = load_m7b_replay_batch(
                        replay_path,
                        expected_corpus_sha256=train_manifest["aggregate_sha256"],
                        expected_encoder_config=trainer.encoder.config.to_dict(),
                        expected_chunk_length=experiment.chunk_length,
                        expected_burn_in_steps=experiment.burn_in_steps,
                    )
                    cache_index = int(replay_path.stem.split("-")[-1])
                    selected_trace_count = int(
                        train_replay["files"][cache_index]["trace_count"]
                    )
                scheduled = phase_stratified_imitation_chunks(
                    chunks,
                    maximum_multiplier=experiment.maximum_phase_multiplier,
                    seed=args.run_seed * 10_000_019 + epoch * 10_007 + trace_batch_index,
                )
                optimizer_metrics = []
                for chunk_start in range(0, len(scheduled), experiment.optimizer_batch_chunks):
                    optimizer_metrics.append(
                        train_m7b_chunk_batch(
                            trainer,
                            scheduled[
                                chunk_start : chunk_start
                                + experiment.optimizer_batch_chunks
                            ],
                        )
                    )
                aggregate = merge_m7b_metric_totals(optimizer_metrics)
                record = {
                    "type": "train-batch",
                    "epoch": epoch + 1,
                    "trace_batch": trace_batch_index + 1,
                    "trace_batch_count": trace_batch_count,
                    "trace_count": selected_trace_count,
                    "scheduled_chunks": len(scheduled),
                    "optimizer_steps": len(optimizer_metrics),
                    "mean_loss": sum(metric["loss"] for metric in optimizer_metrics)
                    / len(optimizer_metrics),
                    "mean_gradient_norm": sum(
                        metric["gradient_norm"] for metric in optimizer_metrics
                    )
                    / len(optimizer_metrics),
                    "wall_seconds": time.perf_counter() - started,
                    **aggregate,
                }
                metric_history.append(record)
                append_jsonl(metrics_path, record)
                progress = replace(
                    progress,
                    next_epoch=epoch,
                    next_trace_batch=trace_batch_index + 1,
                    total_trace_batches_completed=(
                        progress.total_trace_batches_completed + 1
                    ),
                )
                if (
                    progress.total_trace_batches_completed
                    % experiment.checkpoint_interval_batches
                    == 0
                ):
                    save(checkpoint_path)
                if (
                    progress.total_trace_batches_completed
                    % args.progress_interval_batches
                    == 0
                ):
                    print(
                        json.dumps(
                            {
                                "protocol": "m7b",
                                "state": "training",
                                "epoch": epoch + 1,
                                "trace_batch": trace_batch_index + 1,
                                "trace_batch_count": trace_batch_count,
                                "total_trace_batches": (
                                    progress.total_trace_batches_completed
                                ),
                                "accuracy": aggregate["accuracy"],
                                "cross_entropy": aggregate["cross_entropy"],
                                "wall_seconds": record["wall_seconds"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                file_stop_requested = (
                    args.stop_file is not None and args.stop_file.exists()
                )
                budget_stop_requested = (
                    args.stop_after_batches is not None
                    and progress.total_trace_batches_completed
                    >= args.stop_after_batches
                )
                if stop_requested or file_stop_requested or budget_stop_requested:
                    save(checkpoint_path)
                    stopped = True
                    break
            if stopped:
                break
            validation = evaluate_m7b_imitation(
                trainer,
                environment_factory,
                validation_paths,
                trace_batch_size=experiment.trace_batch_size,
                optimizer_batch_chunks=experiment.optimizer_batch_chunks,
                chunk_length=experiment.chunk_length,
                burn_in_steps=experiment.burn_in_steps,
                replay_batch_paths=validation_replay_batches,
            )
            selection_key = m7b_validation_selection_key(validation)
            improved = (
                progress.best_validation_key is None
                or selection_key > progress.best_validation_key
            )
            epochs_without_improvement = (
                0 if improved else progress.epochs_without_improvement + 1
            )
            epoch_record = {
                "type": "validation",
                "epoch": epoch + 1,
                "selection_key": list(selection_key),
                "improved": improved,
                "wall_seconds": time.perf_counter() - started,
                **validation,
            }
            metric_history.append(epoch_record)
            append_jsonl(metrics_path, epoch_record)
            completed = (
                epochs_without_improvement >= experiment.early_stopping_patience
                or epoch + 1 >= experiment.max_epochs
            )
            reason = (
                "early_stopping"
                if epochs_without_improvement >= experiment.early_stopping_patience
                else "max_epochs"
                if epoch + 1 >= experiment.max_epochs
                else ""
            )
            progress = M7BDistillationProgress(
                next_epoch=epoch + 1,
                next_trace_batch=0,
                total_trace_batches_completed=progress.total_trace_batches_completed,
                epochs_without_improvement=epochs_without_improvement,
                best_validation_key=(
                    selection_key if improved else progress.best_validation_key
                ),
                completed=completed,
                completion_reason=reason,
            )
            if improved:
                save(best_checkpoint_path, evaluation_only=True)
                best_validation_path.write_text(
                    json.dumps(
                        epoch_record,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            save(checkpoint_path)
            if completed:
                break
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    state = "stopped" if stopped else "complete" if progress.completed else "incomplete"
    print(
        json.dumps(
            {
                "protocol": "m7b",
                "state": state,
                "checkpoint": str(checkpoint_path),
                "best_evaluation_checkpoint": str(best_checkpoint_path),
                "next_epoch": progress.next_epoch,
                "trace_batches": progress.total_trace_batches_completed,
                "completion_reason": progress.completion_reason,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
