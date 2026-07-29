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

from sts_env.trace import EpisodeTrace
from sts_env.training import (
    LightspeedEnvironmentFactory,
    M7CDaggerTrainingConfig,
    M7CDaggerTrainingProgress,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    load_m6_checkpoint,
    load_m7_checkpoint,
    load_m7b_checkpoint,
    load_m7c_checkpoint,
    m7c_validation_selection_key,
    save_m7c_checkpoint,
)
from sts_env.training.experiment import build_runtime_manifest
from sts_env.training.m7b_distillation import (
    corpus_trace_paths,
    evaluate_m7b_imitation,
    merge_m7b_metric_totals,
    phase_stratified_imitation_chunks,
    sha256_file,
    train_m7b_chunk_batch,
    validate_m7b_training_objective,
    verify_m7b_corpus_manifest,
)
from sts_env.training.m7c_dagger import (
    build_m7c_imitation_chunks,
    evaluate_m7c_imitation,
    m7c_corpus_trace_paths,
    verify_m7c_corpus_manifest,
)
from sts_env.training.self_imitation import build_imitation_chunks
from sts_env.training.m7c_protocol import (
    m7c_frozen_inputs_identity,
    m7c_seed_registry,
    require_registered_seed_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one persistent M7-C DAgger control round."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "m7c_dagger_control.json",
    )
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument(
        "--train-corpus",
        action="append",
        required=True,
        metavar="RANGE_NAME=MANIFEST",
    )
    parser.add_argument(
        "--on-policy-validation-corpus",
        required=True,
        metavar="RANGE_NAME=MANIFEST",
    )
    parser.add_argument(
        "--teacher-corpus",
        type=Path,
        help="frozen M7-B teacher corpus retained in the persistent DAgger aggregate",
    )
    parser.add_argument(
        "--teacher-anchor-validation-corpus",
        metavar="RANGE_NAME=MANIFEST",
    )
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


def parse_named_manifest(value: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name or not path_text:
        raise ValueError("M7-C corpus must use RANGE_NAME=MANIFEST")
    return name, Path(path_text).resolve()


def load_configuration(
    path: Path,
    *,
    run_seed: int,
    round_index: int,
    smoke: bool,
) -> tuple[M7CDaggerTrainingConfig, RecurrentPPOConfig, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "m7c-dagger":
        raise ValueError("M7-C config must declare protocol=m7c-dagger")
    frozen_inputs = dict(payload.get("frozen_inputs") or {})
    frozen_teacher_payload = dict(frozen_inputs.get("teacher_corpus") or {})
    frozen_teacher_range = tuple(
        int(value) for value in frozen_teacher_payload.get("seed_range") or ()
    )
    frozen_teacher_hash = str(frozen_teacher_payload.get("aggregate_sha256") or "")
    frozen_checkpoint_payload = dict(frozen_inputs.get("initial_checkpoint") or {})
    frozen_m6_baseline_payload = dict(
        frozen_inputs.get("m6_baseline_checkpoint") or {}
    )
    expected_frozen_inputs = m7c_frozen_inputs_identity()
    expected_teacher = dict(expected_frozen_inputs["teacher_corpus"])
    expected_checkpoint = dict(expected_frozen_inputs["initial_checkpoint"])
    if (
        frozen_teacher_range
        != (int(expected_teacher["seed_start"]), int(expected_teacher["seed_count"]))
        or frozen_teacher_hash != str(expected_teacher["aggregate_sha256"])
        or frozen_checkpoint_payload != expected_checkpoint
        or frozen_m6_baseline_payload
        != dict(expected_frozen_inputs["m6_baseline_checkpoint"])
    ):
        raise ValueError("M7-C config differs from the frozen input protocol")
    experiment_payload = dict(payload["experiment"])
    run_seeds = tuple(int(seed) for seed in experiment_payload["run_seeds"])
    mixing = tuple(float(value) for value in experiment_payload["teacher_mix_probabilities"])
    training_ranges = tuple(
        tuple(int(value) for value in pair)
        for pair in experiment_payload["training_round_seed_ranges"]
    )
    validation_ranges = tuple(
        tuple(int(value) for value in pair)
        for pair in experiment_payload["on_policy_validation_seed_ranges"]
    )
    if run_seed not in run_seeds or not 0 <= round_index < len(mixing):
        raise ValueError("M7-C run seed or round index is not configured")
    if len(training_ranges) != len(mixing) or len(validation_ranges) != len(mixing):
        raise ValueError("M7-C config has incomplete round ranges")
    registry = m7c_seed_registry()
    expected_ranges = (
        *(registry[f"dagger_round_{index}"] for index in range(len(mixing))),
        registry["teacher_anchor"],
        *(registry[f"on_policy_round_{index}"] for index in range(len(mixing))),
        registry["promotion"],
        registry["formal_gate"],
    )
    configured_ranges = (
        *training_ranges,
        tuple(int(value) for value in experiment_payload["teacher_anchor_validation_seed_range"]),
        *validation_ranges,
        tuple(int(value) for value in experiment_payload["promotion_seed_range"]),
        tuple(int(value) for value in experiment_payload["formal_gate_seed_range"]),
    )
    if tuple((seed_range.start, seed_range.count) for seed_range in expected_ranges) != configured_ranges:
        raise ValueError("M7-C config seed ranges differ from the registered protocol")
    experiment = M7CDaggerTrainingConfig(
        run_seed=run_seed,
        round_index=round_index,
        device=str(experiment_payload["device"]),
        max_epochs=int(experiment_payload["epochs_per_round"]),
        early_stopping_patience=int(experiment_payload["early_stopping_patience"]),
        trace_batch_size=int(experiment_payload["trace_batch_size"]),
        optimizer_batch_chunks=int(experiment_payload["optimizer_batch_chunks"]),
        checkpoint_interval_batches=int(experiment_payload["checkpoint_interval_batches"]),
        chunk_length=int(experiment_payload["chunk_length"]),
        burn_in_steps=int(experiment_payload["burn_in_steps"]),
        maximum_phase_multiplier=int(experiment_payload["maximum_phase_multiplier"]),
    )
    ppo = RecurrentPPOConfig(**payload["ppo"])
    validate_m7b_training_objective(ppo)
    dagger_batch_repeat = int(experiment_payload["dagger_batch_repeat"])
    smoke_teacher_trace_limit = int(
        experiment_payload["smoke_teacher_trace_limit"]
    )
    if dagger_batch_repeat <= 0 or smoke_teacher_trace_limit <= 0:
        raise ValueError("M7-C DAgger batch repeat and smoke teacher limit must be positive")
    if smoke:
        experiment = replace(
            experiment,
            device="cpu",
            max_epochs=2,
            early_stopping_patience=2,
            trace_batch_size=4,
            optimizer_batch_chunks=4,
        )
    protocol = {
        "run_seeds": run_seeds,
        "teacher_mix_probabilities": mixing,
        "training_round_seed_ranges": training_ranges,
        "teacher_anchor_validation_seed_range": tuple(
            int(value)
            for value in experiment_payload["teacher_anchor_validation_seed_range"]
        ),
        "on_policy_validation_seed_ranges": validation_ranges,
        "promotion_seed_range": tuple(
            int(value) for value in experiment_payload["promotion_seed_range"]
        ),
        "formal_gate_seed_range": tuple(
            int(value) for value in experiment_payload["formal_gate_seed_range"]
        ),
        "dagger_batch_repeat": dagger_batch_repeat,
        "smoke_teacher_trace_limit": smoke_teacher_trace_limit,
        "frozen_teacher_corpus": {
            "seed_start": frozen_teacher_range[0],
            "seed_count": frozen_teacher_range[1],
            "aggregate_sha256": frozen_teacher_hash,
        },
        "initial_checkpoint": expected_checkpoint,
        "m6_baseline_checkpoint": dict(
            expected_frozen_inputs["m6_baseline_checkpoint"]
        ),
    }
    return experiment, ppo, protocol


def resolve_manifest(
    value: str,
    *,
    expected_name: str,
    expected_round_index: int,
) -> tuple[str, Path, dict[str, Any]]:
    name, path = parse_named_manifest(value)
    if name != expected_name:
        raise ValueError(f"M7-C expected corpus {expected_name}, got {name}")
    registry = m7c_seed_registry()
    if name not in registry:
        raise ValueError(f"M7-C corpus range is not registered: {name}")
    seed_range = registry[name]
    require_registered_seed_range(
        name,
        start=seed_range.start,
        count=seed_range.count,
        registry=registry,
    )
    manifest = verify_m7c_corpus_manifest(
        path,
        expected_seed_start=seed_range.start,
        expected_seed_count=seed_range.count,
        expected_round_index=expected_round_index,
    )
    return name, path, manifest


def resolve_teacher_anchor_manifest(value: str) -> tuple[Path, dict[str, Any]]:
    name, path = parse_named_manifest(value)
    if name != "teacher_anchor":
        raise ValueError(f"M7-C expected teacher_anchor corpus, got {name}")
    registered = m7c_seed_registry()[name]
    require_registered_seed_range(
        name,
        start=registered.start,
        count=registered.count,
    )
    return path, verify_m7b_corpus_manifest(
        path,
        expected_seed_start=registered.start,
        expected_seed_count=registered.count,
    )


def combine_corpus_hash(manifests: tuple[tuple[str, dict[str, Any]], ...]) -> str:
    digest = hashlib.sha256()
    for name, manifest in manifests:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(manifest["aggregate_sha256"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_frozen_teacher_manifest(
    manifest: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    seed_range = tuple(int(value) for value in manifest["seed_range"])
    expected_seed_range = (
        int(expected["seed_start"]),
        int(expected["seed_start"]) + int(expected["seed_count"]) - 1,
    )
    if seed_range != expected_seed_range:
        raise ValueError("M7-C frozen teacher corpus has an unexpected seed range")
    aggregate_sha256 = str(manifest["aggregate_sha256"])
    if aggregate_sha256 != str(expected["aggregate_sha256"]):
        raise ValueError("M7-C frozen teacher corpus hash differs from the protocol")
    return {
        "seed_range": list(seed_range),
        "trace_count": int(manifest["trace_count"]),
        "aggregate_sha256": aggregate_sha256,
    }


def persistent_training_units(
    *,
    teacher_manifest: dict[str, Any] | None,
    train_entries: tuple[tuple[str, Path, dict[str, Any]], ...],
    dagger_batch_repeat: int,
    maximum_teacher_traces: int | None = None,
) -> tuple[tuple[str, Path], ...]:
    if dagger_batch_repeat <= 0 or (
        maximum_teacher_traces is not None and maximum_teacher_traces <= 0
    ):
        raise ValueError("M7-C persistent aggregate limits must be positive")
    teacher_paths = (
        () if teacher_manifest is None else corpus_trace_paths(teacher_manifest)
    )
    if maximum_teacher_traces is not None:
        teacher_paths = teacher_paths[:maximum_teacher_traces]
    dagger_paths = tuple(
        path
        for _, _, manifest in train_entries
        for path in m7c_corpus_trace_paths(manifest)
    )
    return (
        *(("teacher", path) for path in teacher_paths),
        *(("dagger", path) for path in dagger_paths * dagger_batch_repeat),
    )


def initialize_network(
    trainer: RecurrentPPOTrainer,
    checkpoint: Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    protocol = str(payload.get("protocol") or "m6")
    if protocol == "m7c-dagger":
        loaded = load_m7c_checkpoint(checkpoint, device="cpu")
    elif protocol == "m7b":
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
        raise ValueError("M7-C initialization checkpoint architecture differs")
    trainer.network.load_state_dict(source.network.state_dict())
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "protocol": protocol,
        "run_seed": int(loaded.config.run_seed),
        "round_index": (
            None if protocol != "m7c-dagger" else int(loaded.config.round_index)
        ),
        "completed": (
            None if protocol != "m7c-dagger" else bool(loaded.progress.completed)
        ),
        "evaluation_only": bool(loaded.manifest.get("evaluation_only")),
        "optimizer_restored": False,
    }


def validate_round_initialization(
    initialization: dict[str, Any],
    *,
    round_index: int,
    expected_initial_checkpoint: dict[str, Any],
    smoke: bool,
) -> None:
    if smoke:
        return
    if round_index == 0:
        expected = dict(expected_initial_checkpoint)
        if any(
            initialization.get(name) != value
            for name, value in expected.items()
        ):
            raise ValueError("M7-C round 0 must use the frozen M7-B initialization")
        return
    if (
        initialization.get("protocol") != "m7c-dagger"
        or initialization.get("run_seed") != 17
        or initialization.get("round_index") != round_index - 1
        or initialization.get("evaluation_only") is not True
    ):
        raise ValueError(
            "M7-C later rounds require the selected checkpoint from the prior round"
        )


def validate_behavior_checkpoint(
    manifest: dict[str, Any],
    *,
    initialization: dict[str, Any],
    label: str,
) -> None:
    behavior_policy = dict(manifest.get("behavior_policy") or {})
    if behavior_policy.get("checkpoint_sha256") != initialization.get("sha256"):
        raise ValueError(
            f"M7-C {label} corpus was not collected by this round's initialization checkpoint"
        )


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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if (args.resume is None) == (args.initialize_from is None):
        raise ValueError("M7-C requires exactly one of resume or initialize-from")
    if args.stop_after_batches is not None and args.stop_after_batches <= 0:
        raise ValueError("stop-after-batches must be positive")
    if args.progress_interval_batches <= 0:
        raise ValueError("progress-interval-batches must be positive")
    if args.stop_file is not None and args.stop_file.exists():
        raise FileExistsError(f"M7-C stop file already exists: {args.stop_file}")
    if args.torch_threads <= 0 or args.torch_interop_threads <= 0:
        raise ValueError("M7-C PyTorch thread counts must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    experiment, ppo_config, protocol = load_configuration(
        args.config,
        run_seed=args.run_seed,
        round_index=args.round_index,
        smoke=args.smoke,
    )
    if experiment.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M7-C CUDA training requested but PyTorch cannot access a GPU")
    expected_training_names = tuple(
        f"dagger_round_{index}" for index in range(args.round_index + 1)
    )
    if args.smoke:
        expected_training_names = ("m7c_smoke",)
    if len(args.train_corpus) != len(expected_training_names):
        raise ValueError("M7-C train corpora do not match the expected round count")
    train_entries = tuple(
        resolve_manifest(
            value,
            expected_name=expected_name,
            expected_round_index=(0 if args.smoke else index),
        )
        for index, (value, expected_name) in enumerate(
            zip(args.train_corpus, expected_training_names, strict=True)
        )
    )
    on_policy_name = f"on_policy_round_{args.round_index}"
    if args.smoke:
        on_policy_name = "m7c_smoke_validation"
    _, on_policy_path, on_policy_manifest = resolve_manifest(
        args.on_policy_validation_corpus,
        expected_name=on_policy_name,
        expected_round_index=args.round_index,
    )
    anchor_path = None
    anchor_manifest = None
    if args.teacher_anchor_validation_corpus is not None:
        anchor_path, anchor_manifest = resolve_teacher_anchor_manifest(
            args.teacher_anchor_validation_corpus
        )
    elif not args.smoke:
        raise ValueError("formal M7-C training requires a teacher-anchor validation corpus")
    train_manifests = tuple((name, manifest) for name, _, manifest in train_entries)
    teacher_manifest = (
        None
        if args.teacher_corpus is None
        else verify_m7b_corpus_manifest(
            args.teacher_corpus,
            expected_seed_start=int(protocol["frozen_teacher_corpus"]["seed_start"]),
            expected_seed_count=int(protocol["frozen_teacher_corpus"]["seed_count"]),
        )
    )
    if teacher_manifest is None and not args.smoke:
        raise ValueError("formal M7-C training requires --teacher-corpus")
    teacher_identity = (
        None
        if teacher_manifest is None
        else validate_frozen_teacher_manifest(
            teacher_manifest,
            expected=protocol["frozen_teacher_corpus"],
        )
    )
    persistent_manifests = (
        train_manifests
        if teacher_manifest is None
        else (
            ("frozen_teacher", teacher_manifest),
            *train_manifests,
        )
    )
    train_hash = combine_corpus_hash(persistent_manifests)
    validation_hashes = {
        "on_policy": str(on_policy_manifest["aggregate_sha256"]),
        "teacher_anchor": (
            None if anchor_manifest is None else str(anchor_manifest["aggregate_sha256"])
        ),
    }
    if any(
        manifest["aggregate_sha256"] == on_policy_manifest["aggregate_sha256"]
        for _, manifest in train_manifests
    ):
        raise ValueError("M7-C training and on-policy validation corpora must differ")
    if anchor_manifest is not None and any(
        manifest["aggregate_sha256"] == anchor_manifest["aggregate_sha256"]
        for _, manifest in persistent_manifests
    ):
        raise ValueError("M7-C training and teacher-anchor corpora must differ")
    run_directory = args.output / f"seed-{args.run_seed}" / f"round-{args.round_index}"
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "checkpoint.pt"
    best_checkpoint_path = run_directory / "best-evaluation-checkpoint.pt"
    metrics_path = run_directory / "metrics.jsonl"
    best_validation_path = run_directory / "best-validation.json"
    initialization = None
    if args.resume is not None:
        loaded = load_m7c_checkpoint(args.resume, device=experiment.device)
        if loaded.manifest.get("evaluation_only"):
            raise ValueError("M7-C evaluation checkpoint cannot resume training")
        if loaded.config != experiment:
            raise ValueError("M7-C resume configuration differs")
        trainer = loaded.trainer
        trainer.config = ppo_config
        trainer.network.config = ppo_config
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = ppo_config.learning_rate
        progress = loaded.progress
        metric_history = list(loaded.metrics)
        manifest = dict(loaded.manifest)
        if manifest.get("training_corpus_sha256") != train_hash:
            raise ValueError("M7-C training corpora changed since checkpoint")
        if dict(manifest.get("validation_corpora") or {}) != validation_hashes:
            raise ValueError("M7-C validation corpora changed since checkpoint")
        if manifest.get("frozen_teacher_corpus") != teacher_identity:
            raise ValueError("M7-C frozen teacher corpus changed since checkpoint")
        metrics_path.write_text(
            "".join(
                json.dumps(metric, ensure_ascii=False, sort_keys=True) + "\n"
                for metric in metric_history
            ),
            encoding="utf-8",
            newline="\n",
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
                "M7-C output already contains training artifacts; use --resume: "
                + ", ".join(str(path) for path in occupied)
            )
        trainer = RecurrentPPOTrainer(
            config=ppo_config,
            seed=args.run_seed,
            device=experiment.device,
        )
        assert args.initialize_from is not None
        initialization = initialize_network(trainer, args.initialize_from.resolve())
        validate_round_initialization(
            initialization,
            round_index=args.round_index,
            expected_initial_checkpoint=protocol["initial_checkpoint"],
            smoke=args.smoke,
        )
        progress = M7CDaggerTrainingProgress()
        metric_history: list[dict[str, Any]] = []
        manifest = build_runtime_manifest(PROJECT_ROOT)
        manifest.update(
            {
                "protocol": "m7c-dagger",
                "run_seed": args.run_seed,
                "round_index": args.round_index,
                "formal_run_seeds": list(protocol["run_seeds"]),
                "git_status": git_status(),
                "config_path": str(args.config.resolve()),
                "config_sha256": sha256_file(args.config),
                "initialization": initialization,
                "training_corpus_sha256": train_hash,
                "frozen_teacher_corpus": teacher_identity,
                "training_corpora": [
                    {
                        "range_name": name,
                        "sha256": corpus["aggregate_sha256"],
                        "seed_range": corpus["seed_range"],
                    }
                    for name, corpus in persistent_manifests
                ],
                "dagger_batch_repeat": protocol["dagger_batch_repeat"],
                "smoke_teacher_trace_limit": (
                    protocol["smoke_teacher_trace_limit"] if args.smoke else None
                ),
                "validation_corpora": validation_hashes,
                "teacher_mix_probability": protocol["teacher_mix_probabilities"][
                    args.round_index
                ],
                "training_objective": "persistent_on_policy_teacher_cross_entropy_only",
            }
        )
    behavior_initialization = dict(initialization or manifest.get("initialization") or {})
    if not behavior_initialization:
        raise ValueError("M7-C training manifest lacks behavior checkpoint provenance")
    validate_behavior_checkpoint(
        train_entries[-1][2],
        initialization=behavior_initialization,
        label=train_entries[-1][0],
    )
    validate_behavior_checkpoint(
        on_policy_manifest,
        initialization=behavior_initialization,
        label=on_policy_name,
    )
    resolved = {
        "protocol": "m7c-dagger",
        "experiment": experiment.to_dict(),
        "ppo": ppo_config.to_dict(),
        "initialization": initialization or manifest.get("initialization"),
        "training_corpora": [str(path) for _, path, _ in train_entries],
        "teacher_corpus": (
            None if args.teacher_corpus is None else str(args.teacher_corpus.resolve())
        ),
        "frozen_teacher_corpus": teacher_identity,
        "on_policy_validation_corpus": str(on_policy_path),
        "teacher_anchor_validation_corpus": (
            None if anchor_path is None else str(anchor_path)
        ),
        "smoke": args.smoke,
    }
    atomic_write_json(run_directory / "resolved-config.json", resolved)
    atomic_write_json(run_directory / "manifest.json", manifest)
    train_units = persistent_training_units(
        teacher_manifest=teacher_manifest,
        train_entries=train_entries,
        dagger_batch_repeat=protocol["dagger_batch_repeat"],
        maximum_teacher_traces=(
            protocol["smoke_teacher_trace_limit"] if args.smoke else None
        ),
    )
    if not train_units:
        raise ValueError("M7-C persistent aggregate contains no training traces")
    on_policy_paths = m7c_corpus_trace_paths(on_policy_manifest)
    anchor_paths = (
        None if anchor_manifest is None else corpus_trace_paths(anchor_manifest)
    )
    environment_factory = LightspeedEnvironmentFactory()

    def save(destination: Path, *, evaluation_only: bool = False) -> None:
        save_m7c_checkpoint(
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
                    "protocol": "m7c-dagger",
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
            ordered_units = list(train_units)
            random.Random(args.run_seed * 1_000_003 + epoch).shuffle(ordered_units)
            trace_batch_count = math.ceil(len(ordered_units) / experiment.trace_batch_size)
            start_batch = progress.next_trace_batch if epoch == progress.next_epoch else 0
            for trace_batch_index in range(start_batch, trace_batch_count):
                start = trace_batch_index * experiment.trace_batch_size
                selected_units = ordered_units[start : start + experiment.trace_batch_size]
                selected_teacher_paths = tuple(
                    path for source, path in selected_units if source == "teacher"
                )
                selected_dagger_paths = tuple(
                    path for source, path in selected_units if source == "dagger"
                )
                teacher_chunks = (
                    ()
                    if not selected_teacher_paths
                    else build_imitation_chunks(
                        environment_factory,
                        trainer,
                        tuple(
                            EpisodeTrace.read_jsonl(path)
                            for path in selected_teacher_paths
                        ),
                        chunk_length=experiment.chunk_length,
                        burn_in_steps=experiment.burn_in_steps,
                        sparse_unsupervised_actions=True,
                    )
                )
                dagger_chunks = (
                    ()
                    if not selected_dagger_paths
                    else build_m7c_imitation_chunks(
                        environment_factory,
                        trainer,
                        tuple(
                            EpisodeTrace.read_jsonl(path)
                            for path in selected_dagger_paths
                        ),
                        chunk_length=experiment.chunk_length,
                        burn_in_steps=experiment.burn_in_steps,
                    )
                )
                chunks = (*teacher_chunks, *dagger_chunks)
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
                                chunk_start : chunk_start + experiment.optimizer_batch_chunks
                            ],
                        )
                    )
                aggregate = merge_m7b_metric_totals(optimizer_metrics)
                record = {
                    "type": "train-batch",
                    "epoch": epoch + 1,
                    "trace_batch": trace_batch_index + 1,
                    "trace_batch_count": trace_batch_count,
                    "trace_count": len(selected_units),
                    "teacher_trace_count": len(selected_teacher_paths),
                    "dagger_trace_count": len(selected_dagger_paths),
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
                                "protocol": "m7c-dagger",
                                "state": "training",
                                "epoch": epoch + 1,
                                "trace_batch": trace_batch_index + 1,
                                "trace_batch_count": trace_batch_count,
                                "total_trace_batches": progress.total_trace_batches_completed,
                                "accuracy": aggregate["accuracy"],
                                "cross_entropy": aggregate["cross_entropy"],
                                "wall_seconds": record["wall_seconds"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                file_stop_requested = args.stop_file is not None and args.stop_file.exists()
                budget_stop_requested = (
                    args.stop_after_batches is not None
                    and progress.total_trace_batches_completed >= args.stop_after_batches
                )
                if stop_requested or file_stop_requested or budget_stop_requested:
                    save(checkpoint_path)
                    stopped = True
                    break
            if stopped:
                break
            on_policy_validation = evaluate_m7c_imitation(
                trainer,
                environment_factory,
                on_policy_paths,
                trace_batch_size=experiment.trace_batch_size,
                optimizer_batch_chunks=experiment.optimizer_batch_chunks,
                chunk_length=experiment.chunk_length,
                burn_in_steps=experiment.burn_in_steps,
            )
            anchor_validation = (
                None
                if anchor_paths is None
                else evaluate_m7b_imitation(
                    trainer,
                    environment_factory,
                    anchor_paths,
                    trace_batch_size=experiment.trace_batch_size,
                    optimizer_batch_chunks=experiment.optimizer_batch_chunks,
                    chunk_length=experiment.chunk_length,
                    burn_in_steps=experiment.burn_in_steps,
                )
            )
            selection_key = m7c_validation_selection_key(
                anchor_validation,
                on_policy_validation,
                require_all_phases=not args.smoke,
            )
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
                "on_policy": on_policy_validation,
                "teacher_anchor": anchor_validation,
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
            progress = M7CDaggerTrainingProgress(
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
                atomic_write_json(best_validation_path, epoch_record)
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
                "protocol": "m7c-dagger",
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
