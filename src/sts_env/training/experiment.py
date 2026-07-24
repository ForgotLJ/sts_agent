from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from sts_env import StsEnv, ToyCombatBackend
from sts_env.training.candidate_q import CandidateQConfig, CandidateQTrainer
from sts_env.training.collector import SynchronousVectorCollector
from sts_env.training.evaluation import EvaluationSummary, evaluate_policy
from sts_env.training.policies import HeuristicPolicy, OneStepSearchPolicy, RandomPolicy
from sts_env.training.seeds import SeedSplit


@dataclass(frozen=True, slots=True)
class ToyExperimentConfig:
    run_seeds: tuple[int, ...] = (17, 29, 43)
    total_steps: int = 8_000
    num_environments: int = 8
    collection_chunk: int = 32
    train_every: int = 4
    updates_per_train: int = 1
    training_seed_start: int = 0
    training_seed_count: int = 20_000
    evaluation_seed_start: int = 1_000_000
    evaluation_seed_count: int = 256
    max_episode_steps: int = 500
    device: str = "cpu"
    save_replay_jsonl: bool = False

    def __post_init__(self) -> None:
        if len(self.run_seeds) < 3:
            raise ValueError("formal M4 evaluation requires at least three run seeds")
        if min(
            self.total_steps,
            self.num_environments,
            self.collection_chunk,
            self.train_every,
            self.updates_per_train,
        ) <= 0:
            raise ValueError("training counts must be positive")
        SeedSplit(
            self.training_seed_start,
            self.training_seed_count,
            self.evaluation_seed_start,
            self.evaluation_seed_count,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_seeds"] = list(self.run_seeds)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToyExperimentConfig:
        values = dict(payload)
        values["run_seeds"] = tuple(int(seed) for seed in values["run_seeds"])
        return cls(**values)


def run_toy_candidate_q_experiment(
    output_directory: str | Path,
    experiment_config: ToyExperimentConfig,
    trainer_config: CandidateQConfig,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "experiment": experiment_config.to_dict(),
                "trainer": trainer_config.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split = SeedSplit(
        experiment_config.training_seed_start,
        experiment_config.training_seed_count,
        experiment_config.evaluation_seed_start,
        experiment_config.evaluation_seed_count,
    )
    runtime_manifest = build_runtime_manifest(Path(__file__).resolve().parents[3])
    evaluation_seeds = split.evaluation_seeds
    environment_factory = lambda: StsEnv(ToyCombatBackend())
    random_summary = evaluate_policy(
        environment_factory,
        RandomPolicy(seed=0xBAD5EED),
        evaluation_seeds,
        max_steps=experiment_config.max_episode_steps,
    )
    heuristic_summary = evaluate_policy(
        environment_factory,
        HeuristicPolicy(),
        evaluation_seeds,
        max_steps=experiment_config.max_episode_steps,
    )
    search_summary = evaluate_policy(
        environment_factory,
        OneStepSearchPolicy(),
        evaluation_seeds,
        max_steps=experiment_config.max_episode_steps,
    )
    run_summaries: list[dict[str, Any]] = []
    for run_seed in experiment_config.run_seeds:
        run_directory = output / f"seed-{run_seed}"
        run_directory.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_directory / "tensorboard"))
        trainer = CandidateQTrainer(
            config=trainer_config,
            seed=run_seed,
            device=experiment_config.device,
        )
        collector = SynchronousVectorCollector(
            environment_factory=environment_factory,
            num_environments=experiment_config.num_environments,
            seeds=split.training_seeds,
        )
        completed_episodes = 0
        training_reward_sum = 0.0
        while trainer.environment_steps < experiment_config.total_steps:
            remaining = experiment_config.total_steps - trainer.environment_steps
            batch = collector.collect(
                min(experiment_config.collection_chunk, remaining),
                lambda observation, _: trainer.select_action(observation, explore=True),
            )
            for transition in batch.transitions:
                training_reward_sum += trainer.observe(transition)
                if trainer.environment_steps % experiment_config.train_every == 0:
                    metrics = trainer.train_step(experiment_config.updates_per_train)
                    if metrics is not None:
                        writer.add_scalar("train/loss", metrics["loss"], trainer.environment_steps)
                        writer.add_scalar(
                            "train/gradient_norm",
                            metrics["gradient_norm"],
                            trainer.environment_steps,
                        )
                writer.add_scalar("train/epsilon", trainer.epsilon, trainer.environment_steps)
            for episode in batch.completed_episodes:
                completed_episodes += 1
                writer.add_scalar(
                    "episode/environment_return",
                    episode.environment_return,
                    completed_episodes,
                )
                writer.add_scalar("episode/length", episode.length, completed_episodes)
                writer.add_scalar("episode/final_hp", episode.final_hp, completed_episodes)
        learned_summary = evaluate_policy(
            environment_factory,
            lambda observation, _: trainer.greedy_action(observation),
            evaluation_seeds,
            max_steps=experiment_config.max_episode_steps,
        )
        writer.add_scalar("evaluation/mean_score", learned_summary.mean_score, trainer.environment_steps)
        writer.add_scalar("evaluation/win_rate", learned_summary.win_rate, trainer.environment_steps)
        writer.flush()
        writer.close()
        checkpoint_path = run_directory / "checkpoint.pt"
        metadata = {
            "evaluation": learned_summary.to_dict(),
            "training_seed_range": [
                split.training_seeds[0],
                split.training_seeds[-1],
            ],
            "evaluation_seed_range": [evaluation_seeds[0], evaluation_seeds[-1]],
            "runtime_manifest": runtime_manifest,
        }
        trainer.save_checkpoint(checkpoint_path, metadata=metadata)
        if experiment_config.save_replay_jsonl:
            trainer.replay.write_jsonl(run_directory / "replay.jsonl")
        run_summary = {
            "run_seed": run_seed,
            "environment_steps": trainer.environment_steps,
            "gradient_steps": trainer.gradient_steps,
            "completed_episodes": completed_episodes,
            "mean_training_reward_per_step": training_reward_sum / trainer.environment_steps,
            "last_loss": trainer.last_loss,
            "last_gradient_norm": trainer.last_gradient_norm,
            "checkpoint": str(checkpoint_path),
            "evaluation": learned_summary.to_dict(),
        }
        (run_directory / "summary.json").write_text(
            json.dumps(run_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run_summaries.append(run_summary)
    learned_scores = [run["evaluation"]["mean_score"] for run in run_summaries]
    learned_mean = statistics.mean(learned_scores)
    learned_std = statistics.stdev(learned_scores) if len(learned_scores) > 1 else 0.0
    aggregate = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": experiment_config.device,
        },
        "runtime_manifest": runtime_manifest,
        "seed_split": {
            "training": [split.training_seeds[0], split.training_seeds[-1]],
            "evaluation": [evaluation_seeds[0], evaluation_seeds[-1]],
        },
        "baselines": {
            "random": random_summary.to_dict(),
            "heuristic": heuristic_summary.to_dict(),
            "one_step_search": search_summary.to_dict(),
        },
        "candidate_q_runs": run_summaries,
        "candidate_q_mean_score": learned_mean,
        "candidate_q_run_ci95": 1.96 * learned_std / math.sqrt(len(learned_scores)),
        "improvement_over_random": learned_mean - random_summary.mean_score,
        "claim_supported": learned_mean > random_summary.mean_score,
    }
    (output / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return aggregate


def build_runtime_manifest(project_root: Path) -> dict[str, Any]:
    git_commit = "unavailable"
    git_dirty: bool | str = "unavailable"
    try:
        git_commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "-C", str(project_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    source_hash = hashlib.sha256()
    included_paths: list[str] = []
    for relative_root in (
        "src",
        "scripts",
        "config",
        "tests",
        "vendor/sts_lightspeed/bindings",
        "vendor/sts_lightspeed/include",
        "vendor/sts_lightspeed/src",
        "vendor/CommunicationMod/src",
    ):
        root = project_root / relative_root
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyd"}:
                continue
            relative = path.relative_to(project_root).as_posix()
            included_paths.append(relative)
            source_hash.update(relative.encode("utf-8"))
            source_hash.update(b"\0")
            source_hash.update(path.read_bytes())
            source_hash.update(b"\0")
    for relative in (
        "pyproject.toml",
        "docs/M6_IMPLEMENTATION_AND_EVALUATION_PLAN.md",
        "vendor/sts_lightspeed/CMakeLists.txt",
        "vendor/CommunicationMod/pom.xml",
    ):
        path = project_root / relative
        if not path.is_file():
            continue
        included_paths.append(relative)
        source_hash.update(relative.encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(path.read_bytes())
        source_hash.update(b"\0")
    runtime_artifacts: dict[str, dict[str, Any]] = {}
    artifact_paths = {
        path
        for suffix in EXTENSION_SUFFIXES
        for path in (project_root / "build").glob(
            f"sts_lightspeed-*/slaythespire*{suffix}"
        )
    }
    artifact_paths.update((project_root / "build").glob("CommunicationMod/*.jar"))
    for path in sorted(artifact_paths):
        relative = path.relative_to(project_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        runtime_artifacts[relative] = {
            "sha256": digest,
            "size": path.stat().st_size,
        }
        source_hash.update(relative.encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(bytes.fromhex(digest))
        source_hash.update(b"\0")
    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "source_sha256": source_hash.hexdigest(),
        "source_file_count": len(included_paths),
        "runtime_artifacts": runtime_artifacts,
        "dependencies": {
            "torch": torch.__version__,
            "tensorboard": importlib.metadata.version("tensorboard"),
        },
        "hardware": {
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
