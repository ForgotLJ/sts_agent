from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.differential import canonical_observation
from sts_env.training import (
    CurriculumEnvironmentFactory,
    DaggerConfig,
    HeuristicPolicy,
    HierarchicalRecurrentPolicy,
    PrefixCorpus,
    collect_dagger_chunks,
    dagger_training_seeds,
    load_m6_checkpoint,
    train_self_imitation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sequential weighted DAgger rounds from one M6 checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "m6_recurrent_ppo.json",
    )
    parser.add_argument("--update-index", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--validation-seed-start", type=int, default=1100000)
    parser.add_argument("--validation-seed-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def evaluate_stage(
    trainer: Any,
    factory: CurriculumEnvironmentFactory,
    seeds: tuple[int, ...],
    max_steps: int,
    heuristic: HeuristicPolicy,
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
            combat_selector=lambda current: heuristic(current.observation),
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
            episode_completed = episode_completed or bool(
                info.get("curriculum_completed", False)
            )
            if terminated or truncated:
                completed += int(
                    episode_completed or float(info.get("raw_reward", 0.0)) > 0
                )
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


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    dagger = DaggerConfig(**payload["dagger"])
    curriculum = dict(payload["curriculum"])
    experiment = dict(payload["experiment"])
    loaded = load_m6_checkpoint(args.checkpoint, device=args.device)
    if loaded.scheduler.current.name != "full_run":
        raise ValueError("DAgger round comparison requires a full-run checkpoint")
    update_index = args.update_index
    if update_index is None:
        update_index = (
            (loaded.update_index // dagger.interval_updates + 1) * dagger.interval_updates
        )
    if update_index <= loaded.update_index or update_index % dagger.interval_updates != 0:
        raise ValueError("comparison update must be the next or a later DAgger update")
    maximum_rounds = dagger.rounds if args.max_rounds is None else args.max_rounds
    if maximum_rounds <= 0 or maximum_rounds > dagger.rounds:
        raise ValueError("max-rounds must be within the configured DAgger rounds")
    validation_seeds = tuple(
        range(
            args.validation_seed_start,
            args.validation_seed_start + args.validation_seed_count,
        )
    )
    if not validation_seeds or args.validation_seed_start < int(
        experiment["validation_seed_start"]
    ):
        raise ValueError("comparison requires non-empty validation seeds")
    validation_stop = int(experiment["validation_seed_start"]) + int(
        experiment["validation_seed_count"]
    )
    if validation_seeds[-1] >= validation_stop:
        raise ValueError("comparison validation seeds exceed the configured validation split")
    spec = loaded.scheduler.current
    corpus = None
    if spec.use_prefix_starts:
        corpus = PrefixCorpus.read(
            args.checkpoint.parent / "curriculum" / f"act-{spec.start_act}"
        )
    factory = CurriculumEnvironmentFactory(spec=spec, prefix_corpus=corpus)
    heuristic = HeuristicPolicy()
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_update": loaded.update_index,
        "comparison_update": update_index,
        "stage": spec.name,
        "validation_seed_range": [validation_seeds[0], validation_seeds[-1]],
        "dagger": dagger.to_dict(),
        "evaluations": [],
    }
    baseline = evaluate_stage(
        loaded.trainer,
        factory,
        validation_seeds,
        int(curriculum["max_episode_steps"]),
        heuristic,
    )
    result["evaluations"].append({"rounds": 0, "validation": baseline})
    write_result(args.output, result)
    print(json.dumps(result["evaluations"][-1], sort_keys=True), flush=True)
    for round_index in range(maximum_rounds):
        seeds = dagger_training_seeds(
            dagger,
            training_seed_start=int(experiment["training_seed_start"]),
            training_seed_count=int(experiment["training_seed_count"]),
            update_index=update_index,
            round_index=round_index,
        )
        chunks = collect_dagger_chunks(
            factory,
            loaded.trainer,
            heuristic,
            seeds,
            max_steps=dagger.max_steps,
            chunk_length=dagger.chunk_length,
            burn_in_steps=dagger.burn_in_steps,
            phase_weights=dagger.phase_weights(),
        )
        if not chunks:
            raise RuntimeError("DAgger round collected no supervised non-combat decisions")
        training = train_self_imitation(
            loaded.trainer,
            chunks,
            epochs=dagger.epochs,
            seed=(
                loaded.config.run_seed * 1_000_003
                + update_index * dagger.rounds
                + round_index
            ),
        )
        validation = evaluate_stage(
            loaded.trainer,
            factory,
            validation_seeds,
            int(curriculum["max_episode_steps"]),
            heuristic,
        )
        result["evaluations"].append(
            {
                "rounds": round_index + 1,
                "seed_range": [seeds[0], seeds[-1]],
                "training": training,
                "validation": validation,
            }
        )
        write_result(args.output, result)
        print(json.dumps(result["evaluations"][-1], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
