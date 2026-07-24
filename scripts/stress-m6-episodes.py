from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tracemalloc

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import (
    CurriculumEnvironmentFactory,
    CurriculumSpec,
    HeuristicPolicy,
    MultiprocessRecurrentRolloutCollector,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
    SubprocessVectorEnvironment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress the M6 episode collector.")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=300_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.episodes <= 0
        or args.workers <= 0
        or args.rollout_steps <= 0
        or args.progress_interval <= 0
        or args.torch_threads <= 0
    ):
        raise ValueError("stress counts must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    factory = CurriculumEnvironmentFactory(
        CurriculumSpec(
            "stress",
            completion_reward=0.0,
            potential_scale=0.0,
            progress_reward_per_floor=0.0,
            max_episode_steps=1000,
        )
    )
    trainer = RecurrentPPOTrainer(
        config=RecurrentPPOConfig(
            recurrent_size=128,
            state_embedding_size=128,
            action_embedding_size=128,
            update_epochs=1,
            minibatch_environments=args.workers,
        ),
        seed=17,
        device=args.device,
    )
    pool = SubprocessVectorEnvironment(factory, args.workers)
    collector = MultiprocessRecurrentRolloutCollector(
        pool,
        trainer,
        seeds=tuple(range(args.seed_start, args.seed_start + max(args.episodes * 2, 1000))),
        combat_selector=HeuristicPolicy(),
    )
    completed = 0
    updates = 0
    next_progress = args.progress_interval
    started = time.perf_counter()
    tracemalloc.start()
    try:
        while completed < args.episodes:
            rollout = collector.collect(args.rollout_steps)
            trainer.update(rollout)
            completed += len(rollout.completed_episodes)
            updates += 1
            if args.output is not None and completed >= next_progress:
                current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                progress_path = args.output.with_name(
                    args.output.stem + ".progress" + args.output.suffix
                )
                progress_path.write_text(
                    json.dumps(
                        {
                            "complete": False,
                            "episodes": completed,
                            "target_episodes": args.episodes,
                            "updates": updates,
                            "current_tracemalloc_bytes": current_bytes,
                            "peak_tracemalloc_bytes": peak_bytes,
                            "wall_seconds": time.perf_counter() - started,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                while next_progress <= completed:
                    next_progress += args.progress_interval
    finally:
        collector.close()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload = {
        "episodes": completed,
        "target_episodes": args.episodes,
        "updates": updates,
        "workers": args.workers,
        "rollout_steps": args.rollout_steps,
        "seed_start": args.seed_start,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "peak_tracemalloc_bytes": peak_bytes,
        "wall_seconds": time.perf_counter() - started,
        "errors": 0,
        "complete": True,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        args.output.write_text(serialized, encoding="utf-8")
        progress_path = args.output.with_name(
            args.output.stem + ".progress" + args.output.suffix
        )
        progress_path.write_text(serialized, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
