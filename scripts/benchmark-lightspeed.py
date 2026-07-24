from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import LightspeedBackend, StsEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the sts_lightspeed Python backend.")
    parser.add_argument("--resets", type=int, default=1_000)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--clones", type=int, default=10_000)
    parser.add_argument("--serializations", type=int, default=5_000)
    parser.add_argument("--throughput-steps", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def latency_summary(samples_ns: list[int]) -> dict[str, float | int]:
    ordered = sorted(samples_ns)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return ordered[index] / 1_000

    return {
        "count": len(ordered),
        "mean_us": statistics.fmean(ordered) / 1_000,
        "p50_us": percentile(0.50),
        "p95_us": percentile(0.95),
        "p99_us": percentile(0.99),
        "max_us": ordered[-1] / 1_000,
    }


def measure(count: int, operation: Callable[[int], None]) -> dict[str, float | int]:
    if count < 1:
        raise ValueError("benchmark counts must be positive")
    samples: list[int] = []
    for index in range(count):
        started = time.perf_counter_ns()
        operation(index)
        samples.append(time.perf_counter_ns() - started)
    return latency_summary(samples)


def rollout_steps(step_count: int, seed: int) -> int:
    random_source = random.Random(seed)
    environment = StsEnv(LightspeedBackend())
    observation, _ = environment.reset(seed=seed)
    completed = 0
    episode_seed = seed
    while completed < step_count:
        observation, _, terminated, truncated, _ = environment.step(
            random_source.randrange(len(observation.legal_actions))
        )
        completed += 1
        if terminated or truncated:
            episode_seed += 1
            observation, _ = environment.reset(seed=episode_seed)
    return completed


def throughput(step_count: int, worker_count: int) -> dict[str, float | int]:
    counts = [step_count // worker_count] * worker_count
    for index in range(step_count % worker_count):
        counts[index] += 1

    started = time.perf_counter()
    if worker_count == 1:
        completed = rollout_steps(counts[0], 10_000)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            completed = sum(
                executor.map(
                    lambda pair: rollout_steps(pair[1], 10_000 + pair[0] * 1_000_000),
                    enumerate(counts),
                )
            )
    elapsed = time.perf_counter() - started
    return {
        "workers": worker_count,
        "steps": completed,
        "seconds": elapsed,
        "steps_per_second": completed / elapsed,
    }


def main() -> int:
    args = parse_args()
    backend = LightspeedBackend()
    reset_result = measure(args.resets, lambda index: backend.reset(seed=index))

    environment = StsEnv(LightspeedBackend())
    observation, _ = environment.reset(seed=20_000)
    random_source = random.Random(20_000)

    def step_operation(_: int) -> None:
        nonlocal observation
        observation, _, terminated, truncated, _ = environment.step(
            random_source.randrange(len(observation.legal_actions))
        )
        if terminated or truncated:
            observation, _ = environment.reset(seed=random_source.randrange(2**32))

    step_result = measure(args.steps, step_operation)
    clone_result = measure(args.clones, lambda _: environment.clone())
    serialization_result = measure(
        args.serializations,
        lambda _: json.dumps(observation.to_dict(), ensure_ascii=False, separators=(",", ":")),
    )

    report = {
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "latency": {
            "reset": reset_result,
            "step": step_result,
            "clone": clone_result,
            "serialize_json": serialization_result,
        },
        "throughput": {
            "single_thread": throughput(args.throughput_steps, 1),
            "multi_thread": throughput(args.throughput_steps, args.threads),
        },
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
