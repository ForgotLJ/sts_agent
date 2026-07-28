from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import (
    EpisodeTrace,
    LightspeedBackend,
    Phase,
    StsEnv,
    TraceStep,
    observation_digest,
)
from sts_env.training import HeuristicPolicy, M7_FINAL_SEED_END, M7_FINAL_SEED_START
from sts_env.training.experiment import build_runtime_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect replayable M7 combat starts.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=10_000)
    parser.add_argument("--per-act", type=int, default=2_000)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.seed_count <= 0 or args.per_act <= 0 or args.max_steps <= 0:
        raise ValueError("M7 combat corpus counts must be positive")
    seed_end = args.seed_start + args.seed_count - 1
    if max(args.seed_start, M7_FINAL_SEED_START) <= min(seed_end, M7_FINAL_SEED_END):
        raise ValueError("M7 combat corpus cannot use formal final seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    policy = HeuristicPolicy()
    counts = {1: 0, 2: 0, 3: 0}
    records: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        if all(count >= args.per_act for count in counts.values()):
            break
        environment = StsEnv(LightspeedBackend())
        observation, info = environment.reset(seed=seed)
        initial_digest = observation_digest(observation)
        steps: list[TraceStep] = []
        previous_phase = None
        for _ in range(args.max_steps):
            if (
                observation.phase is Phase.COMBAT
                and previous_phase is not Phase.COMBAT
                and observation.act in counts
                and counts[observation.act] < args.per_act
            ):
                trace = EpisodeTrace(
                    seed=seed,
                    initial_observation_digest=initial_digest,
                    steps=tuple(steps),
                    backend=str(info.get("backend", "sts_lightspeed")),
                    metadata={
                        "purpose": "m7-combat-start",
                        "source_policy": "heuristic",
                        "act": observation.act,
                        "floor": observation.floor,
                        "enemy_ids": [
                            enemy.monster_id or enemy.name for enemy in observation.enemies
                        ],
                        "public_observation_digest": observation_digest(observation),
                    },
                )
                index = counts[observation.act]
                relative = Path(f"act-{observation.act}") / (
                    f"combat-{index:06d}-floor-{observation.floor:02d}-seed-{seed}.jsonl"
                )
                path = args.output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                trace.write_jsonl(path)
                records.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": sha256_file(path),
                        "size": path.stat().st_size,
                        "seed": seed,
                        "act": observation.act,
                        "floor": observation.floor,
                        "enemy_ids": trace.metadata["enemy_ids"],
                    }
                )
                counts[observation.act] += 1
            if observation.phase is Phase.TERMINAL:
                break
            previous_phase = observation.phase
            action = policy(observation)
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
            if terminated or truncated:
                break
    manifest = {
        "protocol": "m7",
        "schema_version": 1,
        "purpose": "paired full-distribution combat benchmark",
        "source_policy": "heuristic",
        "seed_range": [args.seed_start, seed_end],
        "target_per_act": args.per_act,
        "counts": {str(act): count for act, count in counts.items()},
        "records": records,
        "runtime_manifest": build_runtime_manifest(PROJECT_ROOT),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "records": len(records),
                "counts": manifest["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
