from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import EpisodeTrace, LightspeedBackend, Phase, StsEnv, replay_trace
from sts_env.search import BeliefSearchConfig, ParticleSearchPolicy
from sts_env.training import HeuristicPolicy, bootstrap_mean_interval
from sts_env.training.experiment import build_runtime_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark M7 combat policies on replayed starts.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("heuristic", "search-16", "search-64", "search-256"),
        required=True,
    )
    parser.add_argument("--policy-seed", type=int, default=17)
    parser.add_argument("--max-combat-steps", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_seed(policy_seed: int, trace_index: int, environment_seed: int) -> int:
    digest = hashlib.sha256(
        f"m7-combat:{policy_seed}:{trace_index}:{environment_seed}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def main() -> int:
    args = parse_args()
    if args.max_combat_steps <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("M7 combat benchmark counts must be positive")
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "m7" or manifest.get("purpose") != (
        "paired full-distribution combat benchmark"
    ):
        raise ValueError("unsupported M7 combat corpus")
    records = []
    budget = 0 if args.method == "heuristic" else int(args.method.rsplit("-", 1)[1])
    for trace_index, entry in enumerate(manifest["records"]):
        path = args.corpus / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"combat trace hash mismatch: {path}")
        trace = EpisodeTrace.read_jsonl(path)
        environment = StsEnv(LightspeedBackend())
        observation = replay_trace(environment, trace)
        if observation.phase is not Phase.COMBAT:
            raise ValueError(f"combat trace does not end at combat start: {path}")
        start_hp = observation.player.hp
        heuristic = HeuristicPolicy()
        search = None
        if budget:
            search = ParticleSearchPolicy(
                BeliefSearchConfig(
                    simulations=max(64, budget),
                    simulator_call_budget=budget,
                    max_depth=32,
                    rollout_depth=8,
                ),
                seed=split_seed(args.policy_seed, trace_index, trace.seed),
                rollout_policy=heuristic,
            )
        started = time.perf_counter()
        decisions = 0
        error = ""
        for _ in range(args.max_combat_steps):
            if observation.phase is not Phase.COMBAT:
                break
            try:
                action = (
                    search.select(environment)
                    if search is not None
                    else heuristic(observation)
                )
                observation, _, terminated, truncated, _ = environment.step(action)
                decisions += 1
                if terminated or truncated:
                    break
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
                break
        won_combat = not error and observation.player.hp > 0 and observation.phase is not Phase.COMBAT
        records.append(
            {
                "trace": entry["path"],
                "seed": trace.seed,
                "act": int(entry["act"]),
                "floor": int(entry["floor"]),
                "enemy_ids": list(entry["enemy_ids"]),
                "won_combat": won_combat,
                "start_hp": start_hp,
                "final_hp": observation.player.hp,
                "hp_delta": observation.player.hp - start_hp,
                "decisions": decisions,
                "simulator_calls": (
                    0 if search is None else search.total_simulator_calls
                ),
                "wall_seconds": time.perf_counter() - started,
                "error": error,
            }
        )
    hp_deltas = [float(record["hp_delta"]) for record in records]
    wins = [float(record["won_combat"]) for record in records]
    summary = {
        "episodes": len(records),
        "errors": sum(bool(record["error"]) for record in records),
        "combat_win_rate": statistics.mean(wins),
        "combat_win_rate_bootstrap_ci95": bootstrap_mean_interval(
            wins,
            samples=args.bootstrap_samples,
            seed=args.policy_seed + 1,
        ),
        "mean_hp_delta": statistics.mean(hp_deltas),
        "mean_hp_delta_bootstrap_ci95": bootstrap_mean_interval(
            hp_deltas,
            samples=args.bootstrap_samples,
            seed=args.policy_seed + 2,
        ),
        "mean_decisions": statistics.mean(record["decisions"] for record in records),
        "total_simulator_calls": sum(record["simulator_calls"] for record in records),
        "total_wall_seconds": sum(record["wall_seconds"] for record in records),
        "by_act": {
            str(act): {
                "episodes": len(act_records),
                "combat_win_rate": statistics.mean(
                    float(record["won_combat"]) for record in act_records
                ),
                "mean_hp_delta": statistics.mean(
                    float(record["hp_delta"]) for record in act_records
                ),
            }
            for act in (1, 2, 3)
            if (act_records := [record for record in records if record["act"] == act])
        },
    }
    payload = {
        "protocol": "m7",
        "schema_version": 1,
        "method": args.method,
        "policy_seed": args.policy_seed,
        "search_budget": budget,
        "corpus_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "runtime_manifest": build_runtime_manifest(PROJECT_ROOT),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
