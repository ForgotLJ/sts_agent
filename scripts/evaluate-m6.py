from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import LightspeedBackend, Phase, StsEnv
from sts_env.search import BeliefSearchConfig, ParticleSearchPolicy
from sts_env.training import (
    HeuristicPolicy,
    HierarchicalRecurrentPolicy,
    M6_FINAL_SEED_END,
    M6_FINAL_SEED_START,
    RandomPolicy,
    evaluate_full_runs,
    load_m6_checkpoint,
    validate_m6_evaluation_seed_range,
)
from sts_env.training.experiment import build_runtime_manifest


class HeuristicSearchPolicy:
    def __init__(self, search: ParticleSearchPolicy):
        self.search = search
        self.heuristic = HeuristicPolicy()

    @property
    def total_simulator_calls(self) -> int:
        return self.search.total_simulator_calls

    def select(self, environment: StsEnv):
        if environment.observation.phase is Phase.COMBAT:
            return self.search.select(environment)
        return self.heuristic(environment.observation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate M6 full-run policies.")
    parser.add_argument(
        "--method",
        choices=(
            "random",
            "heuristic",
            "heuristic-search",
            "learned",
            "learned-heuristic",
            "learned-search",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=17)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze_manifest(path: Path, checkpoint: Path | None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported M6 freeze manifest schema")
    if payload.get("final_seed_range") != [M6_FINAL_SEED_START, M6_FINAL_SEED_END]:
        raise ValueError("freeze manifest does not cover the formal final seed range")
    if set(payload.get("checkpoints", {})) != {"17", "29", "43"}:
        raise ValueError("freeze manifest must contain all three formal run seeds")
    source_sha256 = str(payload.get("source_sha256") or "")
    if not source_sha256:
        raise ValueError("freeze manifest lacks a source hash")
    for run_seed, entry in payload["checkpoints"].items():
        if int(entry.get("run_seed", -1)) != int(run_seed):
            raise ValueError("freeze manifest checkpoint run seed is inconsistent")
        if entry.get("stage") != "full_run":
            raise ValueError("freeze manifest checkpoint is not at full_run")
        if entry.get("source_sha256") != source_sha256:
            raise ValueError("freeze manifest checkpoint source hashes differ")
        if entry.get("parameter_source") != "ema" or not entry.get("evaluation_only"):
            raise ValueError("freeze manifest must contain EMA evaluation checkpoints")
        frozen_path = Path(str(entry.get("path") or ""))
        if not frozen_path.is_file() or sha256_file(frozen_path) != entry.get("sha256"):
            raise ValueError("freeze manifest contains a missing or modified checkpoint")
        frozen = load_m6_checkpoint(frozen_path, device="cpu")
        if frozen.config.run_seed != int(run_seed):
            raise ValueError("freeze manifest run seed differs from checkpoint contents")
        if frozen.scheduler.current.name != "full_run":
            raise ValueError("frozen checkpoint contents are not at full_run")
        if frozen.manifest.get("source_sha256") != source_sha256:
            raise ValueError("frozen checkpoint source hash differs from checkpoint contents")
        if (
            not frozen.manifest.get("evaluation_only")
            or frozen.manifest.get("parameter_source") != "ema"
        ):
            raise ValueError("frozen checkpoint contents are not EMA evaluation parameters")
    verification: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_sha256": source_sha256,
        "checkpoint_run_seed": None,
    }
    if checkpoint is None:
        return verification
    resolved = str(checkpoint.resolve())
    for run_seed, entry in payload["checkpoints"].items():
        if entry["path"] != resolved:
            continue
        if sha256_file(checkpoint) != entry["sha256"]:
            raise ValueError("checkpoint hash differs from the frozen checkpoint")
        verification["checkpoint_run_seed"] = int(run_seed)
        return verification
    raise ValueError("checkpoint is not one of the frozen formal checkpoints")


def main() -> int:
    args = parse_args()
    validate_m6_evaluation_seed_range(
        args.seed_start,
        args.seed_count,
        final=args.final,
    )
    freeze_verification = None
    if args.final:
        if args.freeze_manifest is None:
            raise ValueError("formal final evaluation requires --freeze-manifest")
        freeze_verification = verify_freeze_manifest(
            args.freeze_manifest,
            args.checkpoint,
        )
    if args.method.startswith("learned") and args.checkpoint is None:
        raise ValueError("learned methods require --checkpoint")
    loaded = (
        load_m6_checkpoint(args.checkpoint, device="cpu")
        if args.checkpoint is not None
        else None
    )
    trainer = loaded.trainer if loaded is not None else None
    runtime_manifest = build_runtime_manifest(PROJECT_ROOT)
    if (
        freeze_verification is not None
        and runtime_manifest["source_sha256"] != freeze_verification["source_sha256"]
    ):
        raise ValueError("final evaluation source tree differs from the frozen source hash")
    if freeze_verification is not None and loaded is not None:
        if freeze_verification["checkpoint_run_seed"] != loaded.config.run_seed:
            raise ValueError("frozen checkpoint run seed differs from checkpoint contents")
        if loaded.manifest.get("source_sha256") != freeze_verification["source_sha256"]:
            raise ValueError("frozen checkpoint source hash differs from checkpoint contents")

    def policy_factory(policy_seed: int, _: int) -> Any:
        if args.method == "random":
            return RandomPolicy(policy_seed)
        if args.method == "heuristic":
            return HeuristicPolicy()
        search = None
        if args.method.endswith("search"):
            search = ParticleSearchPolicy(
                BeliefSearchConfig(
                    simulations=max(args.search_budget, 64),
                    simulator_call_budget=args.search_budget,
                    max_depth=32,
                    rollout_depth=8,
                ),
                seed=policy_seed,
                rollout_policy=HeuristicPolicy(),
            )
        if args.method == "heuristic-search":
            assert search is not None
            return HeuristicSearchPolicy(search)
        assert trainer is not None
        heuristic = HeuristicPolicy()
        combat_selector = None
        if args.method == "learned-heuristic":
            combat_selector = lambda environment: heuristic(environment.observation)
        elif search is not None:
            combat_selector = search.select
        return HierarchicalRecurrentPolicy(
            trainer,
            combat_selector=combat_selector,
            deterministic=not args.stochastic,
        )

    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    summary = evaluate_full_runs(
        lambda: StsEnv(LightspeedBackend()),
        policy_factory,
        seeds,
        policy_seed=args.policy_seed,
        max_steps=args.max_steps,
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        "scope": "Ironclad A0 full run",
        "method": args.method,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "run_seed": loaded.config.run_seed if loaded is not None else None,
        "seed_range": [seeds[0], seeds[-1]],
        "policy_seed": args.policy_seed,
        "search_budget": args.search_budget if args.method.endswith("search") else 0,
        "combat_policy": {
            "random": "integrated",
            "heuristic": "integrated",
            "heuristic-search": "belief-search",
            "learned": "network",
            "learned-heuristic": "heuristic",
            "learned-search": "belief-search",
        }[args.method],
        "stochastic": args.stochastic,
        "final": args.final,
        "freeze_manifest": freeze_verification,
        "runtime_manifest": runtime_manifest,
        "summary": summary.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.to_dict().items() if key != "episodes"},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
