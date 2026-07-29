from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from sts_env import LightspeedBackend, Phase, StsEnv
from sts_env.search import BeliefSearchConfig, ParticleSearchPolicy
from sts_env.training import (
    HeuristicPolicy,
    HierarchicalRecurrentPolicy,
    M6_FINAL_SEED_END,
    M6_FINAL_SEED_START,
    M7_FINAL_SEED_END,
    M7_FINAL_SEED_START,
    RandomPolicy,
    evaluate_full_runs,
    load_m6_checkpoint,
    load_m7_checkpoint,
    load_m7b_checkpoint,
    load_m7c_checkpoint,
    m7c_seed_registry,
    require_registered_seed_range,
    validate_m7_fixed_budget_progress,
    validate_m7_evaluation_seed_range,
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
    parser = argparse.ArgumentParser(description="Evaluate M7 full-run policies.")
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
    parser.add_argument(
        "--report-label",
        help="method label used in reports while preserving the selected policy behavior",
    )
    parser.add_argument(
        "--m7c-range-name",
        help="require an exact named M7-C pre-registered evaluation range",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument(
        "--m6-posthoc-diagnostic",
        action="store_true",
        help="acknowledge reuse of M6's already revealed final seed range",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_method_name(method: str, label: str | None) -> str:
    if label is None:
        return method
    resolved = label.strip()
    if not resolved:
        raise ValueError("M7 report label cannot be empty")
    return resolved


def load_checkpoint(path: Path) -> tuple[Any, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") == "m7b":
        return load_m7b_checkpoint(path, device="cpu"), "m7b"
    if payload.get("protocol") == "m7c-dagger":
        return load_m7c_checkpoint(path, device="cpu"), "m7c-dagger"
    if payload.get("protocol") == "m7":
        return load_m7_checkpoint(path, device="cpu"), "m7"
    return load_m6_checkpoint(path, device="cpu"), "m6"


def verify_freeze_manifest(path: Path, checkpoint: Path | None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != "m7"
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError("unsupported M7 freeze manifest")
    if payload.get("final_seed_range") != [M7_FINAL_SEED_START, M7_FINAL_SEED_END]:
        raise ValueError("M7 freeze manifest has the wrong final seed range")
    checkpoints = dict(payload.get("checkpoints") or {})
    if not checkpoints:
        raise ValueError("M7 freeze manifest contains no checkpoints")
    source_sha256 = str(payload.get("source_sha256") or "")
    if not source_sha256:
        raise ValueError("M7 freeze manifest lacks a source hash")
    resolved_checkpoint = None if checkpoint is None else str(checkpoint.resolve())
    matched_run_seed = None
    for run_seed, entry in checkpoints.items():
        frozen_path = Path(str(entry.get("path") or ""))
        if not frozen_path.is_file() or sha256_file(frozen_path) != entry.get("sha256"):
            raise ValueError("M7 freeze manifest contains a missing or modified checkpoint")
        loaded = load_m7_checkpoint(frozen_path, device="cpu")
        if loaded.config.run_seed != int(run_seed):
            raise ValueError("M7 frozen checkpoint run seed differs")
        if loaded.scheduler.current.name != "full_run":
            raise ValueError("M7 frozen checkpoint is not at full_run")
        if loaded.manifest.get("source_sha256") != source_sha256:
            raise ValueError("M7 frozen checkpoint source hash differs")
        if (
            not loaded.manifest.get("evaluation_only")
            or loaded.manifest.get("parameter_source") != "ema"
        ):
            raise ValueError("M7 freeze requires EMA evaluation-only checkpoints")
        completion_entry = dict(entry.get("completion_checkpoint") or {})
        completion_path = Path(str(completion_entry.get("path") or ""))
        if (
            not completion_path.is_file()
            or sha256_file(completion_path) != completion_entry.get("sha256")
        ):
            raise ValueError("M7 freeze has a missing or modified completion checkpoint")
        completed = load_m7_checkpoint(completion_path, device="cpu")
        if completed.config.run_seed != int(run_seed):
            raise ValueError("M7 completion checkpoint run seed differs")
        if completed.manifest.get("evaluation_only"):
            raise ValueError("M7 completion checkpoint is not resumable")
        if completed.scheduler.current.name != "full_run":
            raise ValueError("M7 completion checkpoint is not at full_run")
        if completed.manifest.get("source_sha256") != source_sha256:
            raise ValueError("M7 completion checkpoint source hash differs")
        validate_m7_fixed_budget_progress(
            completed.config,
            completed.progress,
            completed.update_index,
        )
        if str(frozen_path.resolve()) == resolved_checkpoint:
            matched_run_seed = int(run_seed)
    if checkpoint is not None and matched_run_seed is None:
        raise ValueError("checkpoint is not present in the M7 freeze manifest")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_sha256": source_sha256,
        "checkpoint_run_seed": matched_run_seed,
    }


def main() -> int:
    args = parse_args()
    validate_m7_evaluation_seed_range(
        args.seed_start,
        args.seed_count,
        final=args.final,
    )
    requested_end = args.seed_start + args.seed_count - 1
    m7c_seed_range = None
    if args.m7c_range_name is not None:
        registry = m7c_seed_registry()
        if args.m7c_range_name not in registry:
            raise ValueError(f"unknown M7-C range: {args.m7c_range_name}")
        registered = registry[args.m7c_range_name]
        m7c_seed_range = require_registered_seed_range(
            registered.name,
            start=args.seed_start,
            count=args.seed_count,
            registry=registry,
        )
    intersects_m6_final = max(args.seed_start, M6_FINAL_SEED_START) <= min(
        requested_end,
        M6_FINAL_SEED_END,
    )
    if intersects_m6_final and not args.m6_posthoc_diagnostic:
        raise ValueError(
            "M6 final seeds are revealed diagnostic data; pass --m6-posthoc-diagnostic"
        )
    if args.method.startswith("learned") and args.checkpoint is None:
        raise ValueError("learned methods require --checkpoint")
    loaded = None
    checkpoint_protocol = None
    if args.checkpoint is not None:
        loaded, checkpoint_protocol = load_checkpoint(args.checkpoint)
    if checkpoint_protocol == "m7c-dagger" and m7c_seed_range is None:
        raise ValueError("M7-C evaluation requires --m7c-range-name")
    if args.m6_posthoc_diagnostic and checkpoint_protocol != "m6":
        raise ValueError("M6 post-hoc diagnostics require an M6 checkpoint")

    freeze_verification = None
    if args.final:
        if args.freeze_manifest is None:
            raise ValueError("formal M7 evaluation requires --freeze-manifest")
        if checkpoint_protocol not in {None, "m7"}:
            raise ValueError("formal M7 evaluation cannot use an M6 checkpoint")
        freeze_verification = verify_freeze_manifest(
            args.freeze_manifest,
            args.checkpoint,
        )
    trainer = loaded.trainer if loaded is not None else None
    runtime_manifest = build_runtime_manifest(PROJECT_ROOT)
    if (
        freeze_verification is not None
        and runtime_manifest["source_sha256"] != freeze_verification["source_sha256"]
    ):
        raise ValueError("M7 evaluation source differs from the source freeze")

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
        "protocol": "m7",
        "purpose": "formal-final" if args.final else "diagnostic",
        "scope": "Ironclad A0 full run",
        "method": report_method_name(args.method, args.report_label),
        "policy_method": args.method,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "checkpoint_protocol": checkpoint_protocol,
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
        "m6_posthoc_diagnostic": args.m6_posthoc_diagnostic,
        "m7c_range_name": (
            None if m7c_seed_range is None else m7c_seed_range.name
        ),
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
