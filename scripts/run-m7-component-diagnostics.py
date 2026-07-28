from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import (
    hierarchical_paired_evaluation_difference,
    summarize_m7_evaluations,
)


def checkpoint_argument(value: str) -> tuple[int, Path]:
    run_seed, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("checkpoint must use RUN_SEED=PATH")
    try:
        return int(run_seed), Path(path)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint run seed must be an integer") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M7 component diagnostics on M6 checkpoints.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=checkpoint_argument,
        required=True,
        help="M6 checkpoint as RUN_SEED=PATH; repeat for every run",
    )
    parser.add_argument(
        "--existing-evaluation-root",
        type=Path,
        required=True,
        help="directory containing run-*/{heuristic,heuristic-search,learned-search}.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=2_000_000)
    parser.add_argument("--seed-count", type=int, default=1_024)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def cached_evaluation_is_valid(
    path: Path,
    *,
    run_seed: int,
    seed_start: int,
    seed_count: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_range = [seed_start, seed_start + seed_count - 1]
    return (
        existing.get("method") == "learned-heuristic"
        and int(existing.get("run_seed", -1)) == run_seed
        and existing.get("seed_range") == expected_range
        and int(dict(existing.get("summary") or {}).get("errors", -1)) == 0
    )


def main() -> int:
    args = parse_args()
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise ValueError("duplicate diagnostic checkpoint run seed")
    if args.parallel <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("diagnostic concurrency and bootstrap samples must be positive")
    for run_seed, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint for run {run_seed}: {checkpoint}")
    args.output.mkdir(parents=True, exist_ok=True)

    def run_evaluation(run_seed: int, checkpoint: Path) -> Path:
        output = args.output / f"run-{run_seed}" / "learned-heuristic.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        if cached_evaluation_is_valid(
            output,
            run_seed=run_seed,
            seed_start=args.seed_start,
            seed_count=args.seed_count,
        ):
            return output
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate-m7.py"),
            "--method",
            "learned-heuristic",
            "--checkpoint",
            str(checkpoint),
            "--seed-start",
            str(args.seed_start),
            "--seed-count",
            str(args.seed_count),
            "--policy-seed",
            str(run_seed),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--m6-posthoc-diagnostic",
            "--output",
            str(output),
        ]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        (output.parent / "learned-heuristic.stdout.log").write_text(
            result.stdout,
            encoding="utf-8",
        )
        (output.parent / "learned-heuristic.stderr.log").write_text(
            result.stderr,
            encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(
                f"learned-heuristic evaluation failed for run {run_seed}: {result.stderr}"
            )
        return output

    generated: list[Path] = []
    with ThreadPoolExecutor(max_workers=min(args.parallel, len(checkpoints))) as executor:
        futures = {
            executor.submit(run_evaluation, run_seed, checkpoint): run_seed
            for run_seed, checkpoint in sorted(checkpoints.items())
        }
        for future in as_completed(futures):
            generated.append(future.result())

    evaluations: list[dict[str, Any]] = []
    for run_seed in sorted(checkpoints):
        for method in (
            "heuristic",
            "heuristic-search",
            "learned-search",
        ):
            path = args.existing_evaluation_root / f"run-{run_seed}" / f"{method}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing existing M6 evaluation: {path}")
            evaluations.append(json.loads(path.read_text(encoding="utf-8")))
        generated_path = args.output / f"run-{run_seed}" / "learned-heuristic.json"
        evaluations.append(json.loads(generated_path.read_text(encoding="utf-8")))

    summary = summarize_m7_evaluations(
        tuple(evaluations),
        reference_method="heuristic",
        bootstrap_samples=args.bootstrap_samples,
    )
    by_method: dict[str, tuple[dict[str, Any], ...]] = {}
    for evaluation in evaluations:
        by_method.setdefault(str(evaluation["method"]), ())
        by_method[str(evaluation["method"])] += (evaluation,)
    component_pairs = {
        "search_effect_on_heuristic": ("heuristic-search", "heuristic"),
        "noncombat_effect_with_heuristic_combat": ("learned-heuristic", "heuristic"),
        "search_effect_on_learned": ("learned-search", "learned-heuristic"),
        "noncombat_effect_with_search_combat": ("learned-search", "heuristic-search"),
    }
    components = {}
    for index, (name, (candidate, reference)) in enumerate(component_pairs.items()):
        components[name] = hierarchical_paired_evaluation_difference(
            by_method[candidate],
            by_method[reference],
            bootstrap_samples=args.bootstrap_samples,
            seed=2_607_280 + index,
        )
    summary_path = args.output / "summary.json"
    components_path = args.output / "components.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    components_path.write_text(
        json.dumps(components, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "generated_evaluations": len(generated),
                "summary": str(summary_path),
                "components": str(components_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
