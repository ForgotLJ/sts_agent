from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SEEDS = (17, 29, 43)
METHODS = ("random", "heuristic", "heuristic-search", "learned", "learned-search")
RUN_METRICS = (
    "win_rate",
    "act1_clear_rate",
    "act2_clear_rate",
    "act3_clear_rate",
    "mean_floor",
    "median_floor",
    "mean_final_hp",
    "mean_proxy_score",
    "mean_decisions",
    "total_simulator_calls",
    "total_wall_seconds",
    "errors",
    "crashes",
    "illegal_actions",
    "recovery_failures",
    "truncations",
    "timeouts",
    "cycles",
)
PAIRED_METRICS = (
    "win",
    "final_floor",
    "final_hp",
    "proxy_score",
    "decisions",
    "simulator_calls",
    "wall_seconds",
)
FINAL_SEED_START = 2_000_000
FINAL_SEED_END = 2_001_023
FINAL_SEED_COUNT = 1_024
TARGET_UPDATES = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formal M6 completion evidence.")
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m6r_server_pipeline",
    )
    parser.add_argument(
        "--gate-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m6r_server_gates",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m6r_server_training",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m6r_server_evaluations",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def last_jsonl(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            last = None
            for line in stream:
                if line.strip():
                    last = line
        return dict(json.loads(last)) if last is not None else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_integer_set(values: Any, expected: tuple[int, ...]) -> bool:
    try:
        return {int(value) for value in values} == set(expected)
    except (TypeError, ValueError):
        return False


def audit(
    pipeline_root: Path,
    gate_root: Path,
    training_root: Path,
    evaluation_root: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    source_freeze = read_json(gate_root / "source-freeze.json")
    source_manifest = dict((source_freeze or {}).get("runtime_manifest") or {})
    source_hash = str(source_manifest.get("source_sha256") or "")
    source_valid = (
        source_freeze is not None
        and int(source_freeze.get("schema_version", -1)) == 1
        and source_freeze.get("status") == "frozen"
        and exact_integer_set(source_freeze.get("run_seeds"), RUN_SEEDS)
        and source_freeze.get("final_seed_range")
        == [FINAL_SEED_START, FINAL_SEED_END]
        and len(source_hash) == 64
        and all(character in "0123456789abcdef" for character in source_hash)
    )
    add("source_freeze", source_valid, source_hash if source_valid else "missing or invalid")

    gate_checks: dict[str, Callable[[dict[str, Any]], bool]] = {
        "python-tests": lambda payload: bool(payload.get("complete"))
        and int(payload.get("errors", -1)) == 0
        and int(payload.get("tests", 0)) >= 100,
        "prefix-recovery": lambda payload: bool(payload.get("complete"))
        and int(payload.get("errors", -1)) == 0
        and int(payload.get("checks", 0)) >= 1_000,
        "stress-10000": lambda payload: bool(payload.get("complete"))
        and int(payload.get("errors", -1)) == 0
        and int(payload.get("episodes", 0)) >= 10_000,
        "communication-differential": lambda payload: bool(payload.get("complete"))
        and int(payload.get("errors", -1)) == 0
        and int(payload.get("records", 0)) > 0
        and int(payload.get("differences", -1)) == 0,
        "teacher-corpus": lambda payload: bool(payload.get("complete"))
        and int(payload.get("errors", -1)) == 0
        and int(payload.get("validated_trace_count", 0)) == 1_038,
    }
    frozen_gates = dict((source_freeze or {}).get("gates") or {})
    for name, validator in gate_checks.items():
        payload = dict(dict(frozen_gates.get(name) or {}).get("payload") or {})
        passed = bool(payload) and validator(payload)
        add(f"gate_{name}", passed, "complete" if passed else "missing or invalid")

    for run_seed in RUN_SEEDS:
        run_directory = training_root / f"seed-{run_seed}"
        metric = last_jsonl(run_directory / "metrics.jsonl")
        passed = (
            metric is not None
            and int(metric.get("update", -1)) == TARGET_UPDATES
            and metric.get("stage") == "full_run"
            and (run_directory / "best-evaluation-checkpoint.pt").is_file()
        )
        detail = (
            f"update={metric.get('update')}, stage={metric.get('stage')}"
            if metric is not None
            else "training incomplete"
        )
        add(f"training_seed_{run_seed}", passed, detail)

    checkpoint_freeze = read_json(gate_root / "checkpoint-freeze.json")
    checkpoint_entries = dict((checkpoint_freeze or {}).get("checkpoints") or {})
    checkpoint_valid = (
        checkpoint_freeze is not None
        and int(checkpoint_freeze.get("schema_version", -1)) == 1
        and checkpoint_freeze.get("final_seed_range")
        == [FINAL_SEED_START, FINAL_SEED_END]
        and checkpoint_freeze.get("source_sha256") == source_hash
        and exact_integer_set(checkpoint_entries.keys(), RUN_SEEDS)
    )
    if checkpoint_valid:
        for run_seed in RUN_SEEDS:
            entry = dict(checkpoint_entries[str(run_seed)])
            path = Path(str(entry.get("path") or ""))
            checkpoint_valid = checkpoint_valid and (
                int(entry.get("run_seed", -1)) == run_seed
                and entry.get("stage") == "full_run"
                and bool(entry.get("evaluation_only"))
                and entry.get("parameter_source") == "ema"
                and path.is_file()
                and sha256_file(path) == entry.get("sha256")
            )
    add(
        "checkpoint_freeze",
        checkpoint_valid,
        "three EMA checkpoints verified" if checkpoint_valid else "missing or invalid",
    )

    learned_wins = 0
    evaluation_paths: list[Path] = []
    for run_seed in RUN_SEEDS:
        for method in METHODS:
            path = evaluation_root / f"run-{run_seed}" / f"{method}.json"
            evaluation_paths.append(path)
            evaluation = read_json(path)
            summary = dict((evaluation or {}).get("summary") or {})
            episodes = list(summary.get("episodes") or [])
            episode_seeds = [int(episode.get("seed", -1)) for episode in episodes]
            wins = sum(bool(episode.get("won")) for episode in episodes)
            errors = sum(bool(str(episode.get("error") or "")) for episode in episodes)
            expected_run_seed = run_seed if method.startswith("learned") else None
            actual_run_seed = (evaluation or {}).get("run_seed")
            search_budget = 64 if method.endswith("search") else 0
            freeze_manifest = dict((evaluation or {}).get("freeze_manifest") or {})
            runtime_manifest = dict((evaluation or {}).get("runtime_manifest") or {})
            valid = (
                evaluation is not None
                and bool(evaluation.get("final"))
                and evaluation.get("method") == method
                and int(evaluation.get("policy_seed", -1)) == run_seed
                and actual_run_seed == expected_run_seed
                and evaluation.get("seed_range") == [FINAL_SEED_START, FINAL_SEED_END]
                and int(evaluation.get("search_budget", -1)) == search_budget
                and int(summary.get("errors", -1)) == 0
                and len(episodes) == FINAL_SEED_COUNT
                and set(episode_seeds) == set(range(FINAL_SEED_START, FINAL_SEED_END + 1))
                and len(episode_seeds) == len(set(episode_seeds))
                and math.isclose(
                    float(summary.get("win_rate", -1.0)),
                    wins / FINAL_SEED_COUNT,
                    abs_tol=1e-12,
                )
                and int(summary.get("errors", -1)) == errors
                and len(list(summary.get("win_rate_ci95") or [])) == 2
                and len(list(summary.get("mean_floor_ci95") or [])) == 2
                and checkpoint_freeze is not None
                and freeze_manifest.get("source_sha256")
                == checkpoint_freeze.get("source_sha256")
                and runtime_manifest.get("source_sha256")
                == checkpoint_freeze.get("source_sha256")
            )
            if method.startswith("learned"):
                learned_wins += wins
            add(
                f"evaluation_{run_seed}_{method}",
                valid,
                f"episodes={len(episodes)}, wins={wins}" if valid else "missing or invalid",
            )

    existing_evaluations = sum(path.is_file() for path in evaluation_paths)
    add("evaluation_file_count", existing_evaluations == 15, f"{existing_evaluations}/15")
    add("learned_a0_win", learned_wins >= 1, f"wins={learned_wins}")

    aggregate_summary = read_json(evaluation_root / "summary.json")
    runs = dict((aggregate_summary or {}).get("runs") or {})
    aggregate = dict((aggregate_summary or {}).get("aggregate") or {})
    summary_valid = (
        aggregate_summary is not None
        and int(aggregate_summary.get("schema_version", -1)) == 1
        and aggregate_summary.get("reference_method") == "heuristic"
        and int(aggregate_summary.get("bootstrap_samples", -1)) == 10_000
        and exact_integer_set(runs.keys(), RUN_SEEDS)
        and len(list(aggregate_summary.get("evaluation_files") or [])) == 15
    )
    if summary_valid:
        for method in METHODS:
            method_summary = dict(aggregate.get(method) or {})
            metrics = dict(method_summary.get("metrics") or {})
            summary_valid = summary_valid and int(method_summary.get("run_count", -1)) == 3
            for metric_name in RUN_METRICS:
                metric = dict(metrics.get(metric_name) or {})
                summary_valid = summary_valid and (
                    len(list(metric.get("values") or [])) == 3
                    and metric.get("mean") is not None
                    and metric.get("standard_deviation") is not None
                )
        for run_seed in RUN_SEEDS:
            run = dict(runs.get(str(run_seed)) or {})
            methods = dict(run.get("methods") or {})
            comparisons = dict(run.get("paired_comparisons") or {})
            summary_valid = summary_valid and len(methods) == 5
            for method in METHODS:
                if method == "heuristic":
                    continue
                comparison = dict(comparisons.get(f"{method}_minus_heuristic") or {})
                summary_valid = summary_valid and (
                    int(comparison.get("sample_count", -1)) == FINAL_SEED_COUNT
                    and comparison.get("seed_range") == [FINAL_SEED_START, FINAL_SEED_END]
                )
                metrics = dict(comparison.get("metrics") or {})
                for metric_name in PAIRED_METRICS:
                    metric = dict(metrics.get(metric_name) or {})
                    summary_valid = summary_valid and (
                        metric.get("mean_difference") is not None
                        and len(list(metric.get("bootstrap_ci95") or [])) == 2
                    )
    add(
        "aggregate_summary",
        summary_valid,
        "15 evaluations with paired statistics" if summary_valid else "missing or invalid",
    )

    status = read_json(pipeline_root / "status.json")
    status_data = dict((status or {}).get("data") or {})
    status_valid = (
        status is not None
        and status.get("stage") == "pipeline_complete"
        and status_data.get("state") == "complete"
        and int(status_data.get("final_evaluations", -1)) == 15
        and int(status_data.get("learned_wins", -1)) == learned_wins
    )
    add(
        "pipeline_status",
        status_valid,
        f"stage={(status or {}).get('stage', 'missing')}, learned_wins={status_data.get('learned_wins', -1)}",
    )

    failed = [check for check in checks if not check["passed"]]
    return {
        "schema_version": 1,
        "status": "complete" if not failed else "incomplete",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "learned_wins": learned_wins,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": [check["name"] for check in failed],
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    payload = audit(
        args.pipeline_root.resolve(),
        args.gate_root.resolve(),
        args.training_root.resolve(),
        args.evaluation_root.resolve(),
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else args.pipeline_root.resolve() / "completion-audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "complete" or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
