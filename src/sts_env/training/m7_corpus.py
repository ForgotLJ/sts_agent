from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sts_env import EpisodeTrace
from sts_env.training.curriculum import PrefixCorpus, materialize_recovery_trace
from sts_env.training.parallel import LightspeedEnvironmentFactory
from sts_env.training.self_imitation import (
    imitation_trace_progress,
    select_weighted_frontier_traces,
)


def build_m7_prefix_corpus(
    run_directory: Path,
    target_act: int,
    candidate_paths: list[Path],
    recovery_environment_factory: Callable[[], Any] | None = None,
) -> PrefixCorpus:
    traces = tuple(EpisodeTrace.read_jsonl(path) for path in candidate_paths)
    traces = tuple(trace for trace in traces if trace_uses_semantic_neow(trace))
    if not traces:
        raise ValueError("no M7 candidate trace uses the public action schema")
    progress: list[tuple[EpisodeTrace, int, bool]] = []
    for trace in traces:
        has_recovery_prefix = bool(
            (trace.metadata or {}).get("curriculum_source_trace")
        )
        if has_recovery_prefix:
            if recovery_environment_factory is None:
                raise ValueError(
                    "M7 prefix extraction requires a recovery environment factory"
                )
            environment = recovery_environment_factory()
            observation = environment.replay_recovery_trace(trace.prefix(0))
        else:
            environment = LightspeedEnvironmentFactory()()
            observation, _ = environment.reset(seed=trace.seed)
        prefix = trace.prefix(0) if observation.act >= target_act else None
        for step_index, step in enumerate(trace.steps):
            if step.action not in observation.legal_actions:
                raise ValueError("M7 prefix source trace contains a stale action")
            observation, _, _, _, _ = environment.step(step.action)
            if prefix is None and observation.act >= target_act:
                prefix = trace.prefix(step_index + 1)
        if prefix is not None:
            progress.append(
                (
                    prefix,
                    observation.act,
                    bool(trace.steps and trace.steps[-1].reward > 0),
                )
            )
    if not progress:
        raise ValueError(f"no supplied trace reaches Act {target_act}")
    if target_act < 3:
        farther = tuple(prefix for prefix, final_act, _ in progress if final_act > target_act)
        prefixes = farther or tuple(prefix for prefix, _, _ in progress)
    else:
        wins = tuple(prefix for prefix, _, won in progress if won)
        prefixes = wins or tuple(prefix for prefix, _, _ in progress)
    corpus = PrefixCorpus(prefixes, target_act)
    corpus.write(run_directory / "curriculum" / f"act-{target_act}")
    return corpus


def m7_teacher_trace_paths(
    run_directory: Path,
    stage_name: str,
    shared_corpus: Path | None = None,
) -> list[Path]:
    stage_candidates = [stage_name]
    if stage_name in {"act3_clear", "full_run"}:
        stage_candidates = ["act2_clear", "act3_clear", "full_run"]
    paths: list[Path] = []
    if shared_corpus is not None:
        for candidate_stage in stage_candidates:
            paths.extend(sorted((shared_corpus / candidate_stage).glob("*.jsonl")))
        if paths:
            return sorted(set(paths))
    for candidate_stage in stage_candidates:
        for source in ("teacher-v5", "teacher-v4", "teacher-v3", "teacher-v2", "teacher"):
            paths.extend(
                sorted(
                    (run_directory / "curriculum" / source / candidate_stage).glob(
                        "*.jsonl"
                    )
                )
            )
    return sorted(set(paths))


def stage_promotion_threshold(
    curriculum_config: dict[str, Any],
    stage_name: str,
) -> float:
    thresholds = dict(curriculum_config.get("stage_promotion_thresholds", {}))
    return float(thresholds.get(stage_name, curriculum_config["promotion_threshold"]))


def load_m7_imitation_traces(
    candidate_paths: list[Path],
    teacher_paths: list[Path],
    maximum_traces: int,
    maximum_candidate_traces: int,
    frontier_trace_repeats: int,
) -> tuple[EpisodeTrace, ...]:
    teacher_limit = min(len(teacher_paths), maximum_traces * 3 // 4)
    candidate_limit = min(
        maximum_candidate_traces,
        maximum_traces - teacher_limit,
    )
    selected_teachers: list[Path] = []
    if teacher_limit:
        ordered_teachers = sorted(teacher_paths)
        selected_teachers = [
            ordered_teachers[index * len(ordered_teachers) // teacher_limit]
            for index in range(teacher_limit)
        ]
    loaded_candidates = tuple(EpisodeTrace.read_jsonl(path) for path in candidate_paths)
    loaded_candidates = tuple(
        trace
        for trace in loaded_candidates
        if trace_uses_semantic_neow(materialize_recovery_trace(trace))
    )
    selected_candidates = select_weighted_frontier_traces(
        loaded_candidates,
        candidate_limit,
        frontier_trace_repeats,
    )
    selected_teacher_traces = tuple(
        trace
        for path in selected_teachers
        if trace_uses_semantic_neow(trace := EpisodeTrace.read_jsonl(path))
    )
    return (*selected_candidates, *selected_teacher_traces)


def trace_uses_semantic_neow(trace: EpisodeTrace) -> bool:
    source_payload = dict((trace.metadata or {}).get("curriculum_source_trace") or {})
    while source_payload:
        trace = EpisodeTrace.from_dict(source_payload)
        source_payload = dict(
            (trace.metadata or {}).get("curriculum_source_trace") or {}
        )
    if not trace.steps:
        return False
    source_id = trace.steps[0].action.source_id
    return isinstance(source_id, str) and source_id.startswith("neow")


def prune_m7_candidate_paths(
    candidate_paths: list[Path],
    capacity: int,
) -> list[Path]:
    traces = {path: EpisodeTrace.read_jsonl(path) for path in candidate_paths}
    ordered = sorted(
        candidate_paths,
        key=lambda path: (
            -imitation_trace_progress(traces[path])[0],
            -imitation_trace_progress(traces[path])[1],
            -imitation_trace_progress(traces[path])[2],
            path.name,
        ),
    )
    for path in ordered[capacity:]:
        path.unlink()
    return sorted(ordered[:capacity])
