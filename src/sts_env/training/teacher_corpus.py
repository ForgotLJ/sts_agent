from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sts_env.trace import EpisodeTrace


def teacher_corpus_digest(corpus: str | Path) -> dict[str, Any]:
    root = Path(corpus).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    stage_trace_counts: dict[str, int] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
        file_count += 1
        total_bytes += path.stat().st_size
        if path.suffix.lower() == ".jsonl":
            stage = path.relative_to(root).parts[0]
            stage_trace_counts[stage] = stage_trace_counts.get(stage, 0) + 1
    return {
        "corpus_path": str(root),
        "aggregate_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "stage_trace_counts": dict(sorted(stage_trace_counts.items())),
    }


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


def build_teacher_corpus_manifest(
    corpus: str | Path,
    expected_stage_counts: dict[str, int],
) -> dict[str, Any]:
    payload = teacher_corpus_digest(corpus)
    root = Path(payload["corpus_path"])
    errors: list[str] = []
    observed = dict(payload["stage_trace_counts"])
    expected = dict(sorted(expected_stage_counts.items()))
    if observed != expected:
        errors.append(
            f"teacher corpus stage counts differ: expected={expected} observed={observed}"
        )
    validated = 0
    minimum_seed: int | None = None
    maximum_seed: int | None = None
    for path in sorted(root.rglob("*.jsonl")):
        try:
            trace = EpisodeTrace.read_jsonl(path)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{path.relative_to(root).as_posix()}: {error}")
            continue
        if trace.seed < 0 or trace.seed > 999_999:
            errors.append(
                f"{path.relative_to(root).as_posix()}: seed {trace.seed} is outside training range"
            )
        if not trace_uses_semantic_neow(trace):
            errors.append(
                f"{path.relative_to(root).as_posix()}: trace lacks semantic Neow action"
            )
        minimum_seed = trace.seed if minimum_seed is None else min(minimum_seed, trace.seed)
        maximum_seed = trace.seed if maximum_seed is None else max(maximum_seed, trace.seed)
        validated += 1
    return {
        **payload,
        "complete": not errors,
        "errors": len(errors),
        "error_messages": errors,
        "validated_trace_count": validated,
        "seed_min": minimum_seed,
        "seed_max": maximum_seed,
    }


def verify_teacher_corpus_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("complete") is not True or int(payload.get("errors", -1)) != 0:
        raise ValueError("teacher corpus manifest is incomplete")
    current = teacher_corpus_digest(str(payload.get("corpus_path") or ""))
    for key in (
        "aggregate_sha256",
        "file_count",
        "total_bytes",
        "stage_trace_counts",
    ):
        if current[key] != payload.get(key):
            raise ValueError(f"teacher corpus differs from frozen manifest: {key}")
    return current
