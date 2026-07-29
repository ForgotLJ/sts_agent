from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from sts_env.training.self_imitation import ImitationChunk


def save_m7b_replay_batch(
    path: str | Path,
    *,
    chunks: tuple[ImitationChunk, ...],
    corpus_sha256: str,
    trace_seeds: tuple[int, ...],
    encoder_config: dict[str, int],
    chunk_length: int,
    burn_in_steps: int,
) -> None:
    if (
        not chunks
        or not corpus_sha256
        or not trace_seeds
        or chunk_length <= 0
        or burn_in_steps < 0
    ):
        raise ValueError("M7-B replay batch arguments are invalid")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(
        {
            "protocol": "m7b-replay-batch",
            "schema_version": 1,
            "corpus_sha256": corpus_sha256,
            "trace_seeds": trace_seeds,
            "encoder_config": dict(encoder_config),
            "chunk_length": chunk_length,
            "burn_in_steps": burn_in_steps,
            "chunks": [_chunk_to_payload(chunk) for chunk in chunks],
        },
        temporary,
    )
    temporary.replace(destination)


def load_m7b_replay_batch(
    path: str | Path,
    *,
    expected_corpus_sha256: str | None = None,
    expected_encoder_config: dict[str, int] | None = None,
    expected_chunk_length: int | None = None,
    expected_burn_in_steps: int | None = None,
) -> tuple[ImitationChunk, ...]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        payload.get("protocol") != "m7b-replay-batch"
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise ValueError("unsupported M7-B replay batch schema")
    checks = (
        (
            expected_corpus_sha256,
            str(payload.get("corpus_sha256") or ""),
            "corpus hash",
        ),
        (
            expected_encoder_config,
            dict(payload.get("encoder_config") or {}),
            "encoder configuration",
        ),
        (
            expected_chunk_length,
            int(payload.get("chunk_length", -1)),
            "chunk length",
        ),
        (
            expected_burn_in_steps,
            int(payload.get("burn_in_steps", -1)),
            "burn-in length",
        ),
    )
    for expected, observed, name in checks:
        if expected is not None and expected != observed:
            raise ValueError(f"M7-B replay batch has the wrong {name}")
    chunks = tuple(_chunk_from_payload(dict(item)) for item in payload.get("chunks", ()))
    if not chunks:
        raise ValueError("M7-B replay batch contains no chunks")
    return chunks


def build_m7b_replay_manifest(
    cache_directory: str | Path,
    *,
    corpus_manifest: dict[str, Any],
    entries: tuple[dict[str, Any], ...],
    trace_batch_size: int,
    encoder_config: dict[str, int],
    chunk_length: int,
    burn_in_steps: int,
) -> dict[str, Any]:
    root = Path(cache_directory).resolve()
    if (
        not entries
        or trace_batch_size <= 0
        or chunk_length <= 0
        or burn_in_steps < 0
    ):
        raise ValueError("M7-B replay manifest arguments are invalid")
    ordered = tuple(sorted((dict(entry) for entry in entries), key=lambda item: item["index"]))
    if [int(entry["index"]) for entry in ordered] != list(range(len(ordered))):
        raise ValueError("M7-B replay batch indices are not contiguous")
    if sum(int(entry["trace_count"]) for entry in ordered) != int(
        corpus_manifest["trace_count"]
    ):
        raise ValueError("M7-B replay cache has the wrong trace count")
    aggregate = hashlib.sha256()
    for entry in ordered:
        relative = str(entry["path"])
        digest = str(entry["sha256"])
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        aggregate.update(b"\0")
    return {
        "protocol": "m7b-replay-cache",
        "schema_version": 1,
        "complete": True,
        "errors": 0,
        "root": str(root),
        "corpus_sha256": corpus_manifest["aggregate_sha256"],
        "corpus_seed_range": corpus_manifest["seed_range"],
        "trace_count": corpus_manifest["trace_count"],
        "trace_batch_size": trace_batch_size,
        "batch_count": len(ordered),
        "encoder_config": dict(encoder_config),
        "chunk_length": chunk_length,
        "burn_in_steps": burn_in_steps,
        "aggregate_sha256": aggregate.hexdigest(),
        "files": list(ordered),
    }


def verify_m7b_replay_manifest(
    manifest_path: str | Path,
    *,
    expected_corpus_sha256: str | None = None,
    expected_encoder_config: dict[str, int] | None = None,
    expected_chunk_length: int | None = None,
    expected_burn_in_steps: int | None = None,
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != "m7b-replay-cache"
        or int(payload.get("schema_version", -1)) != 1
        or payload.get("complete") is not True
        or int(payload.get("errors", -1)) != 0
    ):
        raise ValueError("invalid M7-B replay cache manifest")
    checks = (
        (expected_corpus_sha256, payload.get("corpus_sha256"), "corpus hash"),
        (
            expected_encoder_config,
            dict(payload.get("encoder_config") or {}),
            "encoder configuration",
        ),
        (expected_chunk_length, int(payload.get("chunk_length", -1)), "chunk length"),
        (
            expected_burn_in_steps,
            int(payload.get("burn_in_steps", -1)),
            "burn-in length",
        ),
    )
    for expected, observed, name in checks:
        if expected is not None and expected != observed:
            raise ValueError(f"M7-B replay cache has the wrong {name}")
    root = Path(str(payload["root"]))
    entries = tuple(dict(entry) for entry in payload.get("files", ()))
    if (
        not entries
        or len(entries) != int(payload.get("batch_count", -1))
        or [int(entry["index"]) for entry in entries] != list(range(len(entries)))
        or sum(int(entry["trace_count"]) for entry in entries)
        != int(payload.get("trace_count", -1))
    ):
        raise ValueError("M7-B replay cache manifest is incomplete")
    aggregate = hashlib.sha256()
    for entry in entries:
        relative = str(entry["path"])
        batch_path = root / relative
        if not batch_path.is_file() or batch_path.stat().st_size != int(entry["size"]):
            raise ValueError(f"M7-B replay cache file is missing or resized: {batch_path}")
        digest = sha256_file(batch_path) if verify_file_hashes else str(entry["sha256"])
        if digest != entry["sha256"]:
            raise ValueError(f"M7-B replay cache file differs: {batch_path}")
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        aggregate.update(b"\0")
    if aggregate.hexdigest() != payload.get("aggregate_sha256"):
        raise ValueError("M7-B replay cache aggregate hash differs")
    return payload


def replay_cache_batch_paths(manifest: dict[str, Any]) -> tuple[Path, ...]:
    root = Path(str(manifest["root"]))
    paths = tuple(root / str(entry["path"]) for entry in manifest["files"])
    if not paths:
        raise ValueError("M7-B replay cache contains no batches")
    return paths


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_to_payload(chunk: ImitationChunk) -> dict[str, torch.Tensor | None]:
    return {
        "states": chunk.states,
        "actions": chunk.actions,
        "action_masks": chunk.action_masks,
        "chosen_actions": chunk.chosen_actions,
        "supervision_weights": chunk.supervision_weights,
        "supervision_phases": chunk.supervision_phases,
    }


def _chunk_from_payload(payload: dict[str, Any]) -> ImitationChunk:
    chunk = ImitationChunk(
        states=payload["states"].float(),
        actions=payload["actions"].float(),
        action_masks=payload["action_masks"].bool(),
        chosen_actions=payload["chosen_actions"].long(),
        supervision_weights=payload["supervision_weights"].float(),
        supervision_phases=(
            None
            if payload.get("supervision_phases") is None
            else payload["supervision_phases"].long()
        ),
    )
    lengths = {
        chunk.states.shape[0],
        chunk.actions.shape[0],
        chunk.action_masks.shape[0],
        chunk.chosen_actions.shape[0],
        chunk.supervision_weights.shape[0],
    }
    if chunk.supervision_phases is not None:
        lengths.add(chunk.supervision_phases.shape[0])
    if len(lengths) != 1 or not chunk.supervision_weights.any():
        raise ValueError("M7-B replay cache contains an invalid chunk")
    return chunk
