from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch

from sts_env.training.m7b_distillation import verify_m7b_corpus_manifest
from sts_env.training.m7c_protocol import m7c_frozen_inputs_identity


M7C_FROZEN_INPUTS_PROTOCOL = "m7c-frozen-inputs"
M7C_FROZEN_INPUTS_DIRECTORY = "m7c-frozen-inputs"
M7C_FROZEN_TEACHER_DIRECTORY = "teacher-train"
M7C_FROZEN_CHECKPOINT_NAME = "m7b-seed-17-best-evaluation-checkpoint.pt"
M7C_FROZEN_M6_BASELINE_NAME = "m6-seed-17-baseline-checkpoint.pt"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"unsafe M7-C frozen input path: {value}")
    return path


def _expected_identity(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(m7c_frozen_inputs_identity() if identity is None else identity)


def checkpoint_identity(
    path: str | Path,
    *,
    expected_protocol: str,
) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    protocol = str(payload.get("protocol") or "m6")
    if protocol != expected_protocol:
        raise ValueError(
            f"M7-C frozen checkpoint must use protocol={expected_protocol}"
        )
    config = dict(payload.get("config") or {})
    manifest = dict(payload.get("manifest") or {})
    if manifest.get("evaluation_only") is not True:
        raise ValueError("M7-C frozen checkpoint must be evaluation-only")
    return {
        "protocol": protocol,
        "run_seed": int(config.get("run_seed", -1)),
        "sha256": sha256_file(checkpoint),
    }


def _file_entry(path: Path) -> dict[str, int | str]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def build_m7c_frozen_inputs_manifest(
    *,
    teacher_corpus_manifest: str | Path,
    initialization_checkpoint: str | Path,
    m6_baseline_checkpoint: str | Path,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = _expected_identity(identity)
    teacher_expected = dict(expected["teacher_corpus"])
    teacher = verify_m7b_corpus_manifest(
        teacher_corpus_manifest,
        expected_seed_start=int(teacher_expected["seed_start"]),
        expected_seed_count=int(teacher_expected["seed_count"]),
    )
    if teacher["aggregate_sha256"] != teacher_expected["aggregate_sha256"]:
        raise ValueError("M7-C frozen teacher corpus hash differs from the protocol")
    checkpoint = Path(initialization_checkpoint).resolve()
    if checkpoint_identity(checkpoint, expected_protocol="m7b") != dict(
        expected["initial_checkpoint"]
    ):
        raise ValueError("M7-C initialization checkpoint differs from the protocol")
    m6_baseline = Path(m6_baseline_checkpoint).resolve()
    if checkpoint_identity(m6_baseline, expected_protocol="m6") != dict(
        expected["m6_baseline_checkpoint"]
    ):
        raise ValueError("M7-C M6 baseline checkpoint differs from the protocol")

    teacher_root = Path(str(teacher["root"])).resolve()
    files = {
        f"{M7C_FROZEN_TEACHER_DIRECTORY}/{path.relative_to(teacher_root).as_posix()}": _file_entry(path)
        for path in sorted(candidate for candidate in teacher_root.rglob("*") if candidate.is_file())
    }
    files[M7C_FROZEN_CHECKPOINT_NAME] = _file_entry(checkpoint)
    files[M7C_FROZEN_M6_BASELINE_NAME] = _file_entry(m6_baseline)
    return {
        "protocol": M7C_FROZEN_INPUTS_PROTOCOL,
        "schema_version": 1,
        "identity": expected,
        "files": files,
    }


def verify_m7c_frozen_inputs(
    root: str | Path,
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    manifest_path = root_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _expected_identity(identity)
    if (
        payload.get("protocol") != M7C_FROZEN_INPUTS_PROTOCOL
        or int(payload.get("schema_version", -1)) != 1
        or dict(payload.get("identity") or {}) != expected
    ):
        raise ValueError("invalid M7-C frozen inputs manifest")
    files = {str(path): dict(entry) for path, entry in dict(payload.get("files") or {}).items()}
    required = {
        f"{M7C_FROZEN_TEACHER_DIRECTORY}/manifest.json",
        M7C_FROZEN_CHECKPOINT_NAME,
        M7C_FROZEN_M6_BASELINE_NAME,
    }
    if not required.issubset(files):
        raise ValueError("M7-C frozen inputs manifest lacks required files")
    for relative, entry in files.items():
        target = root_path.joinpath(*safe_relative_path(relative).parts)
        if not target.is_file() or target.stat().st_size != int(entry["size"]):
            raise ValueError(f"M7-C frozen input is missing or resized: {relative}")
        if sha256_file(target) != str(entry["sha256"]):
            raise ValueError(f"M7-C frozen input hash differs: {relative}")

    teacher_expected = dict(expected["teacher_corpus"])
    teacher = verify_m7b_corpus_manifest(
        root_path / M7C_FROZEN_TEACHER_DIRECTORY / "manifest.json",
        expected_seed_start=int(teacher_expected["seed_start"]),
        expected_seed_count=int(teacher_expected["seed_count"]),
    )
    if teacher["aggregate_sha256"] != teacher_expected["aggregate_sha256"]:
        raise ValueError("M7-C imported teacher corpus hash differs from the protocol")
    checkpoint = root_path / M7C_FROZEN_CHECKPOINT_NAME
    if checkpoint_identity(checkpoint, expected_protocol="m7b") != dict(
        expected["initial_checkpoint"]
    ):
        raise ValueError("M7-C imported initialization checkpoint differs from the protocol")
    m6_baseline = root_path / M7C_FROZEN_M6_BASELINE_NAME
    if checkpoint_identity(m6_baseline, expected_protocol="m6") != dict(
        expected["m6_baseline_checkpoint"]
    ):
        raise ValueError("M7-C imported M6 baseline checkpoint differs from the protocol")
    return {
        **payload,
        "root": str(root_path),
        "teacher_corpus_manifest": str(
            root_path / M7C_FROZEN_TEACHER_DIRECTORY / "manifest.json"
        ),
        "initialization_checkpoint": str(checkpoint),
        "m6_baseline_checkpoint": str(m6_baseline),
    }
