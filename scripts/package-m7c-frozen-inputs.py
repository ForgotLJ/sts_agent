from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.m7c_inputs import (
    M7C_FROZEN_CHECKPOINT_NAME,
    M7C_FROZEN_INPUTS_DIRECTORY,
    M7C_FROZEN_M6_BASELINE_NAME,
    M7C_FROZEN_TEACHER_DIRECTORY,
    build_m7c_frozen_inputs_manifest,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the immutable M7-C teacher corpus and initialization checkpoint."
    )
    parser.add_argument("--teacher-corpus", type=Path, required=True)
    parser.add_argument("--initialization-checkpoint", type=Path, required=True)
    parser.add_argument("--m6-baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "m7c-frozen-inputs-v1.tar.gz",
    )
    return parser.parse_args()


def write_checksum_sidecar(output: Path, checksum: str) -> Path:
    checksum_path = output.with_name(output.name + ".sha256")
    checksum_path.write_bytes(f"{checksum}  {output.name}\n".encode("ascii"))
    return checksum_path


def main() -> int:
    args = parse_args()
    manifest = build_m7c_frozen_inputs_manifest(
        teacher_corpus_manifest=args.teacher_corpus,
        initialization_checkpoint=args.initialization_checkpoint,
        m6_baseline_checkpoint=args.m6_baseline_checkpoint,
    )
    teacher_root = Path(str(args.teacher_corpus.resolve().parent))
    checkpoint = args.initialization_checkpoint.resolve()
    m6_baseline = args.m6_baseline_checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m7c-frozen-inputs-") as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(output, "w:gz", compresslevel=6) as archive:
            archive.add(
                manifest_path,
                arcname=f"{M7C_FROZEN_INPUTS_DIRECTORY}/manifest.json",
                recursive=False,
            )
            for relative in sorted(manifest["files"]):
                source = (
                    checkpoint
                    if relative == M7C_FROZEN_CHECKPOINT_NAME
                    else m6_baseline
                    if relative == M7C_FROZEN_M6_BASELINE_NAME
                    else teacher_root
                    / Path(relative).relative_to(M7C_FROZEN_TEACHER_DIRECTORY)
                )
                archive.add(
                    source,
                    arcname=f"{M7C_FROZEN_INPUTS_DIRECTORY}/{relative}",
                    recursive=False,
                )
    checksum = sha256_file(output)
    checksum_path = write_checksum_sidecar(output, checksum)
    print(
        json.dumps(
            {
                "archive": str(output),
                "checksum": str(checksum_path),
                "files": len(manifest["files"]),
                "bytes": output.stat().st_size,
                "sha256": checksum,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
