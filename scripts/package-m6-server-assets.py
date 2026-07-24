from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package portable M6 server assets.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "m6r_recurrent_ppo_formal_v1"
        / "seed-17"
        / "checkpoint.pt",
    )
    parser.add_argument(
        "--teacher-corpus",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "m6_recurrent_ppo_formal_v1"
        / "seed-17"
        / "curriculum"
        / "teacher-v4",
    )
    parser.add_argument(
        "--communication-gate",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "m6r_formal_gates"
        / "communication-differential.json",
    )
    parser.add_argument(
        "--windows-source-freeze",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "m6r_formal_gates"
        / "source-freeze.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "m6-server-assets.tar.gz",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def main() -> int:
    args = parse_args()
    sources = {
        "seed-17/checkpoint.pt": args.checkpoint.resolve(),
        "communication-differential.json": args.communication_gate.resolve(),
        "windows-source-freeze.json": args.windows_source_freeze.resolve(),
    }
    teacher_root = args.teacher_corpus.resolve()
    for required in (*sources.values(), teacher_root):
        if not required.exists():
            raise FileNotFoundError(required)

    archive_entries: dict[str, Path] = dict(sources)
    for path in iter_files(teacher_root):
        relative = path.relative_to(teacher_root).as_posix()
        archive_entries[f"teacher-v4/{relative}"] = path

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            relative: {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
            for relative, path in sorted(archive_entries.items())
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m6-assets-") as temporary_directory:
        manifest_path = Path(temporary_directory) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(output, "w:gz", compresslevel=6) as archive:
            archive.add(manifest_path, arcname="m6/manifest.json")
            for relative, path in sorted(archive_entries.items()):
                archive.add(path, arcname=f"m6/{relative}", recursive=False)
    checksum = sha256_file(output)
    checksum_path = output.with_name(output.name + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "archive": str(output),
                "files": len(archive_entries),
                "bytes": output.stat().st_size,
                "sha256": checksum,
                "checksum": str(checksum_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
