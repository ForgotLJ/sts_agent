from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.m7c_inputs import (
    M7C_FROZEN_INPUTS_DIRECTORY,
    verify_m7c_frozen_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely import and verify immutable M7-C training inputs."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "experiments" / M7C_FROZEN_INPUTS_DIRECTORY,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise ValueError(f"unsupported archive member: {member.name}")
        target = destination.joinpath(*relative.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read archive member: {member.name}")
        with source, target.open("wb") as stream:
            shutil.copyfileobj(source, stream)


def main() -> int:
    args = parse_args()
    archive_path = args.archive.resolve()
    destination = args.destination.resolve()
    if destination.name != M7C_FROZEN_INPUTS_DIRECTORY:
        raise ValueError("M7-C frozen input destination has the wrong directory name")
    if destination.exists() and not args.force:
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="m7c-input-import-", dir=destination.parent
    ) as directory:
        temporary = Path(directory)
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract(archive, temporary)
        imported = temporary / M7C_FROZEN_INPUTS_DIRECTORY
        verified = verify_m7c_frozen_inputs(imported)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(imported, destination)
    print(
        json.dumps(
            {
            "destination": str(destination),
            "files": len(dict(verified["files"])),
            "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
