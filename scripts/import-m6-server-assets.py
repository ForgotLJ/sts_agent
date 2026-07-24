from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import and verify M6 server assets.")
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "server_assets" / "m6",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not allowed in the asset archive: {member.name}")
        target = destination.joinpath(*relative.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError(f"unsupported archive member: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read archive member: {member.name}")
        with source, target.open("wb") as stream:
            shutil.copyfileobj(source, stream)


def verify_assets(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported M6 asset manifest")
    files = dict(manifest.get("files") or {})
    for relative, expected in files.items():
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        path = root.joinpath(*relative_path.parts)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected["size"]):
            raise ValueError(f"asset size mismatch: {relative}")
        if sha256_file(path) != str(expected["sha256"]):
            raise ValueError(f"asset hash mismatch: {relative}")
    return manifest


def main() -> int:
    args = parse_args()
    archive_path = args.archive.resolve()
    destination = args.destination.resolve()
    unsafe_destinations = {
        Path(destination.anchor).resolve(),
        PROJECT_ROOT.resolve(),
        PROJECT_ROOT.resolve().parent,
    }
    if destination in unsafe_destinations:
        raise ValueError(f"unsafe asset destination: {destination}")
    if destination.name.lower() != "m6":
        raise ValueError("asset destination must end in a dedicated 'm6' directory")
    if destination.exists() and not args.force:
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="m6-import-", dir=destination.parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        with tarfile.open(archive_path, "r:gz") as archive:
            safe_extract(archive, temporary_root)
        imported_root = temporary_root / "m6"
        manifest = verify_assets(imported_root)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(imported_root, destination)
    print(
        json.dumps(
            {
                "destination": str(destination),
                "files": len(dict(manifest["files"])),
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
