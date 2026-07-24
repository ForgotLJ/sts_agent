from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class ServerPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = load_script("run_m6_server_test", "scripts/run-m6-server.py")
        cls.audit = load_script("audit_m6_test", "scripts/audit-m6.py")
        cls.importer = load_script(
            "import_m6_server_assets_test", "scripts/import-m6-server-assets.py"
        )

    def test_resource_profiles_scale_orchestration_without_changing_formal_config(self) -> None:
        conservative = self.pipeline.resource_defaults("conservative", 36)
        maximum = self.pipeline.resource_defaults("max", 36)

        self.assertEqual(conservative.parallel_train_runs, 1)
        self.assertEqual(maximum.parallel_train_runs, 3)
        self.assertEqual(maximum.parallel_evaluations, 5)
        self.assertEqual(maximum.stress_workers, 32)
        self.assertEqual(maximum.omp_threads, 1)

    def test_empty_completion_audit_exposes_all_29_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = self.audit.audit(
                root / "pipeline",
                root / "gates",
                root / "training",
                root / "evaluations",
            )

        self.assertEqual(payload["status"], "incomplete")
        self.assertEqual(payload["checks_total"], 29)
        self.assertEqual(payload["checks_passed"], 0)

    def test_asset_import_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "unsafe.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                payload = b"unsafe"
                member = tarfile.TarInfo("../escape.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            with tarfile.open(archive_path, "r:gz") as archive:
                with self.assertRaises(ValueError):
                    self.importer.safe_extract(archive, Path(temporary_directory) / "out")

    def test_asset_manifest_verifies_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "seed-17" / "checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            manifest = {
                "schema_version": 1,
                "files": {
                    "seed-17/checkpoint.pt": {
                        "sha256": self.importer.sha256_file(checkpoint),
                        "size": checkpoint.stat().st_size,
                    }
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            verified = self.importer.verify_assets(root)

        self.assertEqual(verified["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
