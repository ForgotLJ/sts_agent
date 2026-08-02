from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run-a20-map-action-stage.py"


class MapActionStageRunnerTests(unittest.TestCase):
    def test_dry_run_exposes_the_full_frozen_stage_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "stage"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(PROJECT_ROOT / ".local_packages"), str(PROJECT_ROOT / "src"))
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--pilot",
                    str(root / "pilot"),
                    "--output",
                    str(output),
                    "--card-checkpoint",
                    str(root / "card.pt"),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(output.exists())
        payload = json.loads(result.stdout)
        self.assertEqual(payload["protocol"], "a20-map-action-act1-stage-v4-dry-run")
        self.assertEqual(
            [step["name"] for step in payload["steps"]],
            [
                "pilot_diagnostic",
                "collect_corpus",
                "corpus_diagnostic",
                "train_map_model",
                "profile_map_policy",
                "smoke_map_policy",
                "formal_map_policy",
                "replication_map_policy",
                "audit",
            ],
        )
        self.assertEqual(payload["frozen_parameters"]["collection"]["seed_count"], 4096)
        self.assertEqual(payload["frozen_parameters"]["collection"]["seed_start"], 2_322_000)
        self.assertEqual(payload["frozen_parameters"]["collection"]["per_act"], 300)
        self.assertEqual(payload["frozen_parameters"]["collection"]["acts"], [1])
        self.assertEqual(payload["frozen_parameters"]["trained_floor_range"], [0, 0])
        self.assertEqual(
            payload["frozen_parameters"]["evaluation"]["formal"]["seed_start"],
            2_334_000,
        )
        self.assertEqual(payload["frozen_parameters"]["evaluation"]["formal"]["seed_count"], 512)
        self.assertEqual(payload["frozen_parameters"]["evaluation"]["margin_quantile"], "p80")

    def test_dry_run_rejects_an_unfrozen_bootstrap_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--pilot",
                    str(root / "pilot"),
                    "--output",
                    str(root / "stage"),
                    "--card-checkpoint",
                    str(root / "card.pt"),
                    "--bootstrap-samples",
                    "1000",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bootstrap samples are frozen", result.stderr)

    def test_v5_dry_run_declares_fresh_advantage_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(PROJECT_ROOT / ".local_packages"), str(PROJECT_ROOT / "src"))
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--stage-version",
                    "v5",
                    "--pilot",
                    str(root / "pilot"),
                    "--output",
                    str(root / "stage"),
                    "--card-checkpoint",
                    str(root / "card.pt"),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["protocol"], "a20-map-action-act1-stage-v5-dry-run")
        self.assertEqual(payload["frozen_parameters"]["collection"]["per_act"], 1024)
        self.assertEqual(payload["frozen_parameters"]["collection"]["particles_per_action"], 8)
        self.assertEqual(payload["frozen_parameters"]["evaluation"]["margin_quantile"], "p95")
        self.assertEqual(
            payload["frozen_parameters"]["evaluation"]["formal"]["seed_start"],
            2_352_000,
        )
        self.assertEqual(
            payload["frozen_parameters"]["label_mode"],
            "behavior_relative_final_floor_advantage",
        )


if __name__ == "__main__":
    unittest.main()
