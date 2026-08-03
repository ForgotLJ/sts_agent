from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run-a20-map-action-calibration-stage.py"


class MapActionCalibrationStageRunnerTests(unittest.TestCase):
    def test_dry_run_declares_frozen_v6_ranges_and_coverage_bounds(self) -> None:
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
                    "--source-stage",
                    str(root / "v5"),
                    "--map-checkpoint",
                    str(root / "map.pt"),
                    "--card-checkpoint",
                    str(root / "card.pt"),
                    "--output",
                    str(root / "v6"),
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
        self.assertEqual(payload["protocol"], "a20-map-action-calibration-stage-v6-dry-run")
        parameters = payload["frozen_parameters"]
        self.assertEqual(parameters["profile"]["seed_start"], 2_360_000)
        self.assertEqual(parameters["profile"]["seed_count"], 512)
        self.assertEqual(parameters["formal"]["seed_start"], 2_362_000)
        self.assertEqual(parameters["target_override_rate"], 0.075)
        self.assertEqual(parameters["profile_override_rate_interval"], [0.05, 0.10])
        self.assertEqual(parameters["smoke_min_overrides"], 2)


if __name__ == "__main__":
    unittest.main()
