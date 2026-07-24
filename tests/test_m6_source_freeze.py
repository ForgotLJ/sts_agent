from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "freeze-m6-source.py"
SPEC = importlib.util.spec_from_file_location("freeze_m6_source_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load freeze-m6-source.py")
FREEZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZE)


class M6SourceFreezeTests(unittest.TestCase):
    def test_stress_gate_rejects_short_smoke(self) -> None:
        with self.assertRaisesRegex(ValueError, "10,000"):
            FREEZE.validate_required_gate_payload(
                "stress-10000",
                {
                    "complete": True,
                    "errors": 0,
                    "episodes": 10,
                    "target_episodes": 10,
                },
            )

        FREEZE.validate_required_gate_payload(
            "stress-10000",
            {
                "complete": True,
                "errors": 0,
                "episodes": 10_004,
                "target_episodes": 10_000,
            },
        )

    def test_required_gate_minimum_coverage(self) -> None:
        FREEZE.validate_required_gate_payload(
            "python-tests",
            {"complete": True, "errors": 0, "tests": 110},
        )
        FREEZE.validate_required_gate_payload(
            "prefix-recovery",
            {"complete": True, "errors": 0, "checks": 1000},
        )
        FREEZE.validate_required_gate_payload(
            "communication-differential",
            {"complete": True, "errors": 0, "records": 5, "differences": 0},
        )
        FREEZE.validate_required_gate_payload(
            "teacher-corpus",
            {
                "complete": True,
                "errors": 0,
                "stage_trace_counts": {"act1_clear": 1024, "act2_clear": 14},
                "validated_trace_count": 1038,
            },
        )


if __name__ == "__main__":
    unittest.main()
