from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.search.distillation import PolicyValueConfig
from sts_env.search.lightspeed_experiment import (
    LightspeedCombatExperimentConfig,
    run_lightspeed_combat_experiment,
)
from sts_env.search.mcts import BeliefSearchConfig
from sts_env.training.candidate_q import CandidateQConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M5 Lightspeed first-combat experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "m5_lightspeed_combat.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "m5_lightspeed_combat",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = LightspeedCombatExperimentConfig.from_dict(payload["experiment"])
    if args.quick:
        experiment = LightspeedCombatExperimentConfig(
            **{
                **experiment.to_dict(),
                "run_seeds": tuple(experiment.run_seeds),
                "search_call_budgets": (8, 32),
                "candidate_q_combat_steps": 600,
                "teacher_episodes": 32,
                "teacher_call_budget": 32,
                "distillation_updates": 200,
                "evaluation_seed_count": 8,
            }
        )
    summary = run_lightspeed_combat_experiment(
        args.output,
        experiment,
        CandidateQConfig.from_dict(payload["candidate_q"]),
        PolicyValueConfig.from_dict(payload["policy_value"]),
        BeliefSearchConfig.from_dict(payload["search"]),
    )
    print(
        json.dumps(
            {
                "claim_supported": summary["claim_supported"],
                "gates": summary["gates"],
                "scope": summary["scope"],
                "summary": str(args.output / "summary.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["claim_supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
