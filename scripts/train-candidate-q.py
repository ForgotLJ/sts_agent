from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.candidate_q import CandidateQConfig
from sts_env.training.experiment import ToyExperimentConfig, run_toy_candidate_q_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Toy candidate-Q baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "candidate_q_toy.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "candidate_q_toy",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = ToyExperimentConfig.from_dict(payload["experiment"])
    trainer = CandidateQConfig.from_dict(payload["trainer"])
    if args.quick:
        experiment = ToyExperimentConfig(
            **{
                **experiment.to_dict(),
                "run_seeds": tuple(experiment.run_seeds),
                "total_steps": 1_500,
                "evaluation_seed_count": 64,
            }
        )
    summary = run_toy_candidate_q_experiment(args.output, experiment, trainer)
    print(json.dumps({
        "candidate_q_mean_score": summary["candidate_q_mean_score"],
        "random_mean_score": summary["baselines"]["random"]["mean_score"],
        "improvement_over_random": summary["improvement_over_random"],
        "claim_supported": summary["claim_supported"],
        "summary": str(args.output / "summary.json"),
    }, indent=2, sort_keys=True))
    return 0 if summary["claim_supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
