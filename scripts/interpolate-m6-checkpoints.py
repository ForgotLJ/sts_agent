from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training import load_m6_checkpoint, save_m6_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate related M6 checkpoints.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.candidate_weight <= 1.0:
        raise ValueError("candidate weight must be between zero and one")
    reference = load_m6_checkpoint(args.reference, device="cpu")
    candidate = load_m6_checkpoint(args.candidate, device="cpu")
    reference_state = reference.trainer.network.state_dict()
    candidate_state = candidate.trainer.network.state_dict()
    if reference_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint networks have different parameter keys")
    interpolated: dict[str, torch.Tensor] = {}
    for key, candidate_tensor in candidate_state.items():
        reference_tensor = reference_state[key]
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(f"checkpoint tensor shape differs for {key}")
        if torch.is_floating_point(candidate_tensor):
            interpolated[key] = torch.lerp(
                reference_tensor,
                candidate_tensor,
                args.candidate_weight,
            )
        else:
            interpolated[key] = candidate_tensor
    candidate.trainer.network.load_state_dict(interpolated)
    manifest = dict(candidate.manifest)
    manifest["checkpoint_interpolation"] = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "candidate_weight": args.candidate_weight,
        "evaluation_only": True,
    }
    save_m6_checkpoint(
        args.output,
        trainer=candidate.trainer,
        collector_state=candidate.collector_state,
        scheduler=candidate.scheduler,
        config=candidate.config,
        update_index=candidate.update_index,
        metrics=candidate.metrics,
        manifest=manifest,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
