#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sts_env.training.a20_card_ranking import build_card_choice_examples, train_card_ranker
from sts_env.training.a20_coldstart import read_a20_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an Ironclad A20 card-choice ranker from Heart wins.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    records = [record for record in read_a20_records(args.input) if record.get("heart_victory")]
    examples = [example for record in records for example in build_card_choice_examples(record)]
    model, metrics = train_card_ranker(
        examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "a20-card-ranker-IRONCLAD.pt"
    torch.save(
        {
            "protocol": "a20-heart-card-ranking",
            "schema_version": 1,
            "character": "IRONCLAD",
            "input": str(args.input.resolve()),
            "heart_win_records": len(records),
            "examples": len(examples),
            "state_dimension": len(examples[0].state_features),
            "config": {
                "candidate_buckets": model.config.candidate_buckets,
                "hidden_dimension": model.config.hidden_dimension,
            },
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        checkpoint,
    )
    summary = {
        "protocol": "a20-heart-card-ranking",
        "character": "IRONCLAD",
        "heart_win_records": len(records),
        "examples": len(examples),
        "checkpoint": str(checkpoint.resolve()),
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
