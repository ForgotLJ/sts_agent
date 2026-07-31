#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sts_env.training.a20_coldstart import read_a20_records
from sts_env.training.a20_online_card_ranking import (
    A20OnlineCardRankingConfig,
    build_online_value_examples,
    checkpoint_config,
    train_online_value_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an observation-aligned Ironclad A20 value model."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    records = [
        record
        for record in read_a20_records(args.input)
        if record.get("character") == "IRONCLAD"
        and record.get("ascension_level") == 20
        and record.get("is_ascension_mode")
    ]
    examples = [
        example
        for record in records
        for example in build_online_value_examples(record)
    ]
    model, metrics = train_online_value_model(
        examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "a20-online-value-IRONCLAD.pt"
    torch.save(
        {
            "protocol": "a20-heart-online-value",
            "schema_version": 1,
            "character": "IRONCLAD",
            "ascension": 20,
            "input": str(args.input.resolve()),
            "a20_records": len(records),
            "examples": len(examples),
            "state_dimension": model.body[0].in_features,
            "hidden_dimension": model.body[0].out_features,
            "config": checkpoint_config(A20OnlineCardRankingConfig()),
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        checkpoint,
    )
    summary = {
        "protocol": "a20-heart-online-value",
        "character": "IRONCLAD",
        "ascension": 20,
        "a20_records": len(records),
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
