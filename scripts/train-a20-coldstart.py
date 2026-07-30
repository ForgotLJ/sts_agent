#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sts_env.training.a20_coldstart import (
    CHARACTERS,
    build_prefix_examples,
    read_a20_records,
    train_value_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train independent A20 prefix value models.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--character", choices=CHARACTERS, action="append", dest="characters")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_records is not None and args.max_records <= 0:
        raise SystemExit("--max-records must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    characters = tuple(args.characters or CHARACTERS)
    summary: dict[str, object] = {
        "protocol": "a20-prefix-value-coldstart",
        "schema_version": 1,
        "character_isolation": True,
        "input_dir": str(args.input_dir.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "characters": {},
    }
    for character in characters:
        input_path = args.input_dir / f"a20_runs_{character}.jsonl"
        records = read_a20_records(input_path, args.max_records)
        examples = [example for record in records for example in build_prefix_examples(record)]
        model, metrics = train_value_model(
            examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        checkpoint_path = args.output_dir / f"a20-value-{character}.pt"
        torch.save(
            {
                "protocol": summary["protocol"],
                "schema_version": 1,
                "character": character,
                "input_path": str(input_path.resolve()),
                "feature_dimension": len(examples[0].features),
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )
        summary["characters"][character] = {
            "records": len(records),
            "examples": len(examples),
            "checkpoint": str(checkpoint_path.resolve()),
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
