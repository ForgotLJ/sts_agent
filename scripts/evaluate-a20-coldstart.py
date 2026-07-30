#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sts_env.training.a20_coldstart import (
    CHARACTERS,
    A20ValueNetwork,
    _split_examples,
    build_prefix_examples,
    evaluate_value_model,
    read_a20_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen A20 prefix value checkpoints.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--character", choices=CHARACTERS, action="append", dest="characters")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    target_device = torch.device(args.device)
    output: dict[str, object] = {
        "protocol": "a20-prefix-value-coldstart-evaluation",
        "schema_version": 1,
        "character_isolation": True,
        "input_dir": str(args.input_dir.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "characters": {},
    }
    for character in tuple(args.characters or CHARACTERS):
        records = read_a20_records(args.input_dir / f"a20_runs_{character}.jsonl")
        examples = [example for record in records for example in build_prefix_examples(record)]
        checkpoint = torch.load(
            args.model_dir / f"a20-value-{character}.pt",
            map_location=target_device,
            weights_only=True,
        )
        model = A20ValueNetwork(int(checkpoint["feature_dimension"])).to(target_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_examples = _split_examples(examples)["test"]
        output["characters"][character] = {
            "records": len(records),
            "test_metrics": evaluate_value_model(model, test_examples, target_device),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
