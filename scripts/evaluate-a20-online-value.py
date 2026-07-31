#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sts_env.training.a20_coldstart import read_a20_records
from sts_env.training.a20_online_card_ranking import (
    build_online_value_examples,
    evaluate_online_value_model,
    load_online_value_model,
    split_online_value_examples,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen observation-aligned Ironclad A20 value model."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    model = load_online_value_model(args.checkpoint, args.device)
    test_examples = split_online_value_examples(examples)["test"]
    result = {
        "protocol": "a20-heart-online-value-evaluation",
        "schema_version": 1,
        "character": "IRONCLAD",
        "ascension": 20,
        "a20_records": len(records),
        "test_metrics": evaluate_online_value_model(
            model,
            test_examples,
            torch.device(args.device),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
