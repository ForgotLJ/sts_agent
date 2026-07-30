#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sts_env.training.a20_card_ranking import (
    A20CardRanker,
    A20CardRankingConfig,
    _split_examples,
    build_card_choice_examples,
    evaluate_card_ranker,
)
from sts_env.training.a20_coldstart import read_a20_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Ironclad A20 card-choice ranker.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    target_device = torch.device(args.device)
    records = [record for record in read_a20_records(args.input) if record.get("heart_victory")]
    examples = [example for record in records for example in build_card_choice_examples(record)]
    checkpoint = torch.load(args.checkpoint, map_location=target_device, weights_only=True)
    checkpoint_config = checkpoint["config"]
    config = A20CardRankingConfig(
        candidate_buckets=int(checkpoint_config["candidate_buckets"]),
        hidden_dimension=int(checkpoint_config["hidden_dimension"]),
    )
    model = A20CardRanker(int(checkpoint["state_dimension"]), config).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_examples = _split_examples(examples)["test"]
    result = {
        "protocol": "a20-heart-card-ranking-evaluation",
        "schema_version": 1,
        "character": "IRONCLAD",
        "heart_win_records": len(records),
        "test_metrics": evaluate_card_ranker(model, test_examples, target_device),
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
