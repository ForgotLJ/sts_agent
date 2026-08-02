#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env.training.map_action_value import (
    MapActionValueConfig,
    load_map_action_value_examples,
    save_map_action_value_checkpoint,
    sha256_file,
    train_map_action_value_model,
)
from sts_env.training.map_counterfactual import validate_map_counterfactual_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a root-seed-split A20 Ironclad map-action value model."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-evaluation", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--groups-per-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"checkpoint output already exists: {args.output}")
    frozen_evaluation = args.frozen_evaluation or args.output.with_name("frozen-evaluation.json")
    if frozen_evaluation.exists():
        raise FileExistsError(f"frozen evaluation output already exists: {frozen_evaluation}")
    validation = validate_map_counterfactual_corpus(args.input)
    if not validation["valid"] or not validation["complete"]:
        raise ValueError("refusing to train from an invalid or incomplete map counterfactual corpus")
    config = MapActionValueConfig(
        hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
    )
    examples, manifest = load_map_action_value_examples(args.input)
    trained_acts = sorted(
        int(act)
        for act, count in dict(manifest.get("counts") or {}).items()
        if int(count) > 0
    )
    if not trained_acts or any(act not in {1, 2, 3} for act in trained_acts):
        raise ValueError("map counterfactual corpus does not declare valid trained acts")
    trained_floor_range = [
        min(example.floor for example in examples),
        max(example.floor for example in examples),
    ]
    model, encoder, metrics = train_map_action_value_model(
        examples,
        config=config,
        epochs=args.epochs,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    metadata: dict[str, Any] = {
        "character": "IRONCLAD",
        "ascension": 20,
        "trained_acts": trained_acts,
        "trained_floor_range": trained_floor_range,
        "corpus": {
            "path": str(args.input.resolve()),
            "records_sha256": validation["records_sha256"],
            "manifest": manifest,
        },
        "training": {
            "epochs": args.epochs,
            "groups_per_batch": args.groups_per_batch,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": args.device,
        },
    }
    save_map_action_value_checkpoint(
        args.output,
        model,
        encoder,
        metrics=metrics,
        metadata=metadata,
    )
    result = {
        "protocol": "a20-map-action-value-offline-evaluation",
        "schema_version": 1,
        "checkpoint": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
        "feature_dimension": encoder.dimension,
        "config": config.to_dict(),
        "trained_acts": trained_acts,
        "trained_floor_range": trained_floor_range,
        "training": metadata["training"],
        "corpus": {
            "records_sha256": validation["records_sha256"],
            "records": validation["records"],
            "act_counts": validation["act_counts"],
        },
        "metrics": metrics,
    }
    frozen_evaluation.parent.mkdir(parents=True, exist_ok=True)
    frozen_evaluation.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
