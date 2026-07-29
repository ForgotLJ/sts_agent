from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from sts_env.training import (
    LightspeedEnvironmentFactory,
    evaluate_m7c_imitation,
    load_m6_checkpoint,
    load_m7_checkpoint,
    load_m7b_checkpoint,
    load_m7c_checkpoint,
)
from sts_env.training.m7b_distillation import sha256_file
from sts_env.training.m7c_dagger import (
    m7c_corpus_trace_paths,
    verify_m7c_corpus_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint against M7-C DAgger teacher labels."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--report-label", required=True)
    parser.add_argument("--trace-batch-size", type=int, default=64)
    parser.add_argument("--optimizer-batch-chunks", type=int, default=16)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--burn-in-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path, device: str) -> tuple[Any, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    protocol = str(payload.get("protocol") or "m6")
    if protocol == "m7c-dagger":
        return load_m7c_checkpoint(path, device=device), protocol
    if protocol == "m7b":
        return load_m7b_checkpoint(path, device=device), protocol
    if protocol == "m7":
        return load_m7_checkpoint(path, device=device), protocol
    return load_m6_checkpoint(path, device=device), "m6"


def main() -> int:
    args = parse_args()
    if (
        min(args.trace_batch_size, args.optimizer_batch_chunks, args.chunk_length) <= 0
        or args.burn_in_steps < 0
        or not args.report_label.strip()
    ):
        raise ValueError("M7-C imitation evaluation arguments are invalid")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M7-C CUDA evaluation requested unavailable CUDA")
    corpus = verify_m7c_corpus_manifest(args.corpus)
    loaded, checkpoint_protocol = load_checkpoint(args.checkpoint, args.device)
    metrics = evaluate_m7c_imitation(
        loaded.trainer,
        LightspeedEnvironmentFactory(),
        m7c_corpus_trace_paths(corpus),
        trace_batch_size=args.trace_batch_size,
        optimizer_batch_chunks=args.optimizer_batch_chunks,
        chunk_length=args.chunk_length,
        burn_in_steps=args.burn_in_steps,
    )
    payload = {
        "protocol": "m7c-imitation-evaluation",
        "schema_version": 1,
        "method": args.report_label.strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_protocol": checkpoint_protocol,
        "run_seed": loaded.config.run_seed,
        "corpus_manifest": str(args.corpus.resolve()),
        "corpus_sha256": corpus["aggregate_sha256"],
        "seed_range": corpus["seed_range"],
        "round_index": corpus["round_index"],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
