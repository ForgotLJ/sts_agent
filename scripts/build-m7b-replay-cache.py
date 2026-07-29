from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from sts_env.training import (
    LightspeedEnvironmentFactory,
    RunEncoderConfig,
    RunFeatureEncoder,
    build_imitation_chunks,
    load_m6_checkpoint,
    load_m7_checkpoint,
    load_m7b_checkpoint,
)
from sts_env.training.m7b_distillation import corpus_trace_paths, verify_m7b_corpus_manifest
from sts_env.training.m7b_replay import (
    build_m7b_replay_manifest,
    load_m7b_replay_batch,
    save_m7b_replay_batch,
    sha256_file,
)
from sts_env.trace import EpisodeTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build persistent encoded replay batches for M7-B."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-batch-size", type=int, default=64)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--burn-in-steps", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_encoder_config(checkpoint: Path) -> dict[str, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    protocol = str(payload.get("protocol") or "m6")
    if protocol == "m7b":
        loaded = load_m7b_checkpoint(checkpoint)
    elif protocol == "m7":
        loaded = load_m7_checkpoint(checkpoint)
    else:
        loaded = load_m6_checkpoint(checkpoint)
    return loaded.trainer.encoder.config.to_dict()


def build_one(
    index: int,
    trace_paths: tuple[str, ...],
    trace_seeds: tuple[int, ...],
    destination: str,
    corpus_sha256: str,
    encoder_config: dict[str, int],
    chunk_length: int,
    burn_in_steps: int,
) -> dict[str, Any]:
    path = Path(destination)
    if path.is_file():
        chunks = load_m7b_replay_batch(
            path,
            expected_corpus_sha256=corpus_sha256,
            expected_encoder_config=encoder_config,
            expected_chunk_length=chunk_length,
            expected_burn_in_steps=burn_in_steps,
        )
    else:
        encoder = RunFeatureEncoder(RunEncoderConfig(**encoder_config))
        trainer = SimpleNamespace(encoder=encoder)
        traces = tuple(EpisodeTrace.read_jsonl(item) for item in trace_paths)
        chunks = build_imitation_chunks(
            LightspeedEnvironmentFactory(),
            trainer,
            traces,
            chunk_length=chunk_length,
            burn_in_steps=burn_in_steps,
            sparse_unsupervised_actions=True,
        )
        save_m7b_replay_batch(
            path,
            chunks=chunks,
            corpus_sha256=corpus_sha256,
            trace_seeds=trace_seeds,
            encoder_config=encoder_config,
            chunk_length=chunk_length,
            burn_in_steps=burn_in_steps,
        )
    return {
        "index": index,
        "path": path.name,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "trace_count": len(trace_paths),
        "trace_seeds": list(trace_seeds),
        "chunk_count": len(chunks),
        "supervised_steps": int(
            sum(torch.count_nonzero(chunk.supervision_weights).item() for chunk in chunks)
        ),
    }


def main() -> int:
    args = parse_args()
    if (
        min(args.trace_batch_size, args.chunk_length, args.workers, args.progress_interval)
        <= 0
        or args.burn_in_steps < 0
    ):
        raise ValueError("M7-B replay cache arguments are invalid")
    corpus = verify_m7b_corpus_manifest(args.corpus)
    paths = corpus_trace_paths(corpus)
    encoder_config = load_encoder_config(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    batch_count = math.ceil(len(paths) / args.trace_batch_size)
    jobs = []
    for index in range(batch_count):
        start = index * args.trace_batch_size
        selected = paths[start : start + args.trace_batch_size]
        jobs.append(
            (
                index,
                tuple(str(path) for path in selected),
                tuple(int(corpus["files"][offset]["seed"]) for offset in range(start, start + len(selected))),
                str(args.output / f"batch-{index:04d}.pt"),
                str(corpus["aggregate_sha256"]),
                encoder_config,
                args.chunk_length,
                args.burn_in_steps,
            )
        )
    started = time.perf_counter()
    entries = []
    errors = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {pool.submit(build_one, *job): job[0] for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                entries.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "index": index,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
            if completed % args.progress_interval == 0 or completed == len(jobs):
                print(
                    json.dumps(
                        {
                            "completed_batches": completed,
                            "total_batches": len(jobs),
                            "errors": len(errors),
                            "wall_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if errors:
        (args.output / "errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 1
    manifest = build_m7b_replay_manifest(
        args.output,
        corpus_manifest=corpus,
        entries=tuple(entries),
        trace_batch_size=args.trace_batch_size,
        encoder_config=encoder_config,
        chunk_length=args.chunk_length,
        burn_in_steps=args.burn_in_steps,
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol": manifest["protocol"],
                "batch_count": manifest["batch_count"],
                "trace_count": manifest["trace_count"],
                "aggregate_sha256": manifest["aggregate_sha256"],
                "size": sum(int(entry["size"]) for entry in entries),
                "wall_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
