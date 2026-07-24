from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import Phase
from sts_env.training import (
    CurriculumEnvironmentFactory,
    HeuristicPolicy,
    MultiprocessRecurrentRolloutCollector,
    PrefixCorpus,
    RecurrentPPOConfig,
    SubprocessVectorEnvironment,
    load_m6_checkpoint,
)
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a resumed M6 collector step by step.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--match-training-setup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.chunk_size <= 0:
        raise ValueError("diagnostic steps and chunk size must be positive")
    loaded = load_m6_checkpoint(args.checkpoint, device=args.device)
    if args.match_training_setup:
        payload = json.loads((PROJECT_ROOT / "config" / "m6_recurrent_ppo.json").read_text())
        ppo_config = RecurrentPPOConfig(**payload["ppo"])
        loaded.trainer.config = ppo_config
        loaded.trainer.network.config = ppo_config
        for parameter_group in loaded.trainer.optimizer.param_groups:
            parameter_group["lr"] = ppo_config.learning_rate
    spec = loaded.scheduler.current
    corpus = None
    if spec.use_prefix_starts:
        corpus = PrefixCorpus.read(args.checkpoint.parent / "curriculum" / f"act-{spec.start_act}")
    factory = CurriculumEnvironmentFactory(spec=spec, prefix_corpus=corpus)
    pool = SubprocessVectorEnvironment(factory, loaded.config.num_environments)
    heuristic = HeuristicPolicy()
    collector = MultiprocessRecurrentRolloutCollector.from_state_dict(
        pool,
        loaded.trainer,
        loaded.collector_state,
        combat_selector=heuristic,
    )
    temporary_directory = tempfile.TemporaryDirectory() if args.match_training_setup else None
    writer = (
        SummaryWriter(log_dir=temporary_directory.name)
        if temporary_directory is not None
        else None
    )
    try:
        completed_steps = 0
        while completed_steps < args.steps:
            chunk_size = min(args.chunk_size, args.steps - completed_steps)
            try:
                collector.collect(chunk_size)
            except ValueError as error:
                invalid = []
                for environment_index, observation in enumerate(collector.observations):
                    if observation.phase is not Phase.TERMINAL and observation.legal_actions:
                        continue
                    trace = collector._traces[environment_index]
                    last = trace.steps[-1] if trace.steps else None
                    invalid.append(
                        {
                            "environment_index": environment_index,
                            "seed": trace.seed,
                            "trace_steps": len(trace.steps),
                            "phase": observation.phase.value,
                            "floor": observation.floor,
                            "legal_actions": len(observation.legal_actions),
                            "last_action": None if last is None else last.action.to_dict(),
                            "last_terminated": None if last is None else last.terminated,
                            "last_truncated": None if last is None else last.truncated,
                            "last_info": None if last is None else last.info,
                        }
                    )
                print(
                    json.dumps(
                        {
                            "error": str(error),
                            "completed_steps": completed_steps,
                            "active_chunk_size": chunk_size,
                            "invalid_workers": invalid,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 1
            completed_steps += chunk_size
    finally:
        collector.close()
        if writer is not None:
            writer.close()
        if temporary_directory is not None:
            temporary_directory.cleanup()
    print(
        json.dumps(
            {"completed_steps": args.steps, "chunk_size": args.chunk_size},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
