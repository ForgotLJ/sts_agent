from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import json
from pathlib import Path
import random
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import ActionKind, LightspeedBackend, Observation, Phase, StsEnv


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


def working_set_bytes() -> int:
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCountersEx),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_bool
    process = get_current_process()
    success = get_process_memory_info(process, ctypes.byref(counters), counters.cb)
    if not success:
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.working_set_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuzz the sts_lightspeed environment contract.")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0x5EED)
    parser.add_argument("--clone-every", type=int, default=1_000)
    parser.add_argument("--json-every", type=int, default=1_000)
    parser.add_argument("--memory-every", type=int, default=10_000)
    parser.add_argument("--max-memory-growth-mb", type=int, default=256)
    return parser.parse_args()


def assert_sorted_counts(name: str, counts: tuple[tuple[str, int], ...]) -> None:
    if counts != tuple(sorted(counts)):
        raise AssertionError(f"{name} is not sorted")
    if any(not card_id or count <= 0 for card_id, count in counts):
        raise AssertionError(f"{name} contains an invalid card count")


def assert_invariants(observation: Observation) -> None:
    if observation.player.max_hp <= 0:
        raise AssertionError("max_hp must be positive")
    if observation.player.hp < 0 or observation.player.hp > observation.player.max_hp:
        raise AssertionError("player hp is outside [0, max_hp]")
    if observation.player.block < 0 or observation.player.energy < 0:
        raise AssertionError("block and energy must be non-negative")
    if observation.act < 1 or observation.act > 4:
        raise AssertionError("act is outside the simulator range")
    if observation.floor < 0 or observation.floor > 60:
        raise AssertionError("floor is outside the simulator range")
    if len(set(observation.legal_actions)) != len(observation.legal_actions):
        raise AssertionError("legal action list contains duplicates")

    hand_ids = {card.instance_id for card in observation.hand}
    if len(hand_ids) != len(observation.hand):
        raise AssertionError(
            "hand contains duplicate card instance ids: "
            + str([(card.instance_id, card.card_id) for card in observation.hand])
        )
    enemy_ids = {enemy.enemy_id for enemy in observation.enemies}

    if observation.phase is Phase.TERMINAL:
        if observation.legal_actions:
            raise AssertionError("terminal state contains legal actions")
    elif not observation.legal_actions:
        raise AssertionError("non-terminal state contains no legal actions")

    for action in observation.legal_actions:
        if action.kind is ActionKind.PLAY_CARD:
            if action.source_id not in hand_ids:
                raise AssertionError("play-card action references a missing hand instance")
            if action.target_id is not None and action.target_id not in enemy_ids:
                raise AssertionError("play-card action references a missing enemy")
        if action.kind is ActionKind.END_TURN and observation.phase is not Phase.COMBAT:
            raise AssertionError("end-turn action exists outside combat")

    assert_sorted_counts("draw_pile", observation.draw_pile)
    assert_sorted_counts("discard_pile", observation.discard_pile)
    assert_sorted_counts("exhaust_pile", observation.exhaust_pile)
    assert_sorted_counts("deck", observation.deck)


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    random_source = random.Random(args.policy_seed)
    environment = StsEnv(LightspeedBackend())
    episode_seed = args.seed_start
    observation, _ = environment.reset(seed=episode_seed)
    episodes = 0
    phase_counts: Counter[str] = Counter()
    action_kind_counts: Counter[str] = Counter()
    memory_samples: list[tuple[int, int]] = [(0, working_set_bytes())]

    started = time.perf_counter()
    for step_index in range(1, args.steps + 1):
        try:
            assert_invariants(observation)
        except AssertionError as error:
            raise AssertionError(
                f"global_step={step_index} episode_seed={episode_seed} "
                f"phase={observation.phase.value} floor={observation.floor}: {error}"
            ) from error
        phase_counts[observation.phase.value] += 1

        if args.json_every > 0 and step_index % args.json_every == 0:
            encoded = json.dumps(observation.to_dict(), ensure_ascii=False, sort_keys=True)
            if "rng" in encoded.lower() or "draw_order" in encoded.lower():
                raise AssertionError("serialized public observation exposes hidden state")
            if Observation.from_dict(json.loads(encoded)) != observation:
                raise AssertionError("observation JSON round trip changed the state")

        action = random_source.choice(observation.legal_actions)
        action_kind_counts[action.kind.value] += 1

        if args.clone_every > 0 and step_index % args.clone_every == 0:
            branch = environment.clone()
            branch_transition = branch.step(action)
            transition = environment.step(action)
            if transition != branch_transition:
                raise AssertionError(f"clone replay diverged at global step {step_index}")
        else:
            transition = environment.step(action)

        observation, _, terminated, truncated, _ = transition
        if terminated or truncated:
            assert_invariants(observation)
            episodes += 1
            episode_seed += 1
            observation, _ = environment.reset(seed=episode_seed)

        if args.memory_every > 0 and step_index % args.memory_every == 0:
            memory_samples.append((step_index, working_set_bytes()))

    elapsed = time.perf_counter() - started
    memory_samples.append((args.steps, working_set_bytes()))
    stable_samples = memory_samples[1:] if len(memory_samples) > 2 else memory_samples
    memory_growth = stable_samples[-1][1] - stable_samples[0][1]
    growth_limit = args.max_memory_growth_mb * 1024 * 1024
    if memory_growth > growth_limit:
        raise AssertionError(
            f"working set grew by {memory_growth / 1024 / 1024:.1f} MiB, "
            f"above the {args.max_memory_growth_mb} MiB limit"
        )

    print(f"steps={args.steps}")
    print(f"episodes={episodes}")
    print(f"seconds={elapsed:.3f}")
    print(f"steps_per_second={args.steps / elapsed:.1f}")
    print(f"phase_counts={dict(sorted(phase_counts.items()))}")
    print(f"action_kind_counts={dict(sorted(action_kind_counts.items()))}")
    print(
        "memory_samples_mib="
        + str([(step, round(value / 1024 / 1024, 2)) for step, value in memory_samples])
    )
    print(f"stable_memory_growth_mib={memory_growth / 1024 / 1024:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
