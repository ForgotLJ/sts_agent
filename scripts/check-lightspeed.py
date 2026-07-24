from __future__ import annotations

import argparse
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build" / "sts_lightspeed-py311"


def find_extension_modules(
    build_dir: Path,
    suffixes: tuple[str, ...] = tuple(EXTENSION_SUFFIXES),
) -> list[Path]:
    return sorted(
        {
            path
            for suffix in suffixes
            for path in build_dir.glob(f"slaythespire*{suffix}")
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the sts_lightspeed Python extension.")
    parser.add_argument("--imports", type=int, default=100, help="independent import process count")
    parser.add_argument("--seeds", type=int, default=10_000, help="deterministic initialization count")
    return parser.parse_args()


def enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def card_payload(card: Any) -> tuple[Any, ...]:
    return enum_name(card.id), bool(card.upgraded), int(card.upgrade_count), int(card.misc)


def public_payload(module: Any, game: Any, nn_interface: Any) -> dict[str, Any]:
    note_card = game.note_for_yourself_card
    return {
        "scalars": [
            enum_name(game.outcome),
            int(game.act),
            int(game.floor_num),
            enum_name(game.screen_state),
            int(game.seed),
            int(game.cur_map_node_x),
            int(game.cur_map_node_y),
            enum_name(game.cur_room),
            enum_name(game.boss),
            enum_name(game.encounter),
            int(game.cur_hp),
            int(game.max_hp),
            int(game.gold),
            bool(game.blue_key),
            bool(game.green_key),
            bool(game.red_key),
            int(game.card_rarity_factor),
            int(game.potion_chance),
            int(game.monster_chance),
            int(game.shop_chance),
            int(game.treasure_chance),
            int(game.shop_remove_count),
            int(game.speedrun_pace),
        ],
        "deck": [card_payload(card) for card in game.deck],
        "relics": [(enum_name(relic.id), int(relic.data)) for relic in game.relics],
        "note_card": card_payload(note_card),
        "observation": [int(value) for value in nn_interface.getObservation(game)],
        "observation_space_size": int(nn_interface.observation_space_size),
        "character": enum_name(module.CharacterClass.IRONCLAD),
    }


def payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_independent_imports(count: int) -> float:
    if count < 1:
        raise ValueError("--imports must be positive")

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(BUILD_DIR) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    probe = (
        "import hashlib,json,slaythespire as s;"
        "g=s.GameContext(s.CharacterClass.IRONCLAD,0,0);"
        "o=list(s.getNNInterface().getObservation(g));"
        "print(hashlib.sha256(json.dumps(o,separators=(',',':')).encode()).hexdigest())"
    )

    started = time.perf_counter()
    expected_digest: str | None = None
    for process_index in range(count):
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"import process {process_index} failed with {result.returncode}:\n{result.stderr.strip()}"
            )
        digest = result.stdout.strip()
        if not digest:
            raise RuntimeError(f"import process {process_index} produced no digest")
        if expected_digest is None:
            expected_digest = digest
        elif digest != expected_digest:
            raise AssertionError(
                f"seed 0 observation changed across processes: {expected_digest} != {digest}"
            )
    return time.perf_counter() - started


def check_seed_determinism(seed_count: int) -> tuple[float, str]:
    if seed_count < 1:
        raise ValueError("--seeds must be positive")

    sys.path.insert(0, str(BUILD_DIR))
    import slaythespire as module

    nn_interface = module.getNNInterface()
    if int(nn_interface.observation_space_size) != len(nn_interface.getObservationMaximums()):
        raise AssertionError("observation maximum vector has the wrong size")

    aggregate = hashlib.sha256()
    started = time.perf_counter()
    for seed in range(seed_count):
        first = module.SimulatorBridge(seed, 0)
        second = module.SimulatorBridge(seed, 0)
        first_digest = payload_digest(
            {"state": first.observe(), "actions": first.legal_actions()}
        )
        second_digest = payload_digest(
            {"state": second.observe(), "actions": second.legal_actions()}
        )
        if first_digest != second_digest:
            raise AssertionError(f"public initialization state differs for seed {seed}")
        aggregate.update(seed.to_bytes(8, "little", signed=False))
        aggregate.update(bytes.fromhex(first_digest))
    return time.perf_counter() - started, aggregate.hexdigest()


def main() -> int:
    args = parse_args()
    modules = find_extension_modules(BUILD_DIR)
    if len(modules) != 1:
        raise FileNotFoundError(f"expected exactly one slaythespire extension in {BUILD_DIR}, found {len(modules)}")

    import_seconds = check_independent_imports(args.imports)
    seed_seconds, aggregate_digest = check_seed_determinism(args.seeds)
    print(f"extension={modules[0]}")
    print(f"independent_imports={args.imports} seconds={import_seconds:.3f}")
    print(f"deterministic_seeds={args.seeds} seconds={seed_seconds:.3f}")
    print(f"aggregate_public_digest={aggregate_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
