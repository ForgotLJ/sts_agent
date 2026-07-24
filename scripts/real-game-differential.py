from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sts_env import Action, ActionKind, CommunicationBackend, LightspeedBackend
from sts_env.differential import (
    DifferentialAllowlist,
    canonical_observation,
    compare_observations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-seed real-game differential smoke test.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51234)
    parser.add_argument(
        "--connect-wait",
        type=float,
        default=60.0,
        help="seconds to wait for CommunicationMod to start its relay",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="attach to an interrupted in-game relay state instead of sending START",
    )
    parser.add_argument(
        "--reference-trace",
        type=Path,
        help="replay the latest matching CommunicationMod session offline",
    )
    parser.add_argument(
        "--resume-trace",
        type=Path,
        help="reconstruct the simulator by replaying commands from a relay JSONL trace",
    )
    parser.add_argument(
        "--resume-prefix-trace",
        type=Path,
        help="older relay trace used to constrain the initial map branch before resume-trace",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=PROJECT_ROOT / "config" / "differential_allowlist.json",
    )
    parser.add_argument(
        "--act1-boss-history",
        choices=(
            "guardian_unseen",
            "hexaghost_unseen",
            "slime_boss_unseen",
            "all_seen",
        ),
        default="all_seen",
        help="profile-dependent Act 1 boss unlock history",
    )
    parser.add_argument(
        "--neow-history",
        choices=("auto", "full", "limited", "skipped"),
        default="auto",
        help="profile-dependent Neow history; explicit is required at the ambiguous Talk screen",
    )
    parser.add_argument(
        "--final-act-unlocked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether the profile has unlocked keys and the burning elite",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "real_game_traces" / "differential.jsonl"),
        help="JSONL output path, or '-' to emit records on stdout",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def card_id_for_action(observation: Any, action: Action) -> str | None:
    return next(
        (
            card.card_id
            for card in observation.hand
            if card.instance_id == action.source_id
        ),
        None,
    )


def presentation_action(observation: Any) -> Action | None:
    presentation_candidates = [
        action
        for action in observation.legal_actions
        if action.kind not in {ActionKind.USE_POTION, ActionKind.DISCARD_POTION}
    ]
    if len(presentation_candidates) != 1:
        return None
    action = presentation_candidates[0]
    normalized_label = " ".join(action.label.lower().split())
    if normalized_label in {"talk", "continue", "next", "proceed", "leave", "return"}:
        return action
    return None


def semantic_source_id(action: Action) -> str | None:
    if action.source_id is None:
        return None
    normalized = "".join(
        character for character in str(action.source_id).lower() if character.isalnum() or character == ":"
    )
    return normalized or None


def actions_semantically_match(reference: Action, candidate: Action) -> bool:
    if reference.kind is not candidate.kind:
        return False
    reference_source = semantic_source_id(reference)
    candidate_source = semantic_source_id(candidate)
    if reference_source is not None and candidate_source is not None:
        return (
            reference_source == candidate_source
            and reference.target_id == candidate.target_id
        )
    return (
        reference.choice_index == candidate.choice_index
        and reference.target_id == candidate.target_id
    )


def pair_action(reference: Any, candidate: Any) -> tuple[Action, Action | None] | None:
    reference_presentation = presentation_action(reference)
    if reference_presentation is not None:
        candidate_presentation = presentation_action(candidate)
        if (
            candidate_presentation is not None
            and actions_semantically_match(reference_presentation, candidate_presentation)
        ):
            return reference_presentation, candidate_presentation
        return reference_presentation, None

    if reference.phase != candidate.phase:
        return None

    if reference.phase.value == "shop":
        reference_proceed = next(
            (action for action in reference.legal_actions if action.source_id == "proceed"),
            None,
        )
        candidate_proceed = next(
            (action for action in candidate.legal_actions if action.source_id == "proceed"),
            None,
        )
        if reference_proceed is not None and candidate_proceed is not None:
            return reference_proceed, candidate_proceed

    for potion_kind in (ActionKind.USE_POTION, ActionKind.DISCARD_POTION):
        for reference_action in reference.legal_actions:
            if reference_action.kind is not potion_kind:
                continue
            for candidate_action in candidate.legal_actions:
                if actions_semantically_match(reference_action, candidate_action):
                    return reference_action, candidate_action

    for reference_action in reference.legal_actions:
        if reference_action.kind is not ActionKind.PLAY_CARD:
            continue
        reference_card_id = card_id_for_action(reference, reference_action)
        for candidate_action in candidate.legal_actions:
            if candidate_action.kind is not ActionKind.PLAY_CARD:
                continue
            if candidate_action.target_id != reference_action.target_id:
                continue
            if card_id_for_action(candidate, candidate_action) == reference_card_id:
                return reference_action, candidate_action

    for action_kind in (
        ActionKind.END_TURN,
        ActionKind.CHOOSE_OPTION,
        ActionKind.CHOOSE_MAP_NODE,
        ActionKind.CHOOSE_CARD,
        ActionKind.LEAVE,
        ActionKind.BUY,
        ActionKind.REMOVE_CARD,
    ):
        reference_actions = [action for action in reference.legal_actions if action.kind is action_kind]
        candidate_actions = [action for action in candidate.legal_actions if action.kind is action_kind]
        for reference_action in reference_actions:
            for candidate_action in candidate_actions:
                if actions_semantically_match(reference_action, candidate_action):
                    return reference_action, candidate_action
    return None


def infer_neow_history(observation: Any) -> str:
    if observation.floor != 0:
        return "full"
    if observation.phase.value == "map":
        return "skipped"
    if observation.phase.value == "event":
        if presentation_action(observation) is not None:
            raise RuntimeError(
                "Neow history is not observable at the Talk screen; "
                "pass --neow-history full or --neow-history limited"
            )
        option_count = sum(
            action.kind is ActionKind.CHOOSE_OPTION
            for action in observation.legal_actions
        )
        if option_count <= 2:
            return "limited"
    return "full"


def resolve_neow_history(requested: str, observation: Any) -> str:
    return infer_neow_history(observation) if requested == "auto" else requested


def alignment_fingerprint(observation: Any) -> str:
    return json.dumps(
        canonical_observation(observation),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def align_candidate_to_reference(
    reference: Any,
    seed: int,
    neow_history: str = "auto",
    act1_boss_history: str = "all_seen",
    final_act_unlocked: bool = True,
    max_depth: int = 5,
    max_nodes: int = 512,
    required_initial_map_source: str | None = None,
) -> tuple[LightspeedBackend, Any, dict[str, Any]]:
    target = alignment_fingerprint(reference)
    inferred = resolve_neow_history(neow_history, reference)
    histories = (
        [inferred]
        if neow_history != "auto"
        else [inferred]
        + [
            history
            for history in ("skipped", "limited", "full")
            if history != inferred
        ]
    )

    for history in histories:
        root_backend = LightspeedBackend(
            neow_history=history,
            act1_boss_history=act1_boss_history,
            final_act_unlocked=final_act_unlocked,
        )
        root, info = root_backend.reset(seed=seed)
        queue = deque([(root_backend, root, tuple(), False)])
        visited: set[str] = set()
        expanded = 0

        while queue and expanded < max_nodes:
            backend, observation, path, initial_map_matched = queue.popleft()
            fingerprint = alignment_fingerprint(observation)
            if fingerprint == target and (
                required_initial_map_source is None or initial_map_matched
            ):
                return backend, observation, {
                    **info,
                    "alignment_depth": len(path),
                    "alignment_actions": list(path),
                    "required_initial_map_source": required_initial_map_source,
                }
            if fingerprint in visited:
                continue
            visited.add(fingerprint)
            expanded += 1

            if len(path) >= max_depth or observation.phase.value in {"combat", "terminal"}:
                continue
            for action in observation.legal_actions:
                next_initial_map_matched = initial_map_matched
                if (
                    action.kind is ActionKind.CHOOSE_MAP_NODE
                    and required_initial_map_source is not None
                    and not initial_map_matched
                ):
                    if semantic_source_id(action) != required_initial_map_source:
                        continue
                    next_initial_map_matched = True
                branch = backend.clone()
                try:
                    transition = branch.step(action)
                except (RuntimeError, ValueError):
                    continue
                queue.append(
                    (
                        branch,
                        transition.observation,
                        path + (action.label,),
                        next_initial_map_matched,
                    )
                )

    raise RuntimeError(
        "could not reconstruct the real state from a deterministic simulator prefix; "
        f"seed={seed} phase={reference.phase.value} floor={reference.floor}"
    )


def observation_from_relay_state(
    parser: CommunicationBackend,
    state: dict[str, Any],
) -> Any:
    parser._state = state
    game = dict(state.get("game_state") or {})
    map_x, map_y = parser._map_coordinates(game)
    if map_x >= 0 and map_y >= 0:
        parser._map_x = map_x
        parser._map_y = map_y
    return parser._read_observation()


def replay_action_for_command(
    command: str,
    reference: Any,
    candidate: Any,
) -> Action | None:
    parts = command.split()
    if not parts:
        raise RuntimeError("relay trace contains an empty command")
    if parts[0] == "play":
        hand_index = int(parts[1]) - 1
        target_id = int(parts[2]) if len(parts) > 2 else None
        card_id = reference.hand[hand_index].card_id
        return next(
            action
            for action in candidate.legal_actions
            if action.kind is ActionKind.PLAY_CARD
            and action.target_id == target_id
            and card_id_for_action(candidate, action) == card_id
        )
    if parts[0] == "end":
        return next(
            action for action in candidate.legal_actions if action.kind is ActionKind.END_TURN
        )
    if parts[0] == "choose":
        reference_action = next(
            action
            for action in reference.legal_actions
            if action.choice_index == int(parts[1])
        )
        if presentation_action(reference) == reference_action:
            candidate_presentation = presentation_action(candidate)
            if (
                candidate_presentation is not None
                and actions_semantically_match(reference_action, candidate_presentation)
            ):
                return candidate_presentation
            return None
        matches = [
            action
            for action in candidate.legal_actions
            if actions_semantically_match(reference_action, action)
        ]
        if not matches:
            raise RuntimeError(
                f"cannot replay '{command}' by semantic action; "
                f"reference={reference_action.to_dict()} "
                f"candidate_actions={[action.to_dict() for action in candidate.legal_actions]}"
            )
        return matches[0]
    if parts[:2] == ["potion", "use"]:
        source_id = int(parts[2])
        target_id = int(parts[3]) if len(parts) > 3 else None
        return next(
            action
            for action in candidate.legal_actions
            if action.kind is ActionKind.USE_POTION
            and action.source_id == source_id
            and action.target_id == target_id
        )
    if parts[:2] == ["potion", "discard"]:
        source_id = int(parts[2])
        return next(
            action
            for action in candidate.legal_actions
            if action.kind is ActionKind.DISCARD_POTION and action.source_id == source_id
        )
    if parts[0] in {"proceed", "confirm"}:
        reference_presentation = presentation_action(reference)
        candidate_presentation = presentation_action(candidate)
        if reference_presentation is not None and candidate_presentation is None:
            return None
        matches = [
            action
            for action in candidate.legal_actions
            if action.source_id == "proceed" or candidate_presentation == action
        ]
        if not matches:
            raise RuntimeError(
                f"cannot replay '{command}'; "
                f"candidate_actions={[action.to_dict() for action in candidate.legal_actions]}"
            )
        return matches[0]
    if parts[0] in {"return", "cancel", "leave", "skip"}:
        return next(
            action for action in candidate.legal_actions if action.kind is ActionKind.LEAVE
        )
    if parts[0] in {"key", "click"}:
        return None
    raise RuntimeError(f"cannot replay relay command: {command}")


def align_candidate_reward_stage(
    reference: Any,
    backend: LightspeedBackend,
    candidate: Any,
) -> Any:
    reference_card_screen = any(
        action.kind is ActionKind.CHOOSE_CARD for action in reference.legal_actions
    )
    candidate_card_screen = any(
        action.kind is ActionKind.CHOOSE_CARD for action in candidate.legal_actions
    )
    reference_card_entry = any(
        action.kind is ActionKind.CHOOSE_OPTION and action.source_id == "card"
        for action in reference.legal_actions
    )
    if reference_card_screen and not candidate_card_screen:
        card_entry = next(
            action
            for action in candidate.legal_actions
            if action.kind is ActionKind.CHOOSE_OPTION and action.source_id == "card"
        )
        return backend.step(card_entry).observation
    if reference_card_entry and candidate_card_screen:
        backend._pending_card_reward = None
        return backend._read_observation()
    return candidate


def latest_map_source_from_trace(trace_path: Path, seed: int) -> str | None:
    latest_source: str | None = None
    last_game: dict[str, Any] | None = None
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("direction") == "game_to_agent":
            game = dict((record.get("payload") or {}).get("game_state") or {})
            last_game = game if int(game.get("seed", -1)) == seed else None
            continue
        command = str(record.get("command") or "")
        if last_game is None or last_game.get("screen_type") != "MAP" or not command.startswith("choose "):
            continue
        choice_index = int(command.split()[1])
        choices = list(last_game.get("choice_list") or [])
        if 0 <= choice_index < len(choices):
            normalized = "".join(
                character for character in str(choices[choice_index]).lower() if character.isalnum()
            )
            if normalized.startswith("x"):
                latest_source = normalized
    return latest_source


def reconstruct_candidate_from_trace(
    reference: Any,
    seed: int,
    trace_path: Path,
    prefix_trace_path: Path | None = None,
    neow_history: str = "auto",
    act1_boss_history: str = "all_seen",
    final_act_unlocked: bool = True,
) -> tuple[LightspeedBackend, Any, dict[str, Any]]:
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parser = CommunicationBackend()
    last_actionable_state: dict[str, Any] | None = None
    start_index: int | None = None
    start_reference: Any | None = None

    for record_index, record in enumerate(records):
        if record.get("direction") == "game_to_agent":
            state = dict(record.get("payload") or {})
            if state.get("in_game") and CommunicationBackend._is_actionable_state(state):
                last_actionable_state = state
            continue
        command = str(record.get("command") or "")
        if (
            record.get("direction") == "agent_to_game"
            and command != "state"
            and not command.startswith("wait")
            and last_actionable_state is not None
        ):
            start_index = record_index
            start_reference = observation_from_relay_state(parser, last_actionable_state)
            break

    if start_index is None or start_reference is None:
        raise RuntimeError(f"relay trace contains no replayable in-game actions: {trace_path}")

    required_initial_map_source = (
        latest_map_source_from_trace(prefix_trace_path, seed)
        if prefix_trace_path is not None
        else None
    )
    backend, candidate, info = align_candidate_to_reference(
        start_reference,
        seed,
        neow_history=neow_history,
        act1_boss_history=act1_boss_history,
        final_act_unlocked=final_act_unlocked,
        required_initial_map_source=required_initial_map_source,
    )
    last_actionable_state = None
    replayed_actions = 0
    for record_index, record in enumerate(records):
        if record.get("direction") == "game_to_agent":
            state = dict(record.get("payload") or {})
            if state.get("in_game") and CommunicationBackend._is_actionable_state(state):
                last_actionable_state = state
            continue
        if record_index < start_index or record.get("direction") != "agent_to_game":
            continue
        command = str(record.get("command") or "")
        if command == "state" or command.startswith("wait"):
            continue
        if last_actionable_state is None:
            raise RuntimeError(f"relay command has no preceding actionable state: {command}")
        action_reference = observation_from_relay_state(parser, last_actionable_state)
        candidate = align_candidate_reward_stage(action_reference, backend, candidate)
        action = replay_action_for_command(command, action_reference, candidate)
        if (
            command.startswith("choose ")
            and action_reference.phase.value == "map"
        ):
            reference_action = next(
                (
                    candidate_action
                    for candidate_action in action_reference.legal_actions
                    if candidate_action.choice_index == int(command.split()[1])
                ),
                None,
            )
            if reference_action is not None and reference_action.target_x >= 0:
                parser._map_x = reference_action.target_x
                parser._map_y = reference_action.target_y
        if action is not None:
            candidate = backend.step(action).observation
        replayed_actions += 1

    candidate = align_candidate_reward_stage(reference, backend, candidate)

    differences = compare_observations(reference, candidate)
    if differences:
        paths = ", ".join(difference.path for difference in differences[:8])
        raise RuntimeError(
            "relay trace replay did not reconstruct the attached real state; "
            f"differences={paths}"
        )
    return backend, candidate, {
        **info,
        "resume_trace": str(trace_path),
        "replayed_actions": replayed_actions,
    }


def append_record(path: Path | None, record: dict[str, Any]) -> None:
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if path is None:
        print(f"record={serialized}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")


def latest_reference_session(trace_path: Path, seed: int) -> tuple[int, list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_start = f"start ironclad 0 {seed}"
    start_indices = [
        index
        for index, record in enumerate(records)
        if record.get("direction") == "agent_to_game"
        and record.get("command") == expected_start
    ]
    if not start_indices:
        raise RuntimeError(f"relay trace contains no session for seed {seed}: {trace_path}")
    start = start_indices[-1]
    end = next(
        (
            index
            for index in range(start + 1, len(records))
            if records[index].get("direction") == "agent_to_game"
            and str(records[index].get("command") or "").startswith("start ")
        ),
        len(records),
    )
    return start, records[start:end]


def replay_reference_trace(
    args: argparse.Namespace,
    output: Path | None,
    allowlist: DifferentialAllowlist,
) -> int:
    if args.reference_trace is None:
        raise ValueError("reference trace is required")
    session_start, records = latest_reference_session(args.reference_trace, args.seed)
    parser = CommunicationBackend()
    candidate_backend: LightspeedBackend | None = None
    candidate: Any | None = None
    candidate_info: dict[str, Any] | None = None
    reference: Any | None = None
    compared = 0
    unallowed_total = 0

    for session_offset, record in enumerate(records):
        if record.get("direction") == "game_to_agent":
            state = dict(record.get("payload") or {})
            if not state.get("in_game") or not CommunicationBackend._is_actionable_state(state):
                continue
            reference = observation_from_relay_state(parser, state)
            if candidate_backend is None:
                candidate_backend = LightspeedBackend(
                    neow_history=resolve_neow_history(args.neow_history, reference),
                    act1_boss_history=args.act1_boss_history,
                    final_act_unlocked=args.final_act_unlocked,
                )
                candidate, candidate_info = candidate_backend.reset(seed=args.seed)
            if candidate is None:
                raise RuntimeError("candidate initialization failed")
            differences = compare_observations(reference, candidate, allowlist)
            unallowed = [difference for difference in differences if not difference.allowed]
            append_record(
                output,
                {
                    "timestamp": timestamp(),
                    "seed": args.seed,
                    "step": compared,
                    "reference_info": {
                        "backend": "communication_mod_trace",
                        "trace": str(args.reference_trace.resolve()),
                        "session_start_record": session_start,
                    }
                    if compared == 0
                    else None,
                    "candidate_info": candidate_info if compared == 0 else None,
                    "reference_phase": reference.phase.value,
                    "candidate_phase": candidate.phase.value,
                    "source_record_index": session_start + session_offset,
                    "differences": [difference.to_dict() for difference in differences],
                },
            )
            print(
                f"step={compared} reference={reference.phase.value} "
                f"candidate={candidate.phase.value} differences={len(differences)} "
                f"unallowed={len(unallowed)}",
                flush=True,
            )
            compared += 1
            unallowed_total += len(unallowed)
            continue

        command = str(record.get("command") or "")
        if command.startswith("start ") or command == "state" or command.startswith("wait"):
            continue
        if reference is None or candidate_backend is None or candidate is None:
            raise RuntimeError(f"relay command has no preceding actionable state: {command}")
        if command.startswith("choose ") and reference.phase.value == "map":
            reference_action = next(
                (
                    action
                    for action in reference.legal_actions
                    if action.choice_index == int(command.split()[1])
                ),
                None,
            )
            if reference_action is not None and reference_action.target_x >= 0:
                parser._map_x = reference_action.target_x
                parser._map_y = reference_action.target_y
        action = replay_action_for_command(command, reference, candidate)
        if action is not None:
            candidate = candidate_backend.step(action).observation

    if compared == 0:
        raise RuntimeError("relay trace session contains no actionable game state")
    return 0 if unallowed_total == 0 else 1


def main() -> int:
    args = parse_args()
    output = None if args.output == "-" else Path(args.output)
    allowlist = DifferentialAllowlist.from_json(args.allowlist)
    if args.reference_trace is not None:
        return replay_reference_trace(args, output, allowlist)
    reference_backend = CommunicationBackend(
        host=args.host,
        port=args.port,
        connect_wait_timeout=args.connect_wait,
    )
    if args.resume:
        reference, reference_info = reference_backend.attach(seed=args.seed)
    else:
        reference, reference_info = reference_backend.reset(seed=args.seed)
    if args.resume:
        if args.resume_trace is not None:
            candidate_backend, candidate, candidate_info = reconstruct_candidate_from_trace(
                reference,
                args.seed,
                args.resume_trace,
                args.resume_prefix_trace,
                args.neow_history,
                args.act1_boss_history,
                args.final_act_unlocked,
            )
        else:
            candidate_backend, candidate, candidate_info = align_candidate_to_reference(
                reference,
                args.seed,
                neow_history=args.neow_history,
                act1_boss_history=args.act1_boss_history,
                final_act_unlocked=args.final_act_unlocked,
            )
    else:
        candidate_backend = LightspeedBackend(
            neow_history=resolve_neow_history(args.neow_history, reference),
            act1_boss_history=args.act1_boss_history,
            final_act_unlocked=args.final_act_unlocked,
        )
        candidate, candidate_info = candidate_backend.reset(seed=args.seed)

    for step_index in range(args.steps + 1):
        differences = compare_observations(reference, candidate, allowlist)
        append_record(
            output,
            {
                "timestamp": timestamp(),
                "seed": args.seed,
                "step": step_index,
                "reference_info": reference_info if step_index == 0 else None,
                "candidate_info": candidate_info if step_index == 0 else None,
                "reference_phase": reference.phase.value,
                "candidate_phase": candidate.phase.value,
                "differences": [difference.to_dict() for difference in differences],
            },
        )
        unallowed = [difference for difference in differences if not difference.allowed]
        print(
            f"step={step_index} reference={reference.phase.value} "
            f"candidate={candidate.phase.value} differences={len(differences)} "
            f"unallowed={len(unallowed)}",
            flush=True,
        )

        if step_index == args.steps or reference.phase.value == "terminal" or candidate.phase.value == "terminal":
            break
        paired = pair_action(reference, candidate)
        if paired is None:
            print("no_semantically_paired_action", flush=True)
            break
        reference_transition = reference_backend.step(paired[0])
        reference = reference_transition.observation
        reference_terminated = reference_transition.terminated
        candidate_terminated = False
        if paired[1] is None:
            print("reference_only_presentation_action", flush=True)
        else:
            candidate_transition = candidate_backend.step(paired[1])
            candidate = candidate_transition.observation
            candidate_terminated = candidate_transition.terminated
        if reference_terminated or candidate_terminated:
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
