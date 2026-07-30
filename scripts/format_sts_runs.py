#!/usr/bin/env python3
"""Normalize Slay the Spire run-history JSON shards.

The source shards contain one JSON array per file.  This tool emits compact
JSONL files so downstream jobs can stream records without loading the whole
corpus.  The A20 Heart rule is deliberately explicit and configurable because
the source format does not expose a dedicated ``heart_defeated`` field.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


NORMALIZED_FIELDS = (
    "run_id",
    "source_file",
    "character",
    "ascension_level",
    "is_ascension_mode",
    "is_trial",
    "is_daily",
    "is_beta",
    "is_endless",
    "victory",
    "floor_reached",
    "killed_by",
    "seed_played",
    "score",
    "playtime_seconds",
    "local_time",
    "timestamp",
    "build_version",
    "master_deck",
    "relics",
    "potions_obtained",
    "relics_obtained",
    "boss_relics",
    "path_taken",
    "path_per_floor",
    "card_choices",
    "campfire_choices",
    "event_choices",
    "damage_taken",
    "current_hp_per_floor",
    "max_hp_per_floor",
    "gold_per_floor",
    "items_purchased",
    "items_purged",
    "potions_floor_usage",
    "potions_floor_spawned",
    "heart_victory",
    "heart_detection",
    "source_fields_present",
)

SUMMARY_FEATURE_FIELDS = {
    "route": ("path_taken", "path_per_floor"),
    "rewards": ("card_choices", "relics_obtained", "potions_obtained", "boss_relics"),
    "shops": ("items_purchased",),
    "rest_sites": ("campfire_choices",),
    "events": ("event_choices",),
    "final_outcome": ("victory", "floor_reached"),
}

DECISION_LABEL_FIELDS = {
    "card_reward": ("card_choices",),
    "boss_relic": ("boss_relics",),
    "campfire": ("campfire_choices",),
    "event": ("event_choices",),
    "shop_purchase": ("items_purchased",),
    "route": ("path_taken", "path_per_floor"),
}


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield event dictionaries from one source shard."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload = payload.get("runs", payload.get("data", [payload]))
    if not isinstance(payload, list):
        raise ValueError(f"unsupported JSON root in {path}: {type(payload).__name__}")

    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            continue
        event = record.get("event", record)
        if isinstance(event, dict):
            yield event
        else:
            raise ValueError(f"record {index} in {path} has no event object")


def _is_normal_run(event: dict[str, Any]) -> bool:
    return (
        _as_bool(event.get("is_ascension_mode"))
        and not _as_bool(event.get("is_trial"))
        and not _as_bool(event.get("is_daily"))
        and not _as_bool(event.get("is_beta"))
        and not _as_bool(event.get("is_endless"))
    )


def _heart_evidence(event: dict[str, Any], heart_min_floor: int) -> tuple[bool, str]:
    """Classify A20 Heart victory without treating Heart deaths as wins.

    These shards do not contain a dedicated heart-win field.  In the supplied
    2020 format, A20 wins that reach the Heart finish at floor 57, while
    ``killed_by == 'The Heart'`` denotes a loss.  The threshold remains a CLI
    option so later shards with different floor indexing can be reprocessed.
    """

    if not _as_bool(event.get("victory")):
        return False, "victory_false"
    if event.get("killed_by"):
        return False, "killed_by_present"
    floor = _as_int(event.get("floor_reached"))
    if floor is None or floor < heart_min_floor:
        return False, "floor_below_heart_threshold"
    return True, f"victory_and_floor_reached_ge_{heart_min_floor}"


def normalize_event(event: dict[str, Any], source_file: str, heart_min_floor: int) -> dict[str, Any]:
    heart_victory, heart_detection = _heart_evidence(event, heart_min_floor)
    out: dict[str, Any] = {
        "run_id": event.get("play_id"),
        "source_file": source_file,
        "character": event.get("character_chosen"),
        "ascension_level": _as_int(event.get("ascension_level")),
        "is_ascension_mode": _as_bool(event.get("is_ascension_mode")),
        "is_trial": _as_bool(event.get("is_trial")),
        "is_daily": _as_bool(event.get("is_daily")),
        "is_beta": _as_bool(event.get("is_beta")),
        "is_endless": _as_bool(event.get("is_endless")),
        "victory": _as_bool(event.get("victory")),
        "floor_reached": _as_int(event.get("floor_reached")),
        "killed_by": event.get("killed_by"),
        "seed_played": event.get("seed_played"),
        "score": event.get("score"),
        "playtime_seconds": event.get("playtime"),
        "local_time": event.get("local_time"),
        "timestamp": event.get("timestamp"),
        "build_version": event.get("build_version"),
        "master_deck": event.get("master_deck", []),
        "relics": event.get("relics", []),
        "potions_obtained": event.get("potions_obtained", []),
        "relics_obtained": event.get("relics_obtained", []),
        "boss_relics": event.get("boss_relics", []),
        "path_taken": event.get("path_taken", []),
        "path_per_floor": event.get("path_per_floor", []),
        "card_choices": event.get("card_choices", []),
        "campfire_choices": event.get("campfire_choices", []),
        "event_choices": event.get("event_choices", []),
        "damage_taken": event.get("damage_taken", []),
        "current_hp_per_floor": event.get("current_hp_per_floor", []),
        "max_hp_per_floor": event.get("max_hp_per_floor", []),
        "gold_per_floor": event.get("gold_per_floor", []),
        "items_purchased": event.get("items_purchased", []),
        "items_purged": event.get("items_purged", []),
        "potions_floor_usage": event.get("potions_floor_usage", []),
        "potions_floor_spawned": event.get("potions_floor_spawned", []),
        "heart_victory": heart_victory,
        "heart_detection": heart_detection,
        "source_fields_present": sorted(
            {
                field
                for fields in (*SUMMARY_FEATURE_FIELDS.values(), *DECISION_LABEL_FIELDS.values())
                for field in fields
                if field in event
            }
        ),
    }
    return {key: out.get(key) for key in NORMALIZED_FIELDS}


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Slay the Spire Run Formatting Report",
        "",
        f"- Input files: {summary['input_files']}",
        f"- Input records: {summary['input_records']}",
        f"- Duplicate records skipped: {summary['duplicate_records']}",
        f"- Normalized records: {summary['normalized_records']}",
        f"- Normal A20 records: {summary['normal_a20_records']}",
        f"- Normal A20 victories: {summary['normal_a20_victories']}",
        f"- A20 Heart victories: {summary['a20_heart_victories']}",
        f"- Build version filter: {summary.get('build_version_filter') or 'all versions'}",
        "",
        "## A20 Heart Rule",
        "",
        f"`victory == true`, `killed_by` empty, and `floor_reached >= {summary['heart_min_floor']}`.",
        "The source format has no explicit heart-win field; the threshold is configurable.",
        "`killed_by = The Heart` is treated as a loss and never as a victory.",
        "",
        "## A20 Heart Victories By Character",
        "",
        "| Character | Count |",
        "|---|---:|",
    ]
    for character, count in sorted(summary["a20_heart_by_character"].items()):
        lines.append(f"| {character} | {count} |")
    lines.extend(
        [
            "",
            "## Dataset Requirement Check",
            "",
            "The requested 3,000 Heart wins and 20,000 complete A20 attempts are evaluated per character for the selected build version.",
            "",
            "| Character | A20 attempts | Attempts >= 20,000 | Heart wins | Wins >= 3,000 |",
            "|---|---:|:---:|---:|:---:|",
        ]
    )
    attempts_by_character = summary.get("a20_by_character", {})
    wins_by_character = summary.get("a20_heart_by_character", {})
    for character in sorted(set(attempts_by_character) | set(wins_by_character)):
        attempts = attempts_by_character.get(character, 0)
        wins = wins_by_character.get(character, 0)
        lines.append(
            f"| {character} | {attempts} | {'YES' if attempts >= 20000 else 'NO'} | {wins} | {'YES' if wins >= 3000 else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "### Available Summary Coverage",
            "",
            "| Feature | Source field present | Non-empty observation |",
            "|---|---:|---:|",
        ]
    )
    for feature, count in summary.get("feature_source_presence", {}).items():
        non_empty = summary.get("feature_coverage", {}).get(feature, 0)
        lines.append(f"| {feature} | {count} | {non_empty} |")
    lines.extend(
        [
            "",
            "### Low-Level Trajectory Contract",
            "",
            "The source run-history schema has no per-decision pre-state, legal-action set, or selected-action sequence. These fields are unavailable and are not fabricated by this formatter.",
        ]
    )
    lines.extend(
        [
            "",
            "## Data Limitation",
            "",
            "These run-history records contain run summaries (deck, relics, paths, rewards and outcomes), not per-step combat observations/actions. They are suitable for outcome-conditioned analysis and filtering, but not by themselves a low-level action imitation dataset.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self.handle.write("\n")
        self.count += 1

    def close(self) -> None:
        self.handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--heart-min-floor",
        type=int,
        default=57,
        help="minimum floor for the A20 Heart victory rule (default: 57)",
    )
    parser.add_argument(
        "--a20-only",
        action="store_true",
        help="emit only normal A20 and A20 Heart files; skip non-A20 output",
    )
    parser.add_argument(
        "--build-version",
        help="keep only A20 records with this exact build_version (for example 2020-07-30)",
    )
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.json"))
    if not input_files:
        raise SystemExit(f"no JSON shards found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    duplicate_records = 0
    input_records = 0

    writer_names = ["a20_runs.jsonl", "a20_heart_wins.jsonl"]
    if not args.a20_only:
        writer_names = ["runs_normalized.jsonl", "normal_runs.jsonl"] + writer_names
    writers = {name: _JsonlWriter(args.output_dir / name) for name in writer_names}
    character_writers: dict[tuple[str, str], _JsonlWriter] = {}
    character_names = ("DEFECT", "IRONCLAD", "THE_SILENT", "WATCHER")
    for label in ("a20_runs", "a20_heart_wins"):
        for character in character_names:
            name = f"{label}_{character}.jsonl"
            character_writers[(label, character)] = _JsonlWriter(args.output_dir / name)
    if not args.a20_only:
        for character in character_names:
            character_writers[("normal_runs", character)] = _JsonlWriter(
                args.output_dir / f"normal_runs_{character}.jsonl"
            )

    normal_records = 0
    normal_a20_records = 0
    normal_a20_victories = 0
    a20_heart_victories = 0
    normalized_records = 0
    a20_heart_by_character: Counter[str] = Counter()
    a20_by_character: Counter[str] = Counter()
    a20_victories_by_character: Counter[str] = Counter()
    ascension_distribution: Counter[str] = Counter()
    feature_coverage: Counter[str] = Counter()
    feature_source_presence: Counter[str] = Counter()
    decision_label_coverage: Counter[str] = Counter()

    try:
        for path in input_files:
            for event in _iter_events(path):
                input_records += 1
                run_id = event.get("play_id")
                if run_id and run_id in seen_ids:
                    duplicate_records += 1
                    continue
                if run_id:
                    seen_ids.add(run_id)
                record = normalize_event(event, path.name, args.heart_min_floor)
                normalized_records += 1
                ascension_distribution[str(record["ascension_level"])] += 1
                normal = _is_normal_run(record)
                a20 = normal and record["ascension_level"] == 20
                if not args.a20_only:
                    writers["runs_normalized.jsonl"].write(record)
                if not normal:
                    continue
                normal_records += 1
                if not args.a20_only:
                    writers["normal_runs.jsonl"].write(record)
                    character = record.get("character")
                    if character in character_names:
                        character_writers[("normal_runs", character)].write(record)
                if not a20:
                    continue
                if args.build_version and record.get("build_version") != args.build_version:
                    continue
                normal_a20_records += 1
                a20_by_character[record.get("character")] += 1
                writers["a20_runs.jsonl"].write(record)
                character = record.get("character")
                if character in character_names:
                    character_writers[("a20_runs", character)].write(record)
                for feature, fields in SUMMARY_FEATURE_FIELDS.items():
                    if any(field in event for field in fields):
                        feature_source_presence[feature] += 1
                    if any(record.get(field) not in (None, [], {}, "") for field in fields):
                        feature_coverage[feature] += 1
                for label, fields in DECISION_LABEL_FIELDS.items():
                    if any(
                        record.get(field) not in (None, [], {}, "")
                        for field in fields
                    ):
                        decision_label_coverage[label] += 1
                if record["victory"]:
                    normal_a20_victories += 1
                    a20_victories_by_character[character] += 1
                if record["heart_victory"]:
                    a20_heart_victories += 1
                    a20_heart_by_character[character] += 1
                    writers["a20_heart_wins.jsonl"].write(record)
                    if character in character_names:
                        character_writers[("a20_heart_wins", character)].write(record)
    finally:
        for writer in (*writers.values(), *character_writers.values()):
            writer.close()

    output_counts = {name: writer.count for name, writer in writers.items()}
    output_counts.update(
        {writer.path.name: writer.count for writer in character_writers.values()}
    )

    requirement_by_character = {}
    for character in sorted(set(a20_by_character) | set(a20_heart_by_character)):
        attempts = a20_by_character.get(character, 0)
        heart_wins = a20_heart_by_character.get(character, 0)
        requirement_by_character[character] = {
            "a20_attempts": attempts,
            "heart_wins": heart_wins,
            "attempts_target": 20_000,
            "heart_wins_target": 3_000,
            "attempts_target_met": attempts >= 20_000,
            "heart_wins_target_met": heart_wins >= 3_000,
        }

    summary: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_files": len(input_files),
        "input_records": input_records,
        "duplicate_records": duplicate_records,
        "normalized_records": normalized_records,
        "normal_records": normal_records,
        "normal_a20_records": normal_a20_records,
        "normal_a20_victories": normal_a20_victories,
        "a20_heart_victories": a20_heart_victories,
        "heart_min_floor": args.heart_min_floor,
        "a20_only": args.a20_only,
        "build_version_filter": args.build_version,
        "a20_heart_by_character": dict(a20_heart_by_character),
        "a20_by_character": dict(a20_by_character),
        "a20_victories_by_character": dict(a20_victories_by_character),
        "ascension_distribution": dict(ascension_distribution),
        "feature_coverage": dict(feature_coverage),
        "feature_source_presence": dict(feature_source_presence),
        "decision_label_coverage": dict(decision_label_coverage),
        "requirement_by_character": requirement_by_character,
        "trajectory_capability": {
            "pre_decision_state": False,
            "legal_action_set": False,
            "selected_action": False,
            "final_outcome": True,
            "route": feature_source_presence.get("route", 0) == normal_a20_records,
            "rewards": feature_source_presence.get("rewards", 0) == normal_a20_records,
            "shops": feature_source_presence.get("shops", 0) == normal_a20_records,
            "rest_sites": feature_source_presence.get("rest_sites", 0) == normal_a20_records,
            "events": feature_source_presence.get("events", 0) == normal_a20_records,
        },
        "output_counts": output_counts,
        "source_files": [path.name for path in input_files],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(args.output_dir / "REPORT.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
