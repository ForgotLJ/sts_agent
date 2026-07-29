from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


M7C_PROTOCOL = "m7c-dagger"
M7C_TEACHER_ANCHOR_MAX_STEPS = 5_000
M7C_TEACHER_ANCHOR_ALLOW_HORIZON_TRUNCATION = True
M7C_FROZEN_TEACHER_SEED_START = 400_000
M7C_FROZEN_TEACHER_SEED_COUNT = 4_096
M7C_FROZEN_TEACHER_SHA256 = (
    "0dfdd54bccc66b6c16b2f4515fa160ecc46752f21dc1022d1032c57b026fdd14"
)
M7C_INITIAL_CHECKPOINT_PROTOCOL = "m7b"
M7C_INITIAL_CHECKPOINT_RUN_SEED = 17
M7C_INITIAL_CHECKPOINT_SHA256 = (
    "ca6b91ac701306c2dca5aa4e1eef217691c41f515df2b6b250a99d1f2b728383"
)
M7C_M6_BASELINE_CHECKPOINT_PROTOCOL = "m6"
M7C_M6_BASELINE_CHECKPOINT_RUN_SEED = 17
M7C_M6_BASELINE_CHECKPOINT_SHA256 = (
    "51b2e02e87af9753c9f2b0eba8a731733426d51c78cdc6bcc23114a8bdae83d5"
)


@dataclass(frozen=True, slots=True)
class SeedRange:
    name: str
    start: int
    count: int
    purpose: str
    locked: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.purpose or self.start < 0 or self.count <= 0:
            raise ValueError("seed range requires a name, purpose, and positive bounds")

    @property
    def stop(self) -> int:
        return self.start + self.count

    @property
    def end(self) -> int:
        return self.stop - 1

    @property
    def values(self) -> range:
        return range(self.start, self.stop)

    def overlaps(self, other: SeedRange) -> bool:
        return max(self.start, other.start) < min(self.stop, other.stop)

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "count": self.count,
            "purpose": self.purpose,
            "locked": self.locked,
        }


M7C_HISTORICAL_RANGES = (
    SeedRange("m6_training_stream", 0, 1_000_000, "M6/M7 training stream", True),
    SeedRange("m6_validation", 1_100_000, 2_048, "M6 validation", True),
    SeedRange("m7_promotion", 1_200_000, 128, "M7 promotion", True),
    SeedRange("m7_screening", 1_210_000, 1_024, "M7 screening", True),
    SeedRange("m7_selection", 1_300_000, 512, "M7 checkpoint selection", True),
    SeedRange("m7_pilot_gate", 1_400_000, 512, "M7 pilot gate", True),
    SeedRange(
        "m7b_training_teacher",
        M7C_FROZEN_TEACHER_SEED_START,
        M7C_FROZEN_TEACHER_SEED_COUNT,
        "M7-B teacher corpus",
        True,
    ),
    SeedRange("m7b_teacher_validation", 1_500_000, 512, "M7-B validation", True),
    SeedRange("m7b_end_to_end_gate", 1_600_000, 512, "M7-B paired gate", True),
    SeedRange("m6_revealed_final", 2_000_000, 1_024, "revealed M6 final", True),
    SeedRange("m7_final_blind", 3_000_000, 2_048, "M7 final blind test", True),
)

M7C_TRAINING_ROUNDS = (
    SeedRange("dagger_round_0", 2_200_000, 1_024, "M7-C student-on-policy collection"),
    SeedRange("dagger_round_1", 2_201_024, 1_024, "M7-C student-on-policy collection"),
    SeedRange("dagger_round_2", 2_202_048, 1_024, "M7-C student-on-policy collection"),
)

M7C_VALIDATION_RANGES = (
    SeedRange("teacher_anchor", 2_210_000, 512, "teacher-state anchor validation"),
    SeedRange("on_policy_round_0", 2_211_000, 512, "round-0 on-policy validation"),
    SeedRange("on_policy_round_1", 2_212_000, 512, "round-1 on-policy validation"),
    SeedRange("on_policy_round_2", 2_213_000, 512, "round-2 on-policy validation"),
)

M7C_EVALUATION_RANGES = (
    SeedRange("promotion", 2_220_000, 512, "M7-C promotion evaluation", True),
    SeedRange("formal_gate", 2_221_000, 512, "M7-C formal paired gate", True),
)

M7C_DEVELOPMENT_RANGES = (
    SeedRange("m7c_smoke", 2_230_000, 8, "engineering-only smoke collection"),
    SeedRange(
        "m7c_smoke_validation",
        2_230_016,
        8,
        "engineering-only smoke validation",
    ),
)


def m7c_seed_registry() -> dict[str, SeedRange]:
    ranges = (
        *M7C_HISTORICAL_RANGES,
        *M7C_TRAINING_ROUNDS,
        *M7C_VALIDATION_RANGES,
        *M7C_EVALUATION_RANGES,
        *M7C_DEVELOPMENT_RANGES,
    )
    registry = {seed_range.name: seed_range for seed_range in ranges}
    if len(registry) != len(ranges):
        raise ValueError("M7-C seed registry contains duplicate names")
    validate_m7c_registry()
    return registry


def validate_seed_ranges(ranges: Iterable[SeedRange]) -> None:
    ordered = tuple(ranges)
    names = [seed_range.name for seed_range in ordered]
    if len(names) != len(set(names)):
        raise ValueError("seed range names must be unique")
    for index, seed_range in enumerate(ordered):
        for other in ordered[index + 1 :]:
            if seed_range.overlaps(other):
                raise ValueError(
                    f"seed ranges overlap: {seed_range.name} and {other.name}"
                )


def validate_m7c_registry() -> None:
    registered = (
        *M7C_HISTORICAL_RANGES,
        *M7C_TRAINING_ROUNDS,
        *M7C_VALIDATION_RANGES,
        *M7C_EVALUATION_RANGES,
        *M7C_DEVELOPMENT_RANGES,
    )
    names = [seed_range.name for seed_range in registered]
    if len(names) != len(set(names)):
        raise ValueError("M7-C seed registry contains duplicate names")
    m7c_ranges = (
        *M7C_TRAINING_ROUNDS,
        *M7C_VALIDATION_RANGES,
        *M7C_EVALUATION_RANGES,
        *M7C_DEVELOPMENT_RANGES,
    )
    validate_seed_ranges(m7c_ranges)
    for m7c_range in m7c_ranges:
        for historical_range in M7C_HISTORICAL_RANGES:
            if m7c_range.overlaps(historical_range):
                raise ValueError(
                    "M7-C range overlaps historical range: "
                    f"{m7c_range.name} and {historical_range.name}"
                )


def require_registered_seed_range(
    name: str,
    *,
    start: int,
    count: int,
    registry: Mapping[str, SeedRange] | None = None,
) -> SeedRange:
    registered = dict(registry or m7c_seed_registry())
    if name not in registered:
        raise ValueError(f"M7-C seed range is not registered: {name}")
    seed_range = registered[name]
    if (seed_range.start, seed_range.count) != (start, count):
        raise ValueError(
            f"M7-C seed range differs from the pre-registered protocol: {name}"
        )
    return seed_range


def m7c_seed_registry_payload() -> dict[str, object]:
    registry = m7c_seed_registry()
    return {
        "protocol": M7C_PROTOCOL,
        "schema_version": 1,
        "ranges": [seed_range.to_dict() for seed_range in registry.values()],
    }


def m7c_frozen_inputs_identity() -> dict[str, object]:
    return {
        "teacher_corpus": {
            "seed_start": M7C_FROZEN_TEACHER_SEED_START,
            "seed_count": M7C_FROZEN_TEACHER_SEED_COUNT,
            "aggregate_sha256": M7C_FROZEN_TEACHER_SHA256,
        },
        "initial_checkpoint": {
            "protocol": M7C_INITIAL_CHECKPOINT_PROTOCOL,
            "run_seed": M7C_INITIAL_CHECKPOINT_RUN_SEED,
            "sha256": M7C_INITIAL_CHECKPOINT_SHA256,
        },
        "m6_baseline_checkpoint": {
            "protocol": M7C_M6_BASELINE_CHECKPOINT_PROTOCOL,
            "run_seed": M7C_M6_BASELINE_CHECKPOINT_RUN_SEED,
            "sha256": M7C_M6_BASELINE_CHECKPOINT_SHA256,
        },
    }
