from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


MAP_ACTION_PROTOCOL = "a20-map-action-value"
P80_CARD_OVERRIDE_MARGIN = 0.016514360904693604


@dataclass(frozen=True, slots=True)
class MapActionSeedRange:
    name: str
    start: int
    count: int
    purpose: str
    locked: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.purpose or self.start < 0 or self.count <= 0:
            raise ValueError("map-action seed range requires a name and positive bounds")

    @property
    def end(self) -> int:
        return self.start + self.count - 1

    def overlaps(self, other: MapActionSeedRange) -> bool:
        return max(self.start, other.start) <= min(self.end, other.end)

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "count": self.count,
            "purpose": self.purpose,
            "locked": self.locked,
        }


MAP_ACTION_HISTORICAL_RANGES = (
    MapActionSeedRange("card_clone_smoke", 2_302_000, 32, "clone-value smoke", True),
    MapActionSeedRange("card_clone_formal", 2_303_000, 512, "clone-value formal", True),
    MapActionSeedRange("card_clone_telemetry", 2_304_000, 64, "clone-value telemetry", True),
    MapActionSeedRange("card_clone_profile", 2_305_000, 128, "clone-value profile", True),
    MapActionSeedRange("card_clone_p80_smoke", 2_306_000, 32, "clone-value p80 smoke", True),
    MapActionSeedRange("card_clone_p80_formal", 2_307_000, 512, "clone-value p80 formal", True),
    MapActionSeedRange("card_clone_p80_replication", 2_308_000, 512, "clone-value replication", True),
)

MAP_ACTION_RANGES = (
    MapActionSeedRange("map_act1_pilot", 2_310_000, 64, "Act 1 corpus pilot"),
    MapActionSeedRange(
        "map_counterfactual_collection",
        2_312_000,
        4_096,
        "A20 map counterfactual corpus collection",
    ),
    MapActionSeedRange("map_value_profile", 2_318_000, 128, "map margin profile"),
    MapActionSeedRange("map_value_smoke", 2_319_000, 32, "map policy smoke"),
    MapActionSeedRange("map_value_formal", 2_320_000, 512, "map policy formal evaluation"),
    MapActionSeedRange(
        "map_value_replication",
        2_321_000,
        512,
        "map policy independent replication",
    ),
    MapActionSeedRange(
        "map_act1_collection_v2",
        2_322_000,
        4_096,
        "Act 1-only A20 map counterfactual corpus collection",
    ),
    MapActionSeedRange("map_act1_value_profile_v2", 2_328_000, 128, "Act 1 map margin profile"),
    MapActionSeedRange("map_act1_value_smoke_v2", 2_329_000, 32, "Act 1 map policy smoke"),
    MapActionSeedRange("map_act1_value_formal_v2", 2_330_000, 512, "Act 1 map policy formal evaluation"),
    MapActionSeedRange(
        "map_act1_value_replication_v2",
        2_331_000,
        512,
        "Act 1 map policy independent replication",
    ),
    MapActionSeedRange("map_act1_value_profile_v4", 2_332_000, 128, "Act 1 floor-gated map margin profile"),
    MapActionSeedRange("map_act1_value_smoke_v4", 2_333_000, 32, "Act 1 floor-gated map policy smoke"),
    MapActionSeedRange("map_act1_value_formal_v4", 2_334_000, 512, "Act 1 floor-gated map policy formal evaluation"),
    MapActionSeedRange(
        "map_act1_value_replication_v4",
        2_335_000,
        512,
        "Act 1 floor-gated map policy independent replication",
    ),
    MapActionSeedRange(
        "map_act1_collection_v5",
        2_340_000,
        8_192,
        "Act 1 behavior-relative map counterfactual corpus collection",
    ),
    MapActionSeedRange("map_act1_value_profile_v5", 2_350_000, 128, "Act 1 advantage-label map margin profile"),
    MapActionSeedRange("map_act1_value_smoke_v5", 2_351_000, 32, "Act 1 advantage-label map policy smoke"),
    MapActionSeedRange("map_act1_value_formal_v5", 2_352_000, 512, "Act 1 advantage-label map policy formal evaluation"),
    MapActionSeedRange(
        "map_act1_value_replication_v5",
        2_353_000,
        512,
        "Act 1 advantage-label map policy independent replication",
    ),
    MapActionSeedRange(
        "map_act1_value_profile_v6",
        2_360_000,
        512,
        "Act 1 calibrated-margin map profile",
    ),
    MapActionSeedRange(
        "map_act1_value_smoke_v6",
        2_361_000,
        64,
        "Act 1 calibrated-margin map policy smoke",
    ),
    MapActionSeedRange(
        "map_act1_value_formal_v6",
        2_362_000,
        512,
        "Act 1 calibrated-margin map policy formal evaluation",
    ),
    MapActionSeedRange(
        "map_act1_value_replication_v6",
        2_363_000,
        512,
        "Act 1 calibrated-margin map policy independent replication",
    ),
)

MAP_ACTION_COLLECTION_RANGE_NAMES = frozenset(
    {
        "map_act1_pilot",
        "map_counterfactual_collection",
        "map_act1_collection_v2",
        "map_act1_collection_v5",
    }
)
MAP_ACTION_EVALUATION_RANGE_NAMES = frozenset(
    {
        "map_value_profile",
        "map_value_smoke",
        "map_value_formal",
        "map_value_replication",
        "map_act1_value_profile_v2",
        "map_act1_value_smoke_v2",
        "map_act1_value_formal_v2",
        "map_act1_value_replication_v2",
        "map_act1_value_profile_v4",
        "map_act1_value_smoke_v4",
        "map_act1_value_formal_v4",
        "map_act1_value_replication_v4",
        "map_act1_value_profile_v5",
        "map_act1_value_smoke_v5",
        "map_act1_value_formal_v5",
        "map_act1_value_replication_v5",
        "map_act1_value_profile_v6",
        "map_act1_value_smoke_v6",
        "map_act1_value_formal_v6",
        "map_act1_value_replication_v6",
    }
)


def map_action_seed_registry() -> dict[str, MapActionSeedRange]:
    ranges = (*MAP_ACTION_HISTORICAL_RANGES, *MAP_ACTION_RANGES)
    validate_map_action_seed_ranges(ranges)
    return {seed_range.name: seed_range for seed_range in ranges}


def validate_map_action_seed_ranges(ranges: Iterable[MapActionSeedRange]) -> None:
    values = tuple(ranges)
    names = [seed_range.name for seed_range in values]
    if len(names) != len(set(names)):
        raise ValueError("map-action seed range names must be unique")
    for index, seed_range in enumerate(values):
        for other in values[index + 1 :]:
            if seed_range.overlaps(other):
                raise ValueError(
                    f"map-action seed ranges overlap: {seed_range.name} and {other.name}"
                )


def require_map_action_seed_range(
    name: str,
    *,
    start: int,
    count: int,
    allowed_names: frozenset[str],
    registry: Mapping[str, MapActionSeedRange] | None = None,
) -> MapActionSeedRange:
    if name not in allowed_names:
        raise ValueError(f"map-action seed range is not valid for this command: {name}")
    registered = dict(registry or map_action_seed_registry())
    if name not in registered:
        raise ValueError(f"map-action seed range is not registered: {name}")
    seed_range = registered[name]
    if (seed_range.start, seed_range.count) != (start, count):
        raise ValueError(f"map-action seed range differs from the frozen protocol: {name}")
    return seed_range
