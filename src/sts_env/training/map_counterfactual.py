from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from sts_env.env import StsEnv
from sts_env.trace import observation_digest
from sts_env.types import Action, ActionKind, Observation, Phase


PolicyFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class MapCounterfactualConfig:
    particles_per_action: int = 2
    rollout_max_steps: int = 5_000
    use_redeterminization: bool = True

    def __post_init__(self) -> None:
        if self.particles_per_action <= 0 or self.rollout_max_steps <= 0:
            raise ValueError("map counterfactual rollout limits must be positive")


@dataclass(frozen=True, slots=True)
class CounterfactualRollout:
    particle_index: int
    particle_seed: int
    final_floor: int
    final_act: int
    won: bool
    environment_return: float
    steps: int
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "particle_index": self.particle_index,
            "particle_seed": self.particle_seed,
            "final_floor": self.final_floor,
            "final_act": self.final_act,
            "won": self.won,
            "environment_return": self.environment_return,
            "steps": self.steps,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class MapCounterfactualCandidate:
    action: Action
    rollouts: tuple[CounterfactualRollout, ...]

    @property
    def mean_final_floor(self) -> float:
        return sum(rollout.final_floor for rollout in self.rollouts) / len(self.rollouts)

    @property
    def mean_return(self) -> float:
        return sum(rollout.environment_return for rollout in self.rollouts) / len(self.rollouts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "mean_final_floor": self.mean_final_floor,
            "mean_return": self.mean_return,
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
        }


@dataclass(frozen=True, slots=True)
class MapCounterfactualRecord:
    seed: int
    decision_index: int
    act: int
    floor: int
    observation_digest: str
    observation: dict[str, Any]
    behavior_action: Action
    candidates: tuple[MapCounterfactualCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "decision_index": self.decision_index,
            "act": self.act,
            "floor": self.floor,
            "observation_digest": self.observation_digest,
            "observation": self.observation,
            "behavior_action": self.behavior_action.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def map_candidate_actions(observation: Observation) -> tuple[Action, ...]:
    if observation.phase is not Phase.MAP:
        return ()
    return tuple(
        action
        for action in observation.legal_actions
        if action.kind is ActionKind.CHOOSE_MAP_NODE
    )


def evaluate_map_counterfactuals(
    environment: StsEnv,
    *,
    seed: int,
    decision_index: int,
    behavior_action: Action,
    rollout_policy_factory: PolicyFactory,
    config: MapCounterfactualConfig | None = None,
) -> MapCounterfactualRecord:
    config = config or MapCounterfactualConfig()
    observation = environment.observation
    actions = map_candidate_actions(observation)
    if len(actions) < 2:
        raise ValueError("map counterfactuals require at least two legal map actions")
    if behavior_action not in actions:
        raise ValueError("behavior action is not one of the legal map candidates")
    source_digest = observation_digest(observation)
    candidates: list[MapCounterfactualCandidate] = []
    for action_index, action in enumerate(actions):
        rollouts = tuple(
            _evaluate_map_action_particle(
                environment,
                observation,
                action,
                seed=seed,
                decision_index=decision_index,
                action_index=action_index,
                particle_index=particle_index,
                rollout_policy_factory=rollout_policy_factory,
                config=config,
            )
            for particle_index in range(config.particles_per_action)
        )
        candidates.append(MapCounterfactualCandidate(action=action, rollouts=rollouts))
    return MapCounterfactualRecord(
        seed=seed,
        decision_index=decision_index,
        act=observation.act,
        floor=observation.floor,
        observation_digest=source_digest,
        observation=observation.to_dict(),
        behavior_action=behavior_action,
        candidates=tuple(candidates),
    )


def _evaluate_map_action_particle(
    environment: StsEnv,
    observation: Observation,
    action: Action,
    *,
    seed: int,
    decision_index: int,
    action_index: int,
    particle_index: int,
    rollout_policy_factory: PolicyFactory,
    config: MapCounterfactualConfig,
) -> CounterfactualRollout:
    particle_seed = _particle_seed(seed, decision_index, action_index, particle_index)
    branch = (
        environment.redeterminized_clone(particle_seed)
        if config.use_redeterminization
        else environment.clone()
    )
    if observation_digest(branch.observation) != observation_digest(observation):
        raise RuntimeError("counterfactual clone changed the public map observation")
    if action not in branch.observation.legal_actions:
        raise RuntimeError("counterfactual clone does not preserve a legal map action")
    policy = rollout_policy_factory()
    current, reward, terminated, truncated, info = branch.step(action)
    environment_return = reward
    steps = 1
    for step_index in range(config.rollout_max_steps - 1):
        if terminated or truncated:
            break
        next_action = _select_action(policy, branch, current, step_index)
        current, reward, terminated, truncated, info = branch.step(next_action)
        environment_return += reward
        steps += 1
    else:
        raise RuntimeError("counterfactual rollout exceeded its maximum step count")
    outcome = str(info.get("outcome") or ("truncated" if truncated else "terminal"))
    return CounterfactualRollout(
        particle_index=particle_index,
        particle_seed=particle_seed,
        final_floor=current.floor,
        final_act=current.act,
        won=environment_return > 0 and not truncated,
        environment_return=environment_return,
        steps=steps,
        outcome=outcome,
    )


def _select_action(policy: Any, environment: StsEnv, observation: Observation, step_index: int) -> Action:
    if hasattr(policy, "select"):
        return policy.select(environment)
    return policy(observation, step_index)


def _particle_seed(seed: int, decision_index: int, action_index: int, particle_index: int) -> int:
    payload = f"{seed}:{decision_index}:{action_index}:{particle_index}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_map_counterfactual_record(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if int(payload.get("schema_version", -1)) != 1:
        return ("unsupported record schema version",)
    try:
        observation = Observation.from_dict(dict(payload["observation"]))
    except (KeyError, TypeError, ValueError) as error:
        return (f"invalid public observation: {error}",)
    if observation.phase is not Phase.MAP:
        errors.append("record observation is not a map decision")
    if observation_digest(observation) != payload.get("observation_digest"):
        errors.append("record observation digest differs")
    try:
        seed = int(payload["seed"])
        decision_index = int(payload["decision_index"])
    except (KeyError, TypeError, ValueError) as error:
        return tuple(errors + [f"invalid record identity: {error}"])
    if int(payload.get("act", -1)) != observation.act:
        errors.append("record act differs from observation")
    if int(payload.get("floor", -1)) != observation.floor:
        errors.append("record floor differs from observation")
    nodes = {(node.x, node.y): node for node in observation.map_nodes}
    if not nodes:
        errors.append("record observation has no public map graph")
    try:
        behavior_action = Action.from_dict(dict(payload["behavior_action"]))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid behavior action: {error}")
        behavior_action = None
    candidates = list(payload.get("candidates") or [])
    legal_actions = map_candidate_actions(observation)
    candidate_actions: list[Action] = []
    if len(candidates) != len(legal_actions) or len(candidates) < 2:
        errors.append("candidate count differs from legal map actions")
    for action_index, candidate in enumerate(candidates):
        try:
            action = Action.from_dict(dict(candidate["action"]))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"invalid candidate action {action_index}: {error}")
            continue
        candidate_actions.append(action)
        if action not in legal_actions:
            errors.append(f"candidate action {action_index} is not legal")
        if action.kind is not ActionKind.CHOOSE_MAP_NODE:
            errors.append(f"candidate action {action_index} is not a map-node choice")
        node = nodes.get((action.target_x, action.target_y))
        if node is None:
            errors.append(f"candidate action {action_index} target is absent from the map graph")
        elif action.option_type.upper() != node.symbol.upper():
            errors.append(f"candidate action {action_index} room symbol differs from the map graph")
        rollouts = list(candidate.get("rollouts") or [])
        if not rollouts:
            errors.append(f"candidate action {action_index} has no particles")
        final_floors: list[float] = []
        returns: list[float] = []
        for particle_index, rollout in enumerate(rollouts):
            expected_seed = _particle_seed(seed, decision_index, action_index, particle_index)
            try:
                recorded_particle_index = int(rollout.get("particle_index", -1))
                recorded_particle_seed = int(rollout.get("particle_seed", -1))
                final_floor = float(rollout["final_floor"])
                final_act = int(rollout["final_act"])
                environment_return = float(rollout["environment_return"])
                steps = int(rollout.get("steps", 0))
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"candidate action {action_index} has an invalid particle: {error}")
                continue
            if recorded_particle_index != particle_index:
                errors.append(f"candidate action {action_index} has a particle index gap")
            if recorded_particle_seed != expected_seed:
                errors.append(f"candidate action {action_index} has an unexpected particle seed")
            if steps <= 0:
                errors.append(f"candidate action {action_index} has a non-positive rollout length")
            if final_floor < 0 or final_act < 0 or not math.isfinite(environment_return):
                errors.append(f"candidate action {action_index} has a non-finite or invalid outcome")
            if not isinstance(rollout.get("won"), bool):
                errors.append(f"candidate action {action_index} has a non-boolean win outcome")
            if not isinstance(rollout.get("outcome"), str) or not rollout["outcome"]:
                errors.append(f"candidate action {action_index} has an empty terminal outcome")
            final_floors.append(final_floor)
            returns.append(environment_return)
        if final_floors:
            try:
                reported_floor = float(candidate["mean_final_floor"])
                reported_return = float(candidate["mean_return"])
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"candidate action {action_index} has invalid aggregate labels: {error}")
            else:
                if not math.isclose(
                    reported_floor,
                    sum(final_floors) / len(final_floors),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    errors.append(f"candidate action {action_index} mean final floor differs from particles")
                if not math.isclose(
                    reported_return,
                    sum(returns) / len(returns),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    errors.append(f"candidate action {action_index} mean return differs from particles")
    if len(set(candidate_actions)) != len(candidate_actions):
        errors.append("candidate actions are duplicated")
    if tuple(candidate_actions) != legal_actions:
        errors.append("candidate action order differs from legal map actions")
    if behavior_action is not None and behavior_action not in candidate_actions:
        errors.append("behavior action is missing from candidates")
    return tuple(errors)


def validate_map_counterfactual_corpus(root: str | Path) -> dict[str, Any]:
    destination = Path(root)
    manifest_path = destination / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "errors": [f"invalid manifest: {error}"], "records": 0}
    errors: list[str] = []
    if manifest.get("protocol") != "map-counterfactual-rollouts":
        errors.append("manifest protocol differs")
    if int(manifest.get("schema_version", -1)) != 1:
        errors.append("manifest schema version differs")
    if int(manifest.get("ascension", -1)) != 20:
        errors.append("manifest ascension differs")
    if manifest.get("neow_history") not in {"full", "limited", "skipped"}:
        errors.append("manifest Neow history is invalid")
    if manifest.get("act1_boss_history") not in {
        "guardian_unseen",
        "hexaghost_unseen",
        "slime_boss_unseen",
        "all_seen",
    }:
        errors.append("manifest Act 1 boss history is invalid")
    if not isinstance(manifest.get("final_act_unlocked"), bool):
        errors.append("manifest final-act unlock flag is invalid")
    try:
        expected_particles = int(manifest["particles_per_action"])
    except (KeyError, TypeError, ValueError):
        expected_particles = 0
    if expected_particles <= 0:
        errors.append("manifest particle count is invalid")
    try:
        seed_start, seed_end = (int(value) for value in manifest["seed_range"])
        if seed_start < 0 or seed_end < seed_start:
            raise ValueError("range is invalid")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"manifest seed range is invalid: {error}")
        seed_start, seed_end = 0, -1
    if manifest.get("errors"):
        errors.append("manifest reports collection errors")
    records_payload = dict(manifest.get("records") or {})
    records_path = destination / str(records_payload.get("path") or "")
    if not records_path.is_file():
        return {"valid": False, "errors": ["records file is missing"], "records": 0}
    expected_hash = str(records_payload.get("sha256") or "")
    actual_hash = sha256_file(records_path)
    if actual_hash != expected_hash:
        errors.append("records file hash differs from manifest")
    try:
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as error:
        return {"valid": False, "errors": errors + [f"records JSON is invalid: {error}"], "records": 0}
    if not records:
        errors.append("records file is empty")
    manifest_counts = {
        str(key): int(value) for key, value in dict(manifest.get("counts") or {}).items()
    }
    act_counts = {key: 0 for key in manifest_counts}
    identities: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        record_errors = validate_map_counterfactual_record(record)
        errors.extend(f"record {index}: {error}" for error in record_errors)
        try:
            identity = (int(record["seed"]), int(record["decision_index"]))
        except (KeyError, TypeError, ValueError):
            identity = None
        if identity is not None:
            if identity in identities:
                errors.append(f"record {index}: root decision is duplicated")
            identities.add(identity)
            if not seed_start <= identity[0] <= seed_end:
                errors.append(f"record {index}: seed is outside the manifest range")
        try:
            observation_ascension = int(record["observation"]["ascension"])
        except (KeyError, TypeError, ValueError):
            observation_ascension = -1
        if observation_ascension != 20:
            errors.append(f"record {index}: observation ascension differs")
        if expected_particles > 0:
            for candidate_index, candidate in enumerate(record.get("candidates") or []):
                if len(candidate.get("rollouts") or []) != expected_particles:
                    errors.append(
                        f"record {index}: candidate {candidate_index} particle count differs from manifest"
                    )
        act = str(record.get("act"))
        if act in act_counts:
            act_counts[act] += 1
    if act_counts != manifest_counts:
        errors.append("per-act record counts differ from manifest")
    return {
        "valid": not errors,
        "errors": errors,
        "records": len(records),
        "records_sha256": actual_hash,
        "act_counts": act_counts,
        "complete": bool(manifest.get("complete")),
    }
