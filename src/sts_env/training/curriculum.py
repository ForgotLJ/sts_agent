from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

from sts_env.env import StsEnv
from sts_env.differential import canonical_observation
from sts_env.trace import EpisodeTrace, observation_digest, replay_trace
from sts_env.types import Action, Observation, Phase


@dataclass(frozen=True, slots=True)
class CurriculumSpec:
    name: str
    start_act: int = 1
    target_act: int | None = None
    target_floor: int | None = None
    completion_reward: float = 1.0
    potential_scale: float = 1.0
    progress_reward_per_floor: float = 0.05
    discount: float = 0.997
    use_prefix_starts: bool = False
    max_episode_steps: int = 5000
    max_repeated_decisions: int = 4

    def __post_init__(self) -> None:
        if not self.name or self.start_act < 1:
            raise ValueError("curriculum name and start act are invalid")
        if self.target_act is not None and self.target_act <= self.start_act:
            raise ValueError("target act must be greater than the start act")
        if self.target_floor is not None and self.target_floor <= 0:
            raise ValueError("target floor must be positive")
        if (
            self.potential_scale < 0
            or self.progress_reward_per_floor < 0
            or not 0 <= self.discount <= 1
        ):
            raise ValueError("curriculum shaping configuration is invalid")
        if self.max_episode_steps <= 0 or self.max_repeated_decisions <= 0:
            raise ValueError("curriculum episode limits must be positive")
        if self.use_prefix_starts and self.start_act <= 1:
            raise ValueError("prefix starts are only needed after Act 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CURRICULUM = (
    CurriculumSpec("act1_floor6", target_floor=6),
    CurriculumSpec("act1_floor11", target_floor=11),
    CurriculumSpec("act1_floor15", target_floor=15),
    CurriculumSpec("act1_floor16", target_floor=16),
    CurriculumSpec("act1_clear", target_act=2),
    CurriculumSpec("act2_clear", start_act=2, target_act=3, use_prefix_starts=True),
    CurriculumSpec("act3_clear", start_act=3, use_prefix_starts=True),
    CurriculumSpec(
        "full_run",
        completion_reward=0.0,
        potential_scale=0.0,
        progress_reward_per_floor=0.0,
    ),
)


@dataclass(frozen=True, slots=True)
class PrefixCorpus:
    traces: tuple[EpisodeTrace, ...]
    target_act: int

    def __post_init__(self) -> None:
        if self.target_act < 2:
            raise ValueError("prefix corpus target act must be at least 2")
        if not self.traces:
            raise ValueError("prefix corpus must contain at least one trace")

    def write(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        for index, trace in enumerate(self.traces):
            filename = f"act-{self.target_act}-prefix-{index:06d}.jsonl"
            trace.write_jsonl(destination / filename)
            files.append(filename)
        manifest = {
            "schema_version": 1,
            "target_act": self.target_act,
            "files": files,
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, directory: str | Path) -> PrefixCorpus:
        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            traces=tuple(
                EpisodeTrace.read_jsonl(source / filename)
                for filename in manifest["files"]
            ),
            target_act=int(manifest["target_act"]),
        )

    @classmethod
    def extract(
        cls,
        environment_factory: Callable[[], StsEnv],
        traces: tuple[EpisodeTrace, ...],
        target_act: int,
    ) -> PrefixCorpus:
        prefixes: list[EpisodeTrace] = []
        for supplied_trace in traces:
            trace = materialize_recovery_trace(supplied_trace)
            environment = environment_factory()
            observation, _ = environment.reset(seed=trace.seed)
            if observation.act >= target_act:
                prefixes.append(trace.prefix(0))
                continue
            for step_index, step in enumerate(trace.steps):
                observation, _, _, _, _ = environment.step(step.action)
                if observation.act >= target_act:
                    prefixes.append(trace.prefix(step_index + 1))
                    break
        if not prefixes:
            raise ValueError(f"no supplied trace reaches Act {target_act}")
        return cls(tuple(prefixes), target_act)


class CurriculumEnvironment:
    def __init__(
        self,
        environment_factory: Callable[[], StsEnv],
        spec: CurriculumSpec,
        prefix_corpus: PrefixCorpus | None = None,
    ):
        if spec.use_prefix_starts:
            if prefix_corpus is None or prefix_corpus.target_act != spec.start_act:
                raise ValueError("curriculum stage requires a matching prefix corpus")
        self._environment_factory = environment_factory
        self._environment = environment_factory()
        self.spec = spec
        self.prefix_corpus = prefix_corpus
        self._selection_seed: int | None = None
        self._steps = 0
        self._decision_counts: dict[str, int] = {}

    @property
    def observation(self) -> Observation:
        return self._environment.observation

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        if seed is None:
            raise ValueError("curriculum environments require an explicit selection seed")
        self._selection_seed = seed
        self._steps = 0
        self._decision_counts = {}
        if self.spec.use_prefix_starts:
            assert self.prefix_corpus is not None
            trace = self.prefix_corpus.traces[seed % len(self.prefix_corpus.traces)]
            observation = (
                self.replay_recovery_trace(trace)
                if (trace.metadata or {}).get("curriculum_source_trace")
                else replay_trace(self._environment, trace)
            )
            self._steps = 0
            self._decision_counts = {}
            public_digest = observation_digest(observation)
            self._environment = self._environment.redeterminized_clone(seed)
            observation = self._environment.observation
            if observation_digest(observation) != public_digest:
                raise AssertionError("curriculum redeterminization changed the public prefix state")
            info = {
                "backend": trace.backend,
                "seed": trace.seed,
                "curriculum_selection_seed": seed,
                "curriculum_redeterminization_seed": seed,
                "curriculum_prefix_steps": len(trace.steps),
                "curriculum_source_trace": trace.to_dict(),
            }
        else:
            observation, info = self._environment.reset(seed=seed)
        if observation.phase is Phase.TERMINAL or observation.act < self.spec.start_act:
            raise RuntimeError("curriculum reset did not produce a valid start state")
        return observation, {**info, "curriculum_stage": self.spec.name}

    def step(
        self,
        action: int | Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        previous = self.observation
        resolved_action = previous.legal_actions[action] if isinstance(action, int) else action
        decision_key = self._decision_key(previous, resolved_action)
        repeat_count = self._decision_counts.get(decision_key, 0) + 1
        self._decision_counts[decision_key] = repeat_count
        observation, raw_reward, terminated, truncated, info = self._environment.step(
            resolved_action
        )
        self._steps += 1
        completed = not terminated and not truncated and self._stage_completed(observation)
        timed_out = (
            not terminated
            and not truncated
            and not completed
            and self._steps >= self.spec.max_episode_steps
        )
        loop_detected = (
            not terminated
            and not truncated
            and not completed
            and not timed_out
            and repeat_count > self.spec.max_repeated_decisions
        )
        shaped_reward = raw_reward
        progress_reward = self.spec.progress_reward_per_floor * max(
            0,
            observation.floor - previous.floor,
        )
        shaped_reward += progress_reward
        if self.spec.potential_scale > 0:
            next_potential = (
                0.0
                if terminated or truncated or completed or timed_out or loop_detected
                else self._potential(observation)
            )
            shaped_reward += self.spec.potential_scale * (
                self.spec.discount * next_potential - self._potential(previous)
            )
        if completed:
            shaped_reward += self.spec.completion_reward
            truncated = True
        if timed_out:
            truncated = True
        if loop_detected:
            truncated = True
        return observation, shaped_reward, terminated, truncated, {
            **info,
            "raw_reward": raw_reward,
            "curriculum_stage": self.spec.name,
            "curriculum_completed": completed,
            "curriculum_timeout": timed_out,
            "curriculum_loop_detected": loop_detected,
            "curriculum_repeat_count": repeat_count,
            "curriculum_progress_reward": progress_reward,
        }

    def replay_recovery_trace(self, trace: EpisodeTrace) -> Observation:
        self._decision_counts = {}
        observation = self._replay_nested_trace(trace, count_decisions=True)
        self._selection_seed = int(
            (trace.metadata or {}).get("curriculum_selection_seed", trace.seed)
        )
        self._steps = len(trace.steps)
        return observation

    def _replay_nested_trace(
        self,
        trace: EpisodeTrace,
        *,
        count_decisions: bool,
    ) -> Observation:
        metadata = dict(trace.metadata or {})
        source_payload = dict(metadata.get("curriculum_source_trace") or {})
        if source_payload:
            source = EpisodeTrace.from_dict(source_payload)
            observation = self._replay_nested_trace(source, count_decisions=False)
            redeterminization_seed = int(
                metadata.get(
                    "curriculum_redeterminization_seed",
                    metadata.get("curriculum_selection_seed", trace.seed),
                )
            )
            public_digest = observation_digest(observation)
            self._environment = self._environment.redeterminized_clone(
                redeterminization_seed
            )
            observation = self._environment.observation
            if observation_digest(observation) != public_digest:
                raise AssertionError(
                    "curriculum recovery redeterminization changed public state"
                )
        else:
            observation, _ = self._environment.reset(seed=trace.seed)
        if observation_digest(observation) != trace.initial_observation_digest:
            raise AssertionError("curriculum recovery initial observation differs")
        for step_index, expected in enumerate(trace.steps):
            if count_decisions:
                key = self._decision_key(observation, expected.action)
                self._decision_counts[key] = self._decision_counts.get(key, 0) + 1
            observation, reward, terminated, truncated, info = self._environment.step(
                expected.action
            )
            actual = (
                observation_digest(observation),
                reward,
                terminated,
                truncated,
                info,
            )
            recorded = (
                expected.observation_digest,
                expected.reward,
                expected.terminated,
                expected.truncated,
                expected.info,
            )
            if actual != recorded:
                raise AssertionError(f"curriculum recovery diverged at step {step_index}")
        return observation

    def clone(self) -> CurriculumEnvironment:
        cloned = object.__new__(type(self))
        cloned._environment_factory = self._environment_factory
        cloned._environment = self._environment.clone()
        cloned.spec = self.spec
        cloned.prefix_corpus = self.prefix_corpus
        cloned._selection_seed = self._selection_seed
        cloned._steps = self._steps
        cloned._decision_counts = dict(self._decision_counts)
        return cloned

    def redeterminized_clone(
        self,
        search_seed: int,
        known_top: tuple[str, ...] = (),
        known_bottom: tuple[str, ...] = (),
    ) -> CurriculumEnvironment:
        cloned = object.__new__(type(self))
        cloned._environment_factory = self._environment_factory
        cloned._environment = self._environment.redeterminized_clone(
            search_seed,
            known_top=known_top,
            known_bottom=known_bottom,
        )
        cloned.spec = self.spec
        cloned.prefix_corpus = self.prefix_corpus
        cloned._selection_seed = self._selection_seed
        cloned._steps = self._steps
        cloned._decision_counts = dict(self._decision_counts)
        return cloned

    def _stage_completed(self, observation: Observation) -> bool:
        if self.spec.target_act is not None and observation.act >= self.spec.target_act:
            return True
        return self.spec.target_floor is not None and observation.floor >= self.spec.target_floor

    @staticmethod
    def _decision_key(observation: Observation, action: Action) -> str:
        return json.dumps(
            {
                "observation": canonical_observation(observation),
                "action": action.to_dict(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _potential(observation: Observation) -> float:
        if observation.phase is Phase.TERMINAL:
            return 0.0
        hp_ratio = observation.player.hp / max(1, observation.player.max_hp)
        return observation.floor / 60.0 + 0.1 * hp_ratio


def materialize_recovery_trace(trace: EpisodeTrace) -> EpisodeTrace:
    source_payload = dict((trace.metadata or {}).get("curriculum_source_trace") or {})
    if not source_payload:
        return trace
    source = EpisodeTrace.from_dict(source_payload)
    return EpisodeTrace(
        seed=source.seed,
        initial_observation_digest=source.initial_observation_digest,
        steps=(*source.steps, *trace.steps),
        backend=source.backend,
        metadata=dict(source.metadata or {}),
    )


@dataclass(slots=True)
class CurriculumScheduler:
    stages: tuple[CurriculumSpec, ...] = DEFAULT_CURRICULUM
    promotion_threshold: float = 0.25
    stage_index: int = 0

    def __post_init__(self) -> None:
        if not self.stages or not 0 < self.promotion_threshold <= 1:
            raise ValueError("curriculum scheduler configuration is invalid")
        if self.stage_index < 0 or self.stage_index >= len(self.stages):
            raise ValueError("curriculum stage index is invalid")

    @property
    def current(self) -> CurriculumSpec:
        return self.stages[self.stage_index]

    def observe_validation(self, completion_rate: float) -> bool:
        if not 0 <= completion_rate <= 1:
            raise ValueError("completion rate must be in [0, 1]")
        if completion_rate < self.promotion_threshold or self.stage_index == len(self.stages) - 1:
            return False
        self.stage_index += 1
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "promotion_threshold": self.promotion_threshold,
            "stage_index": self.stage_index,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> CurriculumScheduler:
        return cls(
            stages=tuple(CurriculumSpec(**stage) for stage in payload["stages"]),
            promotion_threshold=float(payload["promotion_threshold"]),
            stage_index=int(payload["stage_index"]),
        )
