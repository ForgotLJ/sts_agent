from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Self

from sts_env.env import StsEnv
from sts_env.types import Action, Observation


def observation_digest(observation: Observation) -> str:
    encoded = json.dumps(
        observation.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TraceStep:
    action: Action
    observation_digest: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "step",
            "action": self.action.to_dict(),
            "observation_digest": self.observation_digest,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceStep:
        return cls(
            action=Action.from_dict(payload["action"]),
            observation_digest=str(payload["observation_digest"]),
            reward=float(payload["reward"]),
            terminated=bool(payload["terminated"]),
            truncated=bool(payload["truncated"]),
            info=dict(payload["info"]),
        )


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    seed: int
    initial_observation_digest: str
    steps: tuple[TraceStep, ...]
    backend: str = "sts_lightspeed"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "initial_observation_digest": self.initial_observation_digest,
            "steps": [step.to_dict() for step in self.steps],
            "backend": self.backend,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeTrace:
        return cls(
            seed=int(payload["seed"]),
            initial_observation_digest=str(payload["initial_observation_digest"]),
            steps=tuple(TraceStep.from_dict(step) for step in payload.get("steps", ())),
            backend=str(payload.get("backend", "unknown")),
            metadata=dict(payload.get("metadata") or {}),
        )

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            header = {
                "type": "reset",
                "backend": self.backend,
                "seed": self.seed,
                "initial_observation_digest": self.initial_observation_digest,
                "schema_version": 2,
                "metadata": self.metadata or {},
            }
            stream.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
            for step in self.steps:
                stream.write(json.dumps(step.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path) -> EpisodeTrace:
        with Path(path).open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if not records or records[0].get("type") != "reset":
            raise ValueError("trace must start with a reset record")
        header = records[0]
        return cls(
            seed=int(header["seed"]),
            initial_observation_digest=str(header["initial_observation_digest"]),
            steps=tuple(TraceStep.from_dict(record) for record in records[1:]),
            backend=str(header.get("backend", "unknown")),
            metadata=dict(header.get("metadata") or {}),
        )

    def prefix(self, step_count: int) -> EpisodeTrace:
        if step_count < 0 or step_count > len(self.steps):
            raise ValueError("step_count must be within the recorded trace")
        return EpisodeTrace(
            seed=self.seed,
            initial_observation_digest=self.initial_observation_digest,
            steps=self.steps[:step_count],
            backend=self.backend,
            metadata=dict(self.metadata or {}),
        )


class RunJournal:
    def __init__(
        self,
        environment: StsEnv,
        trace: EpisodeTrace,
        path: Path,
        stream: Any,
        durable: bool,
    ):
        self.environment = environment
        self._trace = trace
        self.path = path
        self._stream = stream
        self._durable = durable

    @property
    def trace(self) -> EpisodeTrace:
        return self._trace

    @classmethod
    def start(
        cls,
        environment: StsEnv,
        path: str | Path,
        seed: int,
        durable: bool = True,
    ) -> RunJournal:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        observation, info = environment.reset(seed=seed)
        trace = EpisodeTrace(
            seed=seed,
            initial_observation_digest=observation_digest(observation),
            steps=(),
            backend=str(info.get("backend", "unknown")),
            metadata=dict(info),
        )
        stream = destination.open("w", encoding="utf-8", newline="\n")
        journal = cls(environment, trace, destination, stream, durable)
        journal._write_record(journal._header_record())
        return journal

    @classmethod
    def recover(
        cls,
        environment: StsEnv,
        path: str | Path,
        durable: bool = True,
    ) -> RunJournal:
        destination = Path(path)
        trace, normalized_records = cls._read_recoverable_trace(destination)
        replay_trace(environment, trace)
        cls._rewrite_valid_prefix(destination, normalized_records, durable)
        stream = destination.open("a", encoding="utf-8", newline="\n")
        return cls(environment, trace, destination, stream, durable)

    def step(
        self,
        action: int | Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        resolved_action = (
            self.environment.observation.legal_actions[action]
            if isinstance(action, int)
            else action
        )
        observation, reward, terminated, truncated, info = self.environment.step(
            resolved_action
        )
        trace_step = TraceStep(
            action=resolved_action,
            observation_digest=observation_digest(observation),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self._trace = EpisodeTrace(
            seed=self._trace.seed,
            initial_observation_digest=self._trace.initial_observation_digest,
            steps=(*self._trace.steps, trace_step),
            backend=self._trace.backend,
            metadata=self._trace.metadata,
        )
        self._write_record(trace_step.to_dict())
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _header_record(self) -> dict[str, Any]:
        return {
            "type": "reset",
            "backend": self._trace.backend,
            "seed": self._trace.seed,
            "initial_observation_digest": self._trace.initial_observation_digest,
            "schema_version": 2,
            "metadata": self._trace.metadata or {},
        }

    def _write_record(self, record: dict[str, Any]) -> None:
        self._stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._stream.flush()
        if self._durable:
            os.fsync(self._stream.fileno())

    @staticmethod
    def _read_recoverable_trace(path: Path) -> tuple[EpisodeTrace, list[dict[str, Any]]]:
        raw_lines = path.read_bytes().splitlines(keepends=True)
        records: list[dict[str, Any]] = []
        for line_index, raw_line in enumerate(raw_lines):
            if not raw_line.strip():
                continue
            try:
                records.append(json.loads(raw_line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if line_index != len(raw_lines) - 1 or raw_line.endswith((b"\n", b"\r")):
                    raise ValueError(f"journal contains a corrupt record at line {line_index + 1}")
                break
        if not records or records[0].get("type") != "reset":
            raise ValueError("journal must start with a reset record")
        header = records[0]
        trace = EpisodeTrace(
            seed=int(header["seed"]),
            initial_observation_digest=str(header["initial_observation_digest"]),
            steps=tuple(TraceStep.from_dict(record) for record in records[1:]),
            backend=str(header.get("backend", "unknown")),
            metadata=dict(header.get("metadata") or {}),
        )
        return trace, records

    @staticmethod
    def _rewrite_valid_prefix(
        path: Path,
        records: list[dict[str, Any]],
        durable: bool,
    ) -> None:
        temporary = path.with_name(path.name + ".recovery.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        temporary.replace(path)


def record_episode(
    environment: StsEnv,
    seed: int,
    policy: Callable[[Observation], Action],
    max_steps: int = 20_000,
) -> EpisodeTrace:
    observation, info = environment.reset(seed=seed)
    initial_digest = observation_digest(observation)
    steps: list[TraceStep] = []

    for _ in range(max_steps):
        action = policy(observation)
        observation, reward, terminated, truncated, info = environment.step(action)
        steps.append(
            TraceStep(
                action=action,
                observation_digest=observation_digest(observation),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        if terminated or truncated:
            return EpisodeTrace(
                seed=seed,
                initial_observation_digest=initial_digest,
                steps=tuple(steps),
                backend=str(info.get("backend", "unknown")),
                metadata=dict(info),
            )
    raise RuntimeError(f"episode did not terminate within {max_steps} steps")


def replay_trace(environment: StsEnv, trace: EpisodeTrace) -> Observation:
    observation, _ = environment.reset(seed=trace.seed)
    actual_initial_digest = observation_digest(observation)
    if actual_initial_digest != trace.initial_observation_digest:
        raise AssertionError(
            "initial observation digest differs: "
            f"{actual_initial_digest} != {trace.initial_observation_digest}"
        )

    for step_index, expected in enumerate(trace.steps):
        if expected.action not in observation.legal_actions:
            raise AssertionError(f"recorded action is not legal at step {step_index}")
        observation, reward, terminated, truncated, info = environment.step(expected.action)
        actual_digest = observation_digest(observation)
        actual_transition = (actual_digest, reward, terminated, truncated, info)
        expected_transition = (
            expected.observation_digest,
            expected.reward,
            expected.terminated,
            expected.truncated,
            expected.info,
        )
        if actual_transition != expected_transition:
            raise AssertionError(f"trace diverged at step {step_index}")
    return observation
