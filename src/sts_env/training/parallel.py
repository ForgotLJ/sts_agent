from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as multiprocessing
from multiprocessing.connection import Connection
import traceback
from typing import Any, Callable

from sts_env import LightspeedBackend, StsEnv
from sts_env.trace import EpisodeTrace, replay_trace
from sts_env.training.curriculum import CurriculumEnvironment, CurriculumSpec, PrefixCorpus
from sts_env.types import Action, Observation


@dataclass(frozen=True, slots=True)
class LightspeedEnvironmentFactory:
    ascension: int = 0
    neow_history: str = "full"

    def __call__(self) -> StsEnv:
        return StsEnv(
            LightspeedBackend(
                ascension=self.ascension,
                neow_history=self.neow_history,
            )
        )


@dataclass(frozen=True, slots=True)
class CurriculumEnvironmentFactory:
    spec: CurriculumSpec
    prefix_corpus: PrefixCorpus | None = None
    ascension: int = 0
    neow_history: str = "full"

    def __call__(self) -> CurriculumEnvironment:
        return CurriculumEnvironment(
            environment_factory=LightspeedEnvironmentFactory(
                ascension=self.ascension,
                neow_history=self.neow_history,
            ),
            spec=self.spec,
            prefix_corpus=self.prefix_corpus,
        )


def _environment_worker(
    connection: Connection,
    environment_factory: Callable[[], Any],
) -> None:
    try:
        environment = environment_factory()
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(("result", environment.reset(seed=int(payload))))
            elif command == "replay":
                trace = EpisodeTrace.from_dict(payload)
                observation = (
                    environment.replay_recovery_trace(trace)
                    if hasattr(environment, "replay_recovery_trace")
                    else replay_trace(environment, trace)
                )
                connection.send(("result", observation))
            elif command == "step":
                connection.send(("result", environment.step(payload)))
            elif command == "close":
                connection.send(("closed", None))
                break
            else:
                raise ValueError(f"unknown worker command: {command}")
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException as error:
        try:
            connection.send(
                (
                    "error",
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class SubprocessVectorEnvironment:
    def __init__(
        self,
        environment_factory: Callable[[], Any],
        num_environments: int,
        *,
        start_method: str = "spawn",
    ):
        if num_environments <= 0:
            raise ValueError("num_environments must be positive")
        context = multiprocessing.get_context(start_method)
        self._connections: list[Connection] = []
        self._processes: list[multiprocessing.Process] = []
        self._closed = False
        for _ in range(num_environments):
            parent, child = context.Pipe()
            process = context.Process(
                target=_environment_worker,
                args=(child, environment_factory),
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        for connection in self._connections:
            status, payload = connection.recv()
            if status != "ready":
                self._raise_worker_error(status, payload)

    @property
    def num_environments(self) -> int:
        return len(self._connections)

    def reset(self, seeds: tuple[int, ...]) -> tuple[tuple[Observation, dict[str, Any]], ...]:
        if len(seeds) != self.num_environments:
            raise ValueError("reset seed count must equal environment count")
        for connection, seed in zip(self._connections, seeds, strict=True):
            connection.send(("reset", seed))
        return tuple(self._receive(connection) for connection in self._connections)

    def reset_at(self, environment_index: int, seed: int) -> tuple[Observation, dict[str, Any]]:
        connection = self._connections[environment_index]
        connection.send(("reset", seed))
        return self._receive(connection)

    def replay(self, traces: tuple[EpisodeTrace, ...]) -> tuple[Observation, ...]:
        if len(traces) != self.num_environments:
            raise ValueError("trace count must equal environment count")
        for connection, trace in zip(self._connections, traces, strict=True):
            connection.send(("replay", trace.to_dict()))
        return tuple(self._receive(connection) for connection in self._connections)

    def step(
        self,
        actions: tuple[Action, ...],
    ) -> tuple[tuple[Observation, float, bool, bool, dict[str, Any]], ...]:
        if len(actions) != self.num_environments:
            raise ValueError("action count must equal environment count")
        for connection, action in zip(self._connections, actions, strict=True):
            connection.send(("step", action))
        return tuple(self._receive(connection) for connection in self._connections)

    def close(self) -> None:
        if self._closed:
            return
        for connection, process in zip(self._connections, self._processes, strict=True):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection, process in zip(self._connections, self._processes, strict=True):
            if process.is_alive() and connection.poll(2.0):
                try:
                    self._receive(connection, expected_status="closed")
                except RuntimeError:
                    pass
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            connection.close()
        self._closed = True

    def __enter__(self) -> SubprocessVectorEnvironment:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _receive(connection: Connection, expected_status: str = "result") -> Any:
        status, payload = connection.recv()
        if status == expected_status:
            return payload
        SubprocessVectorEnvironment._raise_worker_error(status, payload)

    @staticmethod
    def _raise_worker_error(status: str, payload: Any) -> None:
        if status == "error" and isinstance(payload, dict):
            raise RuntimeError(
                f"environment worker failed with {payload.get('type')}: "
                f"{payload.get('message')}\n{payload.get('traceback')}"
            )
        raise RuntimeError(f"unexpected environment worker response: {status}")
