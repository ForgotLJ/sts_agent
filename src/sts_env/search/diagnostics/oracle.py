from __future__ import annotations

from sts_env.env import StsEnv
from sts_env.types import Observation


class ExactCloneOracleSource:
    """Cheating diagnostic source that preserves the simulator's true hidden future."""

    def __init__(self, environment: StsEnv, *, allow_hidden_state: bool = False):
        if not allow_hidden_state:
            raise PermissionError(
                "exact hidden-state cloning is diagnostic-only; pass allow_hidden_state=True explicitly"
            )
        self._environment = environment

    @property
    def observation(self) -> Observation:
        return self._environment.observation

    def sample(self, search_seed: int) -> StsEnv:
        del search_seed
        return self._environment.clone()
