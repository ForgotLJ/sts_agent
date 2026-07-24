from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ParameterEMAConfig:
    enabled: bool = True
    decay: float = 0.998
    stages: tuple[str, ...] = ("full_run",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        if not 0.0 <= self.decay < 1.0 or not self.stages or any(
            not stage for stage in self.stages
        ):
            raise ValueError("parameter EMA configuration is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParameterEMA:
    def __init__(self, module: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError("parameter EMA decay must be in [0, 1)")
        self.decay = decay
        self._averaged = self._clone_state(module)

    def reset(self, module: nn.Module) -> None:
        self._averaged = self._clone_state(module)

    def update(self, module: nn.Module) -> None:
        current = module.state_dict()
        self._validate_state(current)
        with torch.no_grad():
            for key, current_tensor in current.items():
                averaged_tensor = self._averaged[key]
                if torch.is_floating_point(current_tensor):
                    averaged_tensor.mul_(self.decay).add_(
                        current_tensor.detach(),
                        alpha=1.0 - self.decay,
                    )
                else:
                    averaged_tensor.copy_(current_tensor.detach())

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "averaged": {
                key: tensor.detach().cpu().clone()
                for key, tensor in self._averaged.items()
            },
        }

    def load_state_dict(self, module: nn.Module, payload: dict[str, Any]) -> None:
        if float(payload.get("decay", -1.0)) != self.decay:
            raise ValueError("parameter EMA decay differs from the checkpoint")
        current = module.state_dict()
        restored = dict(payload.get("averaged") or {})
        if current.keys() != restored.keys():
            raise ValueError("parameter EMA keys differ from the network")
        self._averaged = {
            key: restored[key].detach().to(current[key].device).clone()
            for key in current
        }
        self._validate_state(current)

    @contextmanager
    def use_averaged_parameters(self, module: nn.Module) -> Iterator[None]:
        current = self._clone_state(module)
        self._validate_state(current)
        module.load_state_dict(self._averaged)
        try:
            yield
        finally:
            module.load_state_dict(current)

    def _validate_state(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self._averaged.keys():
            raise ValueError("parameter EMA keys differ from the network")
        for key, tensor in state.items():
            if tensor.shape != self._averaged[key].shape:
                raise ValueError(f"parameter EMA tensor shape differs for {key}")

    @staticmethod
    def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            key: tensor.detach().clone()
            for key, tensor in module.state_dict().items()
        }
