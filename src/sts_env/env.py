from __future__ import annotations

from dataclasses import replace
from typing import Any

from sts_env.backend import SimulatorBackend
from sts_env.types import Action, ActionKind, Observation, Phase, RunHistoryView


class StsEnv:
    """Small Gymnasium-like wrapper with dynamic legal actions.

    An integer action always indexes the legal action tuple contained in the
    latest public observation. Passing an Action directly is useful for search.
    """

    def __init__(self, backend: SimulatorBackend):
        self._backend = backend
        self._observation: Observation | None = None
        self._history = RunHistoryView()

    @property
    def observation(self) -> Observation:
        if self._observation is None:
            raise RuntimeError("reset() must be called before reading the observation")
        return self._observation

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        observation, info = self._backend.reset(seed=seed)
        self._history = RunHistoryView()
        self._observation = replace(observation, history=self._history)
        return self._observation, info

    def step(
        self, action: int | Action
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        current = self.observation
        resolved_action = self._resolve_action(current, action)
        transition = self._backend.step(resolved_action)
        self._history = self._advance_history(
            self._history,
            current,
            resolved_action,
            transition.observation,
        )
        self._observation = replace(transition.observation, history=self._history)
        return (
            self._observation,
            transition.reward,
            transition.terminated,
            transition.truncated,
            transition.info,
        )

    def action_mask(self, capacity: int) -> tuple[bool, ...]:
        if capacity < len(self.observation.legal_actions):
            raise ValueError("capacity cannot be smaller than the legal action count")
        return (True,) * len(self.observation.legal_actions) + (False,) * (
            capacity - len(self.observation.legal_actions)
        )

    def clone(self) -> StsEnv:
        if not self._backend.supports_clone:
            raise RuntimeError("this backend does not support independent cloning")
        cloned = StsEnv(self._backend.clone())
        cloned._observation = self._observation
        cloned._history = self._history
        return cloned

    def redeterminized_clone(
        self,
        search_seed: int,
        known_top: tuple[str, ...] = (),
        known_bottom: tuple[str, ...] = (),
    ) -> StsEnv:
        if not self._backend.supports_redeterminization:
            raise RuntimeError("this backend does not support belief-state sampling")
        cloned = StsEnv(
            self._backend.redeterminized_clone(
                search_seed,
                known_top=known_top,
                known_bottom=known_bottom,
            )
        )
        cloned._observation = self._observation
        cloned._history = self._history
        return cloned

    @classmethod
    def _advance_history(
        cls,
        history: RunHistoryView,
        previous: Observation,
        action: Action,
        current: Observation,
    ) -> RunHistoryView:
        previous_deck_size = sum(count for _, count in previous.deck)
        current_deck_size = sum(count for _, count in current.deck)
        deck_delta = current_deck_size - previous_deck_size
        room_changed = (
            current.map_x >= 0
            and current.map_y >= 0
            and (
                previous.act != current.act
                or previous.map_x != current.map_x
                or previous.map_y != current.map_y
            )
        )
        combat_won = (
            previous.phase is Phase.COMBAT
            and current.phase is not Phase.COMBAT
            and current.player.hp > 0
        )
        completed_room = cls._room_symbol(previous) if combat_won else ""
        recent_rooms = history.recent_rooms
        if room_changed:
            current_room = cls._room_symbol(current)
            recent_rooms = (*recent_rooms, current_room or "unknown")[-16:]
        action_identity = action.option_type or (
            str(action.source_id) if action.source_id is not None else action.kind.value
        )
        recent_actions = (*history.recent_actions, f"{action.kind.value}:{action_identity}")[-32:]
        hp_delta = current.player.hp - previous.player.hp
        gold_delta = current.player.gold - previous.player.gold
        return RunHistoryView(
            decisions=history.decisions + 1,
            rooms_visited=history.rooms_visited + int(room_changed),
            combats_won=history.combats_won + int(combat_won),
            elites_won=history.elites_won + int(combat_won and completed_room == "E"),
            bosses_won=history.bosses_won + int(combat_won and completed_room == "B"),
            acts_cleared=history.acts_cleared + max(0, current.act - previous.act),
            cards_added=history.cards_added + max(0, deck_delta),
            cards_removed=history.cards_removed + max(0, -deck_delta),
            potions_used=history.potions_used + int(action.kind is ActionKind.USE_POTION),
            potions_discarded=(
                history.potions_discarded + int(action.kind is ActionKind.DISCARD_POTION)
            ),
            gold_spent=history.gold_spent + max(0, -gold_delta),
            hp_lost=history.hp_lost + max(0, -hp_delta),
            hp_healed=history.hp_healed + max(0, hp_delta),
            recent_rooms=recent_rooms,
            recent_actions=recent_actions,
        )

    @staticmethod
    def _room_symbol(observation: Observation) -> str:
        if observation.map_y >= 15:
            return "B"
        return next(
            (
                node.symbol
                for node in observation.map_nodes
                if node.x == observation.map_x and node.y == observation.map_y
            ),
            "",
        )

    @staticmethod
    def _resolve_action(observation: Observation, action: int | Action) -> Action:
        if isinstance(action, int):
            if action < 0 or action >= len(observation.legal_actions):
                raise IndexError(
                    f"action index {action} is outside [0, {len(observation.legal_actions)})"
                )
            return observation.legal_actions[action]

        if action not in observation.legal_actions:
            raise ValueError("action is not legal in the current observation")
        return action
