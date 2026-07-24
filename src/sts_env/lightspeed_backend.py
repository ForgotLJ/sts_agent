from __future__ import annotations

from dataclasses import replace
import importlib
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import secrets
import sys
from typing import Any, Literal, Self

from sts_env.backend import Transition
from sts_env.types import (
    Action,
    ActionKind,
    CardView,
    EnemyView,
    MapNodeView,
    Observation,
    Phase,
    PlayerView,
)


_COMBAT_CARD = 0
_COMBAT_POTION = 1
_COMBAT_SINGLE_SELECT = 2
_COMBAT_MULTI_SELECT = 3
_COMBAT_END_TURN = 4

_REWARD_CARD = 0
_REWARD_GOLD = 1
_REWARD_KEY = 2
_REWARD_POTION = 3
_REWARD_RELIC = 4
_REWARD_CARD_REMOVE = 5
_REWARD_SKIP = 6

_REWARD_SOURCE_IDS = {
    _REWARD_CARD: "card",
    _REWARD_GOLD: "gold",
    _REWARD_KEY: "key",
    _REWARD_POTION: "potion",
    _REWARD_RELIC: "relic",
}


def _load_lightspeed_module() -> Any:
    try:
        return importlib.import_module("slaythespire")
    except ModuleNotFoundError as original_error:
        project_root = Path(__file__).resolve().parents[2]
        build_dir = project_root / "build" / "sts_lightspeed-py311"
        extension_paths = [
            path
            for suffix in EXTENSION_SUFFIXES
            for path in build_dir.glob(f"slaythespire*{suffix}")
        ]
        if not extension_paths:
            raise RuntimeError(
                "sts_lightspeed is not built; run the platform-specific build "
                "script under scripts first"
            ) from original_error
        sys.path.insert(0, str(build_dir))
        return importlib.import_module("slaythespire")


class LightspeedBackend:
    def __init__(
        self,
        ascension: int = 0,
        neow_history: Literal["full", "limited", "skipped"] = "full",
        act1_boss_history: Literal[
            "guardian_unseen",
            "hexaghost_unseen",
            "slime_boss_unseen",
            "all_seen",
        ] = "all_seen",
        final_act_unlocked: bool = True,
    ):
        if ascension < 0 or ascension > 20:
            raise ValueError("ascension must be between 0 and 20")
        if neow_history not in {"full", "limited", "skipped"}:
            raise ValueError("neow_history must be 'full', 'limited', or 'skipped'")
        if act1_boss_history not in {
            "guardian_unseen",
            "hexaghost_unseen",
            "slime_boss_unseen",
            "all_seen",
        }:
            raise ValueError(
                "act1_boss_history must be 'guardian_unseen', 'hexaghost_unseen', "
                "'slime_boss_unseen', or 'all_seen'"
            )
        self._ascension = ascension
        self._neow_history = neow_history
        self._act1_boss_history = act1_boss_history
        self._final_act_unlocked = final_act_unlocked
        self._module = _load_lightspeed_module()
        self._bridge: Any | None = None
        self._observation: Observation | None = None
        self._action_tokens: dict[Action, int] = {}
        self._card_reward_entries: dict[Action, int] = {}
        self._card_reward_skip_actions: set[Action] = set()
        self._pending_card_reward: int | None = None
        self._shop_entries: set[Action] = set()
        self._shop_close_actions: set[Action] = set()
        self._shop_open = False
        self._seed: int | None = None

    @property
    def supports_clone(self) -> bool:
        return True

    @property
    def supports_redeterminization(self) -> bool:
        return True

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        resolved_seed = secrets.randbits(63) if seed is None else seed
        if resolved_seed < 0 or resolved_seed >= 2**64:
            raise ValueError("seed must be in [0, 2**64)")

        self._seed = resolved_seed
        neow_mode = {"full": 0, "limited": 1, "skipped": 2}[self._neow_history]
        act_one_bosses_seen = {
            "guardian_unseen": 0,
            "hexaghost_unseen": 1,
            "slime_boss_unseen": 2,
            "all_seen": 3,
        }[self._act1_boss_history]
        self._bridge = self._module.SimulatorBridge(
            resolved_seed,
            self._ascension,
            neow_mode,
            act_one_bosses_seen,
            self._final_act_unlocked,
        )
        self._pending_card_reward = None
        self._card_reward_skip_actions = set()
        self._shop_open = False
        self._observation = self._read_observation()
        return self._observation, {
            "seed": resolved_seed,
            "backend": "sts_lightspeed",
            "ascension": self._ascension,
            "neow_history": self._neow_history,
            "act1_boss_history": self._act1_boss_history,
            "final_act_unlocked": self._final_act_unlocked,
        }

    def step(self, action: Action) -> Transition:
        bridge = self._require_bridge()
        pending_card_reward = self._card_reward_entries.get(action)
        if pending_card_reward is not None:
            self._pending_card_reward = pending_card_reward
            observation = self._read_observation()
            self._observation = observation
            return self._transition(observation)
        if action in self._card_reward_skip_actions:
            if self._pending_card_reward is None:
                raise RuntimeError("card reward skip has no pending reward")
            bridge.skip_card_reward(self._pending_card_reward)
            self._pending_card_reward = None
            observation = self._read_observation()
            self._observation = observation
            return self._transition(observation)
        if action in self._shop_entries:
            self._shop_open = True
            observation = self._read_observation()
            self._observation = observation
            return self._transition(observation)
        if action in self._shop_close_actions:
            self._shop_open = False
            observation = self._read_observation()
            self._observation = observation
            return self._transition(observation)

        try:
            token = self._action_tokens[action]
        except KeyError as error:
            raise ValueError("backend received an illegal or stale action") from error

        bridge.step(token)
        self._pending_card_reward = None
        observation = self._read_observation()
        self._observation = observation

        return self._transition(observation)

    def _transition(self, observation: Observation) -> Transition:
        bridge = self._require_bridge()

        state = bridge.observe()
        outcome = str(state["outcome"])
        terminated = bool(state["terminated"])
        reward = 1.0 if outcome == "player_victory" else -1.0 if outcome == "player_loss" else 0.0
        return Transition(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={
                "seed": self._seed,
                "outcome": outcome,
                "phase": observation.phase.value,
                "floor": observation.floor,
            },
        )

    def clone(self) -> Self:
        bridge = self._require_bridge()
        cloned = object.__new__(type(self))
        cloned._ascension = self._ascension
        cloned._neow_history = self._neow_history
        cloned._act1_boss_history = self._act1_boss_history
        cloned._final_act_unlocked = self._final_act_unlocked
        cloned._module = self._module
        cloned._bridge = bridge.clone()
        cloned._observation = self._observation
        cloned._action_tokens = self._action_tokens.copy()
        cloned._card_reward_entries = self._card_reward_entries.copy()
        cloned._card_reward_skip_actions = self._card_reward_skip_actions.copy()
        cloned._pending_card_reward = self._pending_card_reward
        cloned._shop_entries = self._shop_entries.copy()
        cloned._shop_close_actions = self._shop_close_actions.copy()
        cloned._shop_open = self._shop_open
        cloned._seed = self._seed
        return cloned

    def redeterminized_clone(
        self,
        search_seed: int,
        known_top: tuple[str, ...] = (),
        known_bottom: tuple[str, ...] = (),
    ) -> Self:
        if search_seed < 0 or search_seed >= 2**64:
            raise ValueError("search_seed must be in [0, 2**64)")
        if self._observation is None:
            raise RuntimeError("reset() must be called before redeterminization")
        if self._observation.phase is not Phase.COMBAT and (known_top or known_bottom):
            raise ValueError("draw-order constraints require a combat observation")
        bridge = self._require_bridge()
        cloned = object.__new__(type(self))
        cloned._ascension = self._ascension
        cloned._neow_history = self._neow_history
        cloned._act1_boss_history = self._act1_boss_history
        cloned._final_act_unlocked = self._final_act_unlocked
        cloned._module = self._module
        cloned._bridge = bridge.redeterminized_clone(
            search_seed,
            list(known_top),
            list(known_bottom),
        )
        cloned._observation = None
        cloned._action_tokens = {}
        cloned._card_reward_entries = {}
        cloned._card_reward_skip_actions = set()
        cloned._pending_card_reward = self._pending_card_reward
        cloned._shop_entries = set()
        cloned._shop_close_actions = set()
        cloned._shop_open = self._shop_open
        cloned._seed = self._seed
        cloned._observation = cloned._read_observation()
        if cloned._observation != self._observation:
            raise RuntimeError("redeterminization changed the public observation")
        return cloned

    def _require_bridge(self) -> Any:
        if self._bridge is None:
            raise RuntimeError("reset() must be called before using the backend")
        return self._bridge

    def _read_observation(self) -> Observation:
        bridge = self._require_bridge()
        state = dict(bridge.observe())
        descriptors = [dict(descriptor) for descriptor in bridge.legal_actions()]
        if str(state["screen_state"]) != "shop":
            self._shop_open = False

        actions: list[Action] = []
        action_tokens: dict[Action, int] = {}
        card_reward_entries: dict[Action, int] = {}
        card_reward_skip_actions: set[Action] = set()
        shop_entries: set[Action] = set()
        shop_close_actions: set[Action] = set()

        def add_action(
            action: Action,
            token: int | None = None,
            card_reward: int | None = None,
            card_reward_skip: bool = False,
            shop_entry: bool = False,
            shop_close: bool = False,
        ) -> None:
            candidate = action
            collision_index = 2
            while (
                candidate in action_tokens
                or candidate in card_reward_entries
                or candidate in card_reward_skip_actions
                or candidate in shop_entries
                or candidate in shop_close_actions
            ):
                candidate = replace(candidate, label=f"{candidate.label} ({collision_index})")
                collision_index += 1
            actions.append(candidate)
            if token is not None:
                action_tokens[candidate] = token
            if card_reward is not None:
                card_reward_entries[candidate] = card_reward
            if card_reward_skip:
                card_reward_skip_actions.add(candidate)
            if shop_entry:
                shop_entries.add(candidate)
            if shop_close:
                shop_close_actions.add(candidate)

        if str(state["screen_state"]) == "rewards":
            self._read_reward_actions(descriptors, add_action)
        elif str(state["screen_state"]) == "shop" and not self._shop_open:
            self._read_shop_entry_actions(descriptors, add_action)
        elif str(state["screen_state"]) == "shop":
            self._read_shop_inventory_actions(descriptors, add_action)
        else:
            for descriptor in descriptors:
                add_action(
                    self._action_from_descriptor(descriptor),
                    token=int(descriptor["token"]),
                )
        self._action_tokens = action_tokens
        self._card_reward_entries = card_reward_entries
        self._card_reward_skip_actions = card_reward_skip_actions
        self._shop_entries = shop_entries
        self._shop_close_actions = shop_close_actions
        selectable_cards = [
            action
            for action in actions
            if action.kind is ActionKind.CHOOSE_CARD and action.source_id is not None
        ]
        if selectable_cards:
            ordered_cards = iter(
                sorted(
                    selectable_cards,
                    key=lambda action: (type(action.source_id).__name__, str(action.source_id)),
                )
            )
            actions = [
                next(ordered_cards)
                if action.kind is ActionKind.CHOOSE_CARD and action.source_id is not None
                else action
                for action in actions
            ]

        player_state = dict(state["player"])
        hand = tuple(
            CardView(
                instance_id=int(card["instance_id"]),
                card_id=str(card["card_id"]),
                name=str(card["name"]),
                cost=int(card["cost"]),
                upgraded=bool(card["upgraded"]),
                playable=bool(card["playable"]),
                requires_target=bool(card["requires_target"]),
            )
            for card in state["hand"]
        )
        relics = tuple((str(relic_id), int(value)) for relic_id, value in state["relics"])
        hide_enemy_intents = any(
            self._stable_source_id(relic_id) == "runicdome" for relic_id, _ in relics
        )
        enemies = tuple(
            EnemyView(
                enemy_id=int(enemy["enemy_id"]),
                monster_id=str(enemy["monster_id"]),
                name=str(enemy["name"]),
                hp=int(enemy["hp"]),
                max_hp=int(enemy["max_hp"]),
                block=int(enemy["block"]),
                intent_damage=(
                    0
                    if int(enemy["hp"]) <= 0 or hide_enemy_intents
                    else int(enemy["intent_damage"])
                ),
                intent_hits=(
                    0
                    if int(enemy["hp"]) <= 0 or hide_enemy_intents
                    else int(enemy["intent_hits"])
                ),
                statuses=(
                    ()
                    if int(enemy["hp"]) <= 0
                    else tuple((str(name), int(value)) for name, value in enemy["statuses"])
                ),
            )
            for enemy in state["enemies"]
        )

        return Observation(
            phase=Phase(str(state["phase"])),
            turn=int(state["turn"]),
            player=PlayerView(
                hp=int(player_state["hp"]),
                max_hp=int(player_state["max_hp"]),
                block=int(player_state["block"]),
                energy=int(player_state["energy"]),
                gold=int(player_state["gold"]),
                statuses=tuple(
                    (str(name), int(value)) for name, value in player_state["statuses"]
                ),
            ),
            hand=hand,
            enemies=enemies,
            draw_pile=tuple((str(card_id), int(count)) for card_id, count in state["draw_pile"]),
            discard_pile=tuple(
                (str(card_id), int(count)) for card_id, count in state["discard_pile"]
            ),
            exhaust_pile=tuple(
                (str(card_id), int(count)) for card_id, count in state["exhaust_pile"]
            ),
            legal_actions=tuple(actions),
            ascension=int(state["ascension"]),
            act=int(state["act"]),
            floor=int(state["floor"]),
            map_x=int(state["map_x"]),
            map_y=int(state["map_y"]),
            screen_state=(
                "card_reward"
                if self._pending_card_reward is not None
                else "shop_screen"
                if str(state["screen_state"]) == "shop" and self._shop_open
                else str(state["screen_state"])
            ),
            deck=tuple((str(card_id), int(count)) for card_id, count in state["deck"]),
            relics=relics,
            potions=tuple(str(potion_id) for _, potion_id in state["potions"]),
            map_nodes=tuple(
                MapNodeView(
                    x=int(node["x"]),
                    y=int(node["y"]),
                    symbol=str(node.get("symbol") or ""),
                    children=tuple(
                        (int(child[0]), int(child[1])) for child in node.get("children", ())
                    ),
                    burning_elite=bool(node.get("burning_elite", False)),
                )
                for node in state.get("map", ())
            ),
            act_boss=self._stable_source_id(str(state.get("act_boss") or "")) or "",
            ruby_key=bool(state.get("ruby_key", False)),
            emerald_key=bool(state.get("emerald_key", False)),
            sapphire_key=bool(state.get("sapphire_key", False)),
            potion_capacity=int(state.get("potion_capacity", len(state["potions"]))),
        )

    def _read_shop_entry_actions(self, descriptors: list[dict[str, Any]], add_action: Any) -> None:
        add_action(
            Action(
                kind=ActionKind.CHOOSE_OPTION,
                source_id="shop",
                choice_index=0,
                label="shop",
                option_type="shop",
            ),
            shop_entry=True,
        )
        skip_descriptor = next(
            descriptor
            for descriptor in descriptors
            if int(descriptor.get("reward_type", -1)) == _REWARD_SKIP
        )
        add_action(
            Action(
                kind=ActionKind.CHOOSE_OPTION,
                source_id="proceed",
                label="Proceed",
                option_type="proceed",
            ),
            token=int(skip_descriptor["token"]),
        )

    def _read_shop_inventory_actions(
        self,
        descriptors: list[dict[str, Any]],
        add_action: Any,
    ) -> None:
        for descriptor in descriptors:
            if int(descriptor.get("reward_type", -1)) == _REWARD_SKIP:
                add_action(
                    Action(kind=ActionKind.LEAVE, label="Return", option_type="leave"),
                    shop_close=True,
                )
                continue
            add_action(
                self._action_from_descriptor(descriptor),
                token=int(descriptor["token"]),
            )

    def _read_reward_actions(self, descriptors: list[dict[str, Any]], add_action: Any) -> None:
        if self._pending_card_reward is not None:
            for descriptor in descriptors:
                if int(descriptor.get("reward_type", -1)) != _REWARD_CARD:
                    continue
                if int(descriptor["idx1"]) != self._pending_card_reward:
                    continue
                card_id = str(descriptor.get("item_id") or descriptor.get("card_id") or "")
                card_name = str(
                    descriptor.get("item_name")
                    or descriptor.get("card_name")
                    or card_id
                    or "Choose card"
                )
                add_action(
                    Action(
                        kind=ActionKind.CHOOSE_CARD,
                        source_id=self._stable_source_id(card_id or card_name),
                        choice_index=int(descriptor["idx2"]),
                        label=card_name,
                        option_type="card",
                    ),
                    token=int(descriptor["token"]),
                )
            add_action(
                Action(
                    kind=ActionKind.LEAVE,
                    source_id="skip_card",
                    label="Skip card reward",
                    option_type="skip_card",
                ),
                card_reward_skip=True,
            )
            return

        reward_choice_index = 0
        seen_card_rewards: set[int] = set()
        source_counts: dict[str, int] = {}
        for descriptor in descriptors:
            reward_type = int(descriptor.get("reward_type", -1))
            if reward_type == _REWARD_SKIP:
                add_action(
                    Action(
                        kind=ActionKind.CHOOSE_OPTION,
                        source_id="proceed",
                        label="Proceed",
                        option_type="proceed",
                    ),
                    token=int(descriptor["token"]),
                )
                continue
            source_id = _REWARD_SOURCE_IDS.get(reward_type)
            if source_id is None:
                add_action(
                    self._action_from_descriptor(descriptor),
                    token=int(descriptor["token"]),
                )
                continue
            reward_index = int(descriptor["idx1"])
            if reward_type == _REWARD_CARD:
                if reward_index in seen_card_rewards:
                    continue
                seen_card_rewards.add(reward_index)
            occurrence = source_counts.get(source_id, 0)
            source_counts[source_id] = occurrence + 1
            item_id = str(descriptor.get("item_id") or "")
            stable_item_id = self._stable_source_id(item_id)
            semantic_base = stable_item_id or source_id
            semantic_id = semantic_base if occurrence == 0 else f"{semantic_base}:{occurrence}"
            action = Action(
                kind=ActionKind.CHOOSE_OPTION,
                source_id=semantic_id,
                choice_index=reward_choice_index,
                label=str(descriptor.get("item_name") or source_id),
                option_type=source_id,
                amount=int(descriptor.get("amount", 0)),
            )
            if reward_type == _REWARD_CARD:
                add_action(action, card_reward=reward_index)
            else:
                add_action(action, token=int(descriptor["token"]))
            reward_choice_index += 1

    def _action_from_descriptor(self, descriptor: dict[str, Any]) -> Action:
        label = str(descriptor.get("label") or "").strip()
        if bool(descriptor.get("potion_action", False)):
            is_discard = bool(descriptor.get("potion_discard", False))
            potion_id = self._stable_source_id(str(descriptor.get("potion_id") or ""))
            return Action(
                kind=ActionKind.DISCARD_POTION if is_discard else ActionKind.USE_POTION,
                source_id=int(descriptor["idx1"]),
                label=label or ("Discard potion" if is_discard else "Use potion"),
                option_type=potion_id or "potion",
            )
        if descriptor["domain"] == "combat":
            action_type = int(descriptor["action_type"])
            if action_type == _COMBAT_CARD:
                return Action(
                    kind=ActionKind.PLAY_CARD,
                    source_id=int(descriptor["source_instance_id"]),
                    target_id=self._optional_target(descriptor),
                    label=label or "Play card",
                    option_type="card",
                )
            if action_type == _COMBAT_POTION:
                is_discard = bool(descriptor.get("discard", False))
                return Action(
                    kind=ActionKind.DISCARD_POTION if is_discard else ActionKind.USE_POTION,
                    source_id=int(descriptor["source_index"]),
                    target_id=self._optional_target(descriptor),
                    label=label or ("Discard potion" if is_discard else "Use potion"),
                    option_type=self._stable_source_id(str(descriptor.get("potion_id") or ""))
                    or "potion",
                )
            if action_type in {_COMBAT_SINGLE_SELECT, _COMBAT_MULTI_SELECT}:
                selected_instance_id = descriptor.get("selected_card_instance_id")
                selected_card_name = str(descriptor.get("selected_card_name") or "")
                return Action(
                    kind=ActionKind.CHOOSE_CARD,
                    source_id=(
                        int(selected_instance_id)
                        if selected_instance_id is not None
                        else None
                    ),
                    choice_index=(
                        None
                        if selected_instance_id is not None
                        else int(descriptor["choice_index"])
                    ),
                    label=selected_card_name or label or "Choose card",
                    option_type="card",
                )
            if action_type == _COMBAT_END_TURN:
                return Action(
                    kind=ActionKind.END_TURN,
                    label=label or "End turn",
                    option_type="end_turn",
                )
            raise ValueError(f"unknown combat action type: {action_type}")

        screen_state = str(descriptor["screen_state"])
        idx1 = int(descriptor["idx1"])
        idx2 = int(descriptor["idx2"])
        if screen_state == "map":
            return Action(
                kind=ActionKind.CHOOSE_MAP_NODE,
                source_id=f"x{idx1}",
                choice_index=idx1,
                label=label or f"Choose map node {idx1}",
                option_type=str(descriptor.get("room_symbol") or ""),
                target_x=int(descriptor.get("target_x", idx1)),
                target_y=int(descriptor.get("target_y", -1)),
            )
        if screen_state == "card_select":
            item_id = str(descriptor.get("item_id") or "")
            return Action(
                kind=ActionKind.CHOOSE_CARD,
                source_id=self._stable_source_id(item_id),
                choice_index=idx1,
                label=label or f"Choose card {idx1}",
                option_type="card",
            )
        if screen_state == "shop":
            reward_type = int(descriptor.get("reward_type", -1))
            if reward_type == _REWARD_CARD_REMOVE:
                kind = ActionKind.REMOVE_CARD
            elif reward_type == _REWARD_SKIP:
                kind = ActionKind.LEAVE
            else:
                kind = ActionKind.BUY
            item_id = str(descriptor.get("item_id") or "")
            item_name = str(descriptor.get("item_name") or item_id)
            if reward_type == _REWARD_CARD_REMOVE:
                source_id = "purge"
            elif reward_type == _REWARD_SKIP:
                source_id = None
            else:
                source_id = self._stable_source_id(item_id or item_name)
            return Action(
                kind=kind,
                source_id=source_id,
                choice_index=None if reward_type == _REWARD_SKIP else idx1,
                label=item_name or label or f"Shop choice {idx1}",
                option_type={
                    _REWARD_CARD: "card",
                    _REWARD_POTION: "potion",
                    _REWARD_RELIC: "relic",
                    _REWARD_CARD_REMOVE: "remove_card",
                    _REWARD_SKIP: "leave",
                }.get(reward_type, "shop"),
                gold_cost=int(descriptor.get("price", 0)),
            )
        if screen_state == "rewards":
            reward_type = int(descriptor.get("reward_type", -1))
            kind = ActionKind.CHOOSE_OPTION
            return Action(
                kind=kind,
                source_id=f"{reward_type}:{idx1}",
                choice_index=idx2,
                label=label or f"Reward choice {idx1}",
                option_type=_REWARD_SOURCE_IDS.get(reward_type, "reward"),
                amount=int(descriptor.get("amount", 0)),
            )
        if screen_state == "boss_relic_rewards":
            item_id = str(descriptor.get("item_id") or "")
            is_skip = item_id == "skip" or idx1 == 3
            return Action(
                kind=ActionKind.LEAVE if is_skip else ActionKind.CHOOSE_OPTION,
                source_id=None if is_skip else self._stable_source_id(item_id),
                choice_index=None if is_skip else idx1,
                label=str(descriptor.get("item_name") or label or "Boss relic"),
                option_type="leave" if is_skip else "boss_relic",
            )
        return Action(
            kind=ActionKind.CHOOSE_OPTION,
            source_id=self._stable_source_id(str(descriptor.get("item_id") or "")),
            choice_index=idx1,
            label=label or f"Choose option {idx1}",
            description=str(descriptor.get("description") or ""),
            option_type=self._stable_source_id(
                str(descriptor.get("option_type") or label)
            )
            or "option",
        )

    @staticmethod
    def _optional_target(descriptor: dict[str, Any]) -> int | None:
        target = int(descriptor["target_index"])
        return target if target >= 0 else None

    @staticmethod
    def _stable_source_id(value: str) -> str | None:
        normalized = "".join(character for character in value.lower() if character.isalnum())
        return normalized or None
