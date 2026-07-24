from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import re
import socket
import time
from typing import Any, Callable, Protocol, Self

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


class CommunicationTransport(Protocol):
    def connect(self) -> None:
        ...

    def receive(self) -> dict[str, Any]:
        ...

    def exchange(self, command: str) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class SocketRelayTransport:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 51234,
        timeout: float = 30.0,
        connect_wait_timeout: float = 0.0,
        retry_interval: float = 0.1,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if connect_wait_timeout < 0:
            raise ValueError("connect_wait_timeout must be non-negative")
        if retry_interval < 0:
            raise ValueError("retry_interval must be non-negative")
        self._host = host
        self._port = port
        self._timeout = timeout
        self._connect_wait_timeout = connect_wait_timeout
        self._retry_interval = retry_interval
        self._socket: socket.socket | None = None
        self._buffer = bytearray()

    def connect(self) -> None:
        if self._socket is not None:
            return
        deadline = time.monotonic() + self._connect_wait_timeout
        while True:
            try:
                connection = socket.create_connection(
                    (self._host, self._port),
                    timeout=self._timeout,
                )
                break
            except OSError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(self._retry_interval, remaining))
        connection.settimeout(self._timeout)
        self._socket = connection

    def receive(self) -> dict[str, Any]:
        connection = self._require_socket()
        while True:
            newline_index = self._buffer.find(b"\n")
            if newline_index >= 0:
                line = bytes(self._buffer[:newline_index])
                del self._buffer[: newline_index + 1]
                return json.loads(line.decode("utf-8"))
            chunk = connection.recv(65536)
            if not chunk:
                raise ConnectionError("CommunicationMod relay closed the connection")
            self._buffer.extend(chunk)

    def exchange(self, command: str) -> dict[str, Any]:
        connection = self._require_socket()
        connection.sendall(command.encode("utf-8") + b"\n")
        return self.receive()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
            self._buffer.clear()

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("transport is not connected")
        return self._socket


class CommunicationBackend:
    def __init__(
        self,
        ascension: int = 0,
        host: str = "127.0.0.1",
        port: int = 51234,
        timeout: float = 30.0,
        connect_wait_timeout: float = 0.0,
        state_sync_timeout: float = 10.0,
        transport: CommunicationTransport | None = None,
    ):
        if ascension < 0 or ascension > 20:
            raise ValueError("ascension must be between 0 and 20")
        if state_sync_timeout <= 0:
            raise ValueError("state_sync_timeout must be positive")
        self._ascension = ascension
        self._state_sync_timeout = state_sync_timeout
        self._transport = transport or SocketRelayTransport(
            host,
            port,
            timeout,
            connect_wait_timeout,
        )
        self._state: dict[str, Any] | None = None
        self._observation: Observation | None = None
        self._action_commands: dict[Action, str] = {}
        self._seed: int | None = None
        self._map_x = -1
        self._map_y = -1

    @property
    def supports_clone(self) -> bool:
        return False

    @property
    def supports_redeterminization(self) -> bool:
        return False

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        seed = self._validate_seed(seed)

        self._transport.connect()
        state = self._await_state(
            self._transport.receive(),
            lambda candidate: not candidate.get("in_game")
            and "start" in candidate.get("available_commands", ()),
            "main menu",
        )
        if state.get("in_game"):
            raise RuntimeError("real game must be at the main menu before reset()")
        if "start" not in state.get("available_commands", ()):
            raise RuntimeError("CommunicationMod is not ready to start a game")

        seed_string = self._seed_string(seed)
        self._seed = seed
        self._map_x = -1
        self._map_y = -1
        self._state = self._await_state(
            self._exchange(f"start ironclad {self._ascension} {seed_string}"),
            lambda candidate: bool(candidate.get("in_game"))
            and self._is_publicly_actionable_state(candidate),
            "new run",
        )
        if not self._state.get("in_game"):
            raise RuntimeError("CommunicationMod did not enter a new run before timeout")
        self._require_matching_seed(self._state, seed)
        self._observation = self._read_observation()
        return self._observation, {
            "seed": seed,
            "game_seed": seed_string,
            "backend": "communication_mod",
            "ascension": self._ascension,
        }

    def attach(self, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        seed = self._validate_seed(seed)
        self._transport.connect()
        state = self._await_state(
            self._transport.receive(),
            lambda candidate: bool(candidate.get("in_game"))
            and self._is_publicly_actionable_state(candidate),
            "action-ready in-game state",
        )
        if not state.get("in_game"):
            raise RuntimeError("real game must be in progress before attach()")

        game = dict(state.get("game_state") or {})
        self._require_matching_seed(state, seed)

        self._seed = seed
        self._state = state
        self._map_x, self._map_y = self._map_coordinates(dict(state["game_state"]))
        self._observation = self._read_observation()
        return self._observation, {
            "seed": seed,
            "game_seed": self._seed_string(seed),
            "backend": "communication_mod",
            "ascension": int(game.get("ascension_level", self._ascension)),
            "attached": True,
        }

    def step(self, action: Action) -> Transition:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        previous_signature = self._observation_signature(self._observation)
        try:
            command = self._action_commands[action]
        except KeyError as error:
            raise ValueError("backend received an illegal or stale action") from error

        changed_signature: str | None = None

        def transitioned(candidate: dict[str, Any]) -> bool:
            nonlocal changed_signature
            if not candidate.get("in_game") or not self._is_publicly_actionable_state(candidate):
                return False
            signature = self._observation_signature_for_state(candidate)
            if signature == previous_signature:
                changed_signature = None
                return False
            available = {
                str(available_command).lower()
                for available_command in candidate.get("available_commands", ())
            }
            if "wait" not in available:
                return True
            if signature == changed_signature:
                return True
            changed_signature = signature
            return False

        self._state = self._await_state(
            self._exchange(command),
            transitioned,
            "action-ready in-game state",
        )
        if action.kind is ActionKind.CHOOSE_MAP_NODE and action.target_x >= 0:
            self._map_x = action.target_x
            self._map_y = action.target_y
        observation = self._read_observation()
        if self._observation_signature(observation) == previous_signature:
            raise RuntimeError(
                f"CommunicationMod command '{command}' produced no observable state change "
                "before the synchronization timeout"
            )
        self._observation = observation
        terminated = observation.phase is Phase.TERMINAL
        victory = bool(self._state.get("game_state", {}).get("screen_state", {}).get("victory"))
        reward = 1.0 if terminated and victory else -1.0 if terminated else 0.0
        return Transition(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={
                "seed": self._seed,
                "phase": observation.phase.value,
                "floor": observation.floor,
                "victory": victory if terminated else None,
            },
        )

    def clone(self) -> Self:
        raise RuntimeError("the real game backend cannot clone independent game processes")

    def redeterminized_clone(
        self,
        search_seed: int,
        known_top: tuple[str, ...] = (),
        known_bottom: tuple[str, ...] = (),
    ) -> Self:
        raise RuntimeError("the real game backend cannot sample hidden game states")

    def close(self) -> None:
        self._transport.close()

    def _exchange(self, command: str) -> dict[str, Any]:
        state = self._transport.exchange(command)
        if "error" in state:
            raise RuntimeError(f"CommunicationMod rejected '{command}': {state['error']}")
        return state

    def _await_state(
        self,
        state: dict[str, Any],
        predicate: Callable[[dict[str, Any]], bool],
        description: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self._state_sync_timeout
        while not predicate(state):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return state
            available = {str(command).lower() for command in state.get("available_commands", ())}
            if "wait" in available:
                wait_ms = max(1, min(250, int(remaining * 1000)))
                state = self._exchange(f"wait {wait_ms}")
                continue
            if "state" in available:
                time.sleep(min(0.1, remaining))
                state = self._exchange("state")
                continue
            raise RuntimeError(
                f"CommunicationMod cannot synchronize to {description}; "
                f"available commands are {sorted(available)}"
            )
        return state

    @staticmethod
    def _is_actionable_state(state: dict[str, Any]) -> bool:
        if not state.get("in_game"):
            return True
        if not state.get("ready_for_command", False):
            return False
        game = dict(state.get("game_state") or {})
        if str(game.get("screen_type", "")) in {"GAME_OVER", "COMPLETE"}:
            return True
        available = {str(command).lower() for command in state.get("available_commands", ())}
        passive_commands = {"key", "click", "wait", "state"}
        return bool(available - passive_commands)

    @staticmethod
    def _observation_signature(observation: Observation) -> str:
        return json.dumps(
            observation.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _observation_signature_for_state(self, state: dict[str, Any]) -> str:
        return self._observation_signature(self._observation_for_state(state))

    def _observation_for_state(self, state: dict[str, Any]) -> Observation:
        previous_state = self._state
        previous_action_commands = self._action_commands
        try:
            self._state = state
            return self._read_observation()
        finally:
            self._state = previous_state
            self._action_commands = previous_action_commands

    def _is_publicly_actionable_state(self, state: dict[str, Any]) -> bool:
        if not self._is_actionable_state(state):
            return False
        if not state.get("in_game"):
            return True
        observation = self._observation_for_state(state)
        return observation.phase is Phase.TERMINAL or bool(observation.legal_actions)

    def _read_observation(self) -> Observation:
        if self._state is None or not self._state.get("in_game"):
            raise RuntimeError("CommunicationMod did not return an in-game state")
        game = dict(self._state["game_state"])
        screen_type = str(game.get("screen_type", "NONE"))
        phase = self._phase(game, screen_type)
        combat = dict(game.get("combat_state") or {})
        player_json = dict(combat.get("player") or {})

        actions: list[Action] = []
        commands: dict[Action, str] = {}

        def add_action(action: Action, command: str) -> None:
            candidate = action
            collision_index = 2
            while candidate in commands:
                candidate = replace(candidate, label=f"{candidate.label} ({collision_index})")
                collision_index += 1
            actions.append(candidate)
            commands[candidate] = command

        available = {str(command).lower() for command in self._state.get("available_commands", ())}
        hand_json = list(combat.get("hand") or [])
        monsters_json = list(combat.get("monsters") or [])
        presentation_overlay = str(game.get("screen_name", "")) in {"FTUE", "SETTINGS"}

        if not presentation_overlay and "play" in available:
            target_indices = [
                index
                for index, monster in enumerate(monsters_json)
                if not monster.get("is_gone", False) and not monster.get("half_dead", False)
            ]
            for hand_index, card in enumerate(hand_json):
                if not card.get("is_playable", False):
                    continue
                if card.get("has_target", False):
                    for target_index in target_indices:
                        add_action(
                            Action(
                                kind=ActionKind.PLAY_CARD,
                                source_id=hand_index,
                                target_id=target_index,
                                label=f"Play {card.get('name', card.get('id'))}",
                                option_type="card",
                            ),
                            f"play {hand_index + 1} {target_index}",
                        )
                else:
                    add_action(
                        Action(
                            kind=ActionKind.PLAY_CARD,
                            source_id=hand_index,
                            label=f"Play {card.get('name', card.get('id'))}",
                            option_type="card",
                        ),
                        f"play {hand_index + 1}",
                    )

        if not presentation_overlay and "end" in available:
            add_action(
                Action(
                    kind=ActionKind.END_TURN,
                    label="End turn",
                    option_type="end_turn",
                ),
                "end",
            )

        if not presentation_overlay and "potion" in available:
            for potion_index, potion in enumerate(game.get("potions") or []):
                if potion.get("can_discard", False):
                    add_action(
                        Action(
                            kind=ActionKind.DISCARD_POTION,
                            source_id=potion_index,
                            label=f"Discard {potion.get('name', potion.get('id'))}",
                            option_type=self._stable_id(
                                str(potion.get("id", potion.get("name", "potion")))
                            ),
                        ),
                        f"potion discard {potion_index}",
                    )
                if not potion.get("can_use", False):
                    continue
                if potion.get("requires_target", False):
                    for target_index in range(len(monsters_json)):
                        if target_index in target_indices:
                            add_action(
                                Action(
                                    kind=ActionKind.USE_POTION,
                                    source_id=potion_index,
                                    target_id=target_index,
                                    label=f"Use {potion.get('name', potion.get('id'))}",
                                    option_type=self._stable_id(
                                        str(potion.get("id", potion.get("name", "potion")))
                                    ),
                                ),
                                f"potion use {potion_index} {target_index}",
                            )
                else:
                    add_action(
                        Action(
                            kind=ActionKind.USE_POTION,
                            source_id=potion_index,
                            label=f"Use {potion.get('name', potion.get('id'))}",
                            option_type=self._stable_id(
                                str(potion.get("id", potion.get("name", "potion")))
                            ),
                        ),
                        f"potion use {potion_index}",
                    )

        if not presentation_overlay and "choose" in available:
            for choice_index, label in enumerate(game.get("choice_list") or []):
                choice_label = str(label)
                source_id = self._choice_source_id(screen_type, choice_label)
                choice_kind = self._choice_kind(phase, screen_type)
                if screen_type == "SHOP_SCREEN" and source_id == "purge":
                    choice_kind = ActionKind.REMOVE_CARD
                public_fields = self._choice_public_fields(
                    game,
                    screen_type,
                    choice_index,
                    choice_label,
                )
                add_action(
                    Action(
                        kind=choice_kind,
                        source_id=source_id,
                        choice_index=choice_index,
                        label=choice_label,
                        **public_fields,
                    ),
                    f"choose {choice_index}",
                )

        if not presentation_overlay and ("proceed" in available or "confirm" in available):
            add_action(
                Action(
                    kind=ActionKind.CHOOSE_OPTION,
                    source_id="proceed",
                    label="Proceed",
                    option_type="proceed",
                ),
                "proceed",
            )
        if not presentation_overlay and available.intersection(
            {"return", "cancel", "leave", "skip"}
        ):
            add_action(
                Action(kind=ActionKind.LEAVE, label="Return", option_type="leave"),
                "return",
            )
        if (
            presentation_overlay
            and "key" in available
        ):
            add_action(
                Action(
                    kind=ActionKind.CHOOSE_OPTION,
                    source_id="presentation",
                    label="Continue",
                    option_type="presentation",
                ),
                "key cancel 1000",
            )

        self._action_commands = commands
        hand = tuple(
            CardView(
                instance_id=index,
                card_id=str(card.get("id", "")),
                name=str(card.get("name", card.get("id", ""))),
                cost=int(card.get("cost", 0)),
                upgraded=int(card.get("upgrades", 0)) > 0,
                playable=bool(card.get("is_playable", False)),
                requires_target=bool(card.get("has_target", False)),
            )
            for index, card in enumerate(hand_json)
        )
        relics = tuple(
            (str(relic.get("id", "")), max(0, int(relic.get("counter", -1))))
            for relic in game.get("relics") or []
        )
        hide_enemy_intents = any(
            self._stable_id(relic_id) == "runicdome" for relic_id, _ in relics
        )
        enemies = tuple(
            EnemyView(
                enemy_id=index,
                monster_id=str(monster.get("id", "")),
                name=str(monster.get("name", monster.get("id", ""))),
                hp=int(monster.get("current_hp", 0)),
                max_hp=int(monster.get("max_hp", 0)),
                block=int(monster.get("block", 0)),
                intent_damage=(
                    0
                    if int(monster.get("current_hp", 0)) <= 0 or hide_enemy_intents
                    else self._intent_damage(monster)
                ),
                intent_hits=(
                    0
                    if int(monster.get("current_hp", 0)) <= 0 or hide_enemy_intents
                    else self._intent_hits(monster)
                ),
                statuses=(
                    ()
                    if int(monster.get("current_hp", 0)) <= 0
                    else self._power_state(monster.get("powers") or [])
                ),
            )
            for index, monster in enumerate(monsters_json)
        )

        return Observation(
            phase=phase,
            turn=int(combat.get("turn", 0)),
            player=PlayerView(
                hp=int(player_json.get("current_hp", game.get("current_hp", 0))),
                max_hp=int(player_json.get("max_hp", game.get("max_hp", 0))),
                block=int(player_json.get("block", 0)),
                energy=int(player_json.get("energy", 0)),
                gold=int(game.get("gold", 0)),
                statuses=self._power_state(player_json.get("powers") or []),
            ),
            hand=hand,
            enemies=enemies,
            draw_pile=self._card_counts(combat.get("draw_pile") or []),
            discard_pile=self._card_counts(combat.get("discard_pile") or []),
            exhaust_pile=self._card_counts(combat.get("exhaust_pile") or []),
            legal_actions=tuple(actions),
            ascension=int(game.get("ascension_level", 0)),
            act=int(game.get("act", 0)),
            floor=int(game.get("floor", 0)),
            map_x=self._map_x,
            map_y=self._map_y,
            screen_state=(
                str(game.get("screen_name", "")).lower()
                if presentation_overlay
                else screen_type.lower()
            ),
            deck=self._card_counts(game.get("deck") or []),
            relics=relics,
            potions=tuple(str(potion.get("id", "")) for potion in game.get("potions") or []),
            map_nodes=tuple(
                MapNodeView(
                    x=int(node.get("x", -1)),
                    y=int(node.get("y", -1)),
                    symbol=str(node.get("symbol", "")),
                    children=tuple(
                        (int(child.get("x", -1)), int(child.get("y", -1)))
                        for child in node.get("children") or []
                    ),
                    burning_elite=bool(node.get("burning_elite", False)),
                )
                for node in game.get("map") or []
            ),
            act_boss=self._stable_id(str(game.get("act_boss", ""))),
            ruby_key=bool((game.get("keys") or {}).get("ruby", False)),
            emerald_key=bool((game.get("keys") or {}).get("emerald", False)),
            sapphire_key=bool((game.get("keys") or {}).get("sapphire", False)),
            potion_capacity=len(game.get("potions") or []),
        )

    @staticmethod
    def _phase(game: dict[str, Any], screen_type: str) -> Phase:
        if screen_type in {"GAME_OVER", "COMPLETE"}:
            return Phase.TERMINAL
        if game.get("room_phase") == "COMBAT" and game.get("combat_state") is not None:
            return Phase.COMBAT
        if screen_type == "MAP":
            return Phase.MAP
        choices = [
            "".join(character for character in str(choice).lower() if character.isalnum())
            for choice in game.get("choice_list") or []
        ]
        if (
            screen_type == "EVENT"
            and choices in [["leave"], ["return"]]
        ):
            return Phase.MAP
        if screen_type in {"SHOP_ROOM", "SHOP_SCREEN"}:
            return Phase.SHOP
        if screen_type == "REST":
            return Phase.MAP if game.get("room_phase") == "COMPLETE" else Phase.REST_SITE
        if screen_type in {"CARD_REWARD", "COMBAT_REWARD", "BOSS_REWARD", "GRID", "HAND_SELECT"}:
            return Phase.CARD_REWARD
        if (
            str(game.get("screen_name", "")) in {"FTUE", "SETTINGS"}
            and game.get("room_phase") == "COMPLETE"
            and game.get("room_type") in {"MonsterRoom", "MonsterRoomElite", "MonsterRoomBoss"}
        ):
            return Phase.CARD_REWARD
        return Phase.EVENT

    @staticmethod
    def _choice_kind(phase: Phase, screen_type: str) -> ActionKind:
        if screen_type == "MAP":
            return ActionKind.CHOOSE_MAP_NODE
        if screen_type in {"CARD_REWARD", "GRID", "HAND_SELECT"}:
            return ActionKind.CHOOSE_CARD
        if screen_type in {"COMBAT_REWARD", "BOSS_REWARD", "SHOP_ROOM"}:
            return ActionKind.CHOOSE_OPTION
        if screen_type == "SHOP_SCREEN":
            return ActionKind.BUY
        return ActionKind.CHOOSE_OPTION

    @classmethod
    def _choice_public_fields(
        cls,
        game: dict[str, Any],
        screen_type: str,
        choice_index: int,
        label: str,
    ) -> dict[str, Any]:
        screen = dict(game.get("screen_state") or {})
        normalized_label = cls._stable_id(label)
        result: dict[str, Any] = {"option_type": normalized_label or "option"}

        if screen_type == "MAP":
            next_nodes = list(screen.get("next_nodes") or [])
            if choice_index < len(next_nodes):
                node = dict(next_nodes[choice_index])
                result.update(
                    option_type=str(node.get("symbol", "")),
                    target_x=int(node.get("x", -1)),
                    target_y=int(node.get("y", -1)),
                )
            elif bool(screen.get("boss_available", False)):
                result.update(option_type="B", target_x=0, target_y=16)
            return result

        if screen_type == "SHOP_SCREEN":
            if normalized_label == "purge":
                result.update(
                    option_type="remove_card",
                    gold_cost=int(screen.get("purge_cost", 0)),
                )
                return result
            for option_type, key in (("card", "cards"), ("relic", "relics"), ("potion", "potions")):
                for item in screen.get(key) or []:
                    item_id = cls._stable_id(str(item.get("id", item.get("name", ""))))
                    item_name = cls._stable_id(str(item.get("name", "")))
                    if normalized_label in {item_id, item_name}:
                        result.update(
                            option_type=option_type,
                            description=str(item.get("name", item.get("id", ""))),
                            gold_cost=int(item.get("price", 0)),
                        )
                        return result
            return result

        if screen_type == "CARD_REWARD":
            cards = list(screen.get("cards") or [])
            if choice_index < len(cards):
                card = dict(cards[choice_index])
                result.update(
                    option_type="card",
                    description=str(card.get("name", card.get("id", ""))),
                )
            return result

        if screen_type == "COMBAT_REWARD":
            rewards = list(screen.get("rewards") or [])
            if choice_index < len(rewards):
                reward = dict(rewards[choice_index])
                reward_type = str(reward.get("reward_type", "reward")).lower()
                result["option_type"] = reward_type
                if reward_type in {"gold", "stolen_gold"}:
                    result["amount"] = int(reward.get("gold", 0))
            return result

        if screen_type == "BOSS_REWARD":
            relics = list(screen.get("relics") or [])
            if choice_index < len(relics):
                relic = dict(relics[choice_index])
                result.update(
                    option_type="boss_relic",
                    description=str(relic.get("name", relic.get("id", ""))),
                )
            return result

        if screen_type == "EVENT":
            options = [
                dict(option)
                for option in screen.get("options") or []
                if not option.get("disabled", False)
            ]
            if choice_index < len(options):
                result["description"] = str(options[choice_index].get("text", ""))
            return result

        if screen_type in {"GRID", "HAND_SELECT"}:
            cards = list(screen.get("cards") or screen.get("hand") or [])
            if choice_index < len(cards):
                card = dict(cards[choice_index])
                result.update(
                    option_type="card",
                    description=str(card.get("name", card.get("id", ""))),
                )
            return result

        return result

    @classmethod
    def _choice_source_id(cls, screen_type: str, label: str) -> str | None:
        if screen_type not in {
            "MAP",
            "SHOP_ROOM",
            "SHOP_SCREEN",
            "CARD_REWARD",
            "COMBAT_REWARD",
            "BOSS_REWARD",
            "GRID",
            "HAND_SELECT",
        }:
            return None
        normalized = re.sub(r"[^a-z0-9]", "", label.lower())
        return normalized or None

    @staticmethod
    def _stable_id(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    @staticmethod
    def _map_coordinates(game: dict[str, Any]) -> tuple[int, int]:
        screen = dict(game.get("screen_state") or {})
        current = dict(screen.get("current_node") or {})
        if "x" not in current or "y" not in current:
            return -1, -1
        return int(current["x"]), int(current["y"])

    @staticmethod
    def _card_counts(cards: list[dict[str, Any]]) -> tuple[tuple[str, int], ...]:
        counts = Counter(str(card.get("id", "")) for card in cards)
        return tuple(sorted((card_id, count) for card_id, count in counts.items() if card_id))

    @staticmethod
    def _power_state(powers: list[dict[str, Any]]) -> tuple[tuple[str, int], ...]:
        normalized: list[tuple[str, int]] = []
        for power in powers:
            power_id = str(power.get("id", power.get("name", "")))
            if not power_id:
                continue
            normalized_id = {
                "anger": "ENRAGE",
                "confusion": "CONFUSED",
                "weakened": "WEAK",
            }.get(power_id.lower(), power_id)
            amount = int(power.get("amount", power.get("damage", 1)))
            if normalized_id == "CONFUSED" and amount < 0:
                amount = 1
            normalized.append((normalized_id, amount))
        return tuple(sorted(normalized))

    @staticmethod
    def _intent_damage(monster: dict[str, Any]) -> int:
        adjusted_damage = int(monster.get("move_adjusted_damage", -1))
        base_damage = int(monster.get("move_base_damage", -1))
        return max(0, adjusted_damage if adjusted_damage >= 0 else base_damage)

    @classmethod
    def _intent_hits(cls, monster: dict[str, Any]) -> int:
        if cls._intent_damage(monster) <= 0:
            return 0
        return max(0, int(monster.get("move_hits", 0)))

    @staticmethod
    def _seed_string(seed: int) -> str:
        try:
            from sts_env.lightspeed_backend import _load_lightspeed_module

            return str(_load_lightspeed_module().get_seed_str(seed))
        except (RuntimeError, ModuleNotFoundError):
            if seed == 0:
                return "0"
            raise RuntimeError("seed conversion requires the built sts_lightspeed extension")

    @staticmethod
    def _validate_seed(seed: int | None) -> int:
        if seed is None:
            raise ValueError("CommunicationBackend requires an explicit seed for validation")
        if seed < 0 or seed >= 2**64:
            raise ValueError("seed must be in [0, 2**64)")
        return seed

    @staticmethod
    def _require_matching_seed(state: dict[str, Any], seed: int) -> None:
        game = dict(state.get("game_state") or {})
        actual_seed = int(game.get("seed", -1))
        expected_signed_seed = seed if seed < 2**63 else seed - 2**64
        if actual_seed not in {seed, expected_signed_seed}:
            raise RuntimeError(
                f"real game seed {actual_seed} does not match expected seed {seed}"
            )
