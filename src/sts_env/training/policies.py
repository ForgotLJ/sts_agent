from __future__ import annotations

import random
from typing import Callable

from sts_env.env import StsEnv
from sts_env.types import Action, ActionKind, Observation, Phase


class RandomPolicy:
    def __init__(self, seed: int):
        self._random = random.Random(seed)

    def __call__(self, observation: Observation, _: int = 0) -> Action:
        if not observation.legal_actions:
            raise ValueError("cannot act in a terminal observation")
        return self._random.choice(observation.legal_actions)


class HeuristicPolicy:
    def __call__(self, observation: Observation, _: int = 0) -> Action:
        actions = observation.legal_actions
        if not actions:
            raise ValueError("cannot act in a terminal observation")
        if observation.phase is Phase.COMBAT:
            return self._combat_action(observation)
        if observation.phase is Phase.MAP:
            map_actions = [
                action for action in actions if action.kind is ActionKind.CHOOSE_MAP_NODE
            ]
            if map_actions:
                return max(map_actions, key=lambda action: self._map_score(observation, action))
        if observation.phase is Phase.REST_SITE:
            hp_ratio = observation.player.hp / max(1, observation.player.max_hp)
            rest_threshold = 0.9 if observation.act >= 3 else 0.85 if observation.act == 2 else 0.7
            preferred = "rest" if hp_ratio < rest_threshold else "smith"
            action = next((item for item in actions if item.option_type == preferred), None)
            if action is not None:
                return action
        if observation.phase is Phase.SHOP:
            shop_entry = next((item for item in actions if item.source_id == "shop"), None)
            if shop_entry is not None:
                if observation.history.recent_actions[-1:] == ("leave:leave",):
                    proceed = next(
                        (item for item in actions if item.source_id == "proceed"),
                        None,
                    )
                    if proceed is not None:
                        return proceed
                if observation.player.gold >= 75:
                    return shop_entry
                proceed = next((item for item in actions if item.source_id == "proceed"), None)
                if proceed is not None:
                    return proceed
            purchases = [
                action
                for action in actions
                if action.kind in {ActionKind.BUY, ActionKind.REMOVE_CARD}
            ]
            if purchases:
                return max(purchases, key=lambda action: self._shop_score(action))
        card_choices = [action for action in actions if action.kind is ActionKind.CHOOSE_CARD]
        if card_choices:
            best = max(
                card_choices,
                key=lambda action: self._card_choice_score(observation, action),
            )
            skip = next(
                (
                    action
                    for action in actions
                    if action.kind is ActionKind.LEAVE and action.option_type == "skip_card"
                ),
                None,
            )
            if skip is not None and self._card_choice_score(observation, best) <= 1:
                return skip
            return best
        boss_relics = [action for action in actions if action.option_type == "boss_relic"]
        if boss_relics:
            return max(boss_relics, key=self._boss_relic_score)
        reward_order = ("gold", "relic", "key", "potion", "card")
        for option_type in reward_order:
            action = next((item for item in actions if item.option_type == option_type), None)
            if action is not None:
                return action
        safe_options = [
            action
            for action in actions
            if action.gold_cost <= observation.player.gold
            and action.hp_cost < observation.player.hp
            and action.source_id != "proceed"
        ]
        if safe_options:
            return max(safe_options, key=self._option_score)
        proceed = next((item for item in actions if item.source_id == "proceed"), None)
        if proceed is not None:
            return proceed
        return actions[0]

    @staticmethod
    def _combat_action(observation: Observation) -> Action:
        actions = observation.legal_actions
        incoming = sum(
            enemy.intent_damage * max(1, enemy.intent_hits)
            for enemy in observation.enemies
            if enemy.hp > 0
        )
        playable_cards = [
            (
                action,
                next(card for card in observation.hand if card.instance_id == action.source_id),
            )
            for action in actions
            if action.kind is ActionKind.PLAY_CARD
        ]
        living = {enemy.enemy_id: enemy for enemy in observation.enemies if enemy.hp > 0}
        danger = incoming - observation.player.block
        potion = next(
            (action for action in actions if action.kind is ActionKind.USE_POTION),
            None,
        )
        if potion is not None and (
            danger >= observation.player.hp
            or danger >= max(10, observation.player.hp // 2)
            or observation.floor in {16, 33, 50}
            and observation.turn == 1
        ):
            return potion
        defenses = [
            (
                action,
                card,
                HeuristicPolicy._effective_block_value(observation, card.card_id, card.upgraded),
            )
            for action, card in playable_cards
            if HeuristicPolicy._effective_block_value(
                observation,
                card.card_id,
                card.upgraded,
            )
            > 0
        ]
        if danger >= max(4, observation.player.hp // 8) and defenses:
            return max(
                defenses,
                key=lambda item: (
                    min(item[2], danger) * 4.0
                    + item[2]
                    - max(0, item[1].cost) * 1.5,
                    item[2],
                    -max(0, item[1].cost),
                ),
            )[0]
        powers = [
            (action, card)
            for action, card in playable_cards
            if HeuristicPolicy._card_id(card.card_id) in HeuristicPolicy._POWER_PRIORITIES
        ]
        if powers and danger <= max(8, observation.player.hp // 5):
            return max(
                powers,
                key=lambda item: (
                    HeuristicPolicy._POWER_PRIORITIES[
                        HeuristicPolicy._card_id(item[1].card_id)
                    ],
                    -max(0, item[1].cost),
                ),
            )[0]
        utility = [
            (action, card)
            for action, card in playable_cards
            if HeuristicPolicy._card_id(card.card_id) in HeuristicPolicy._UTILITY_PRIORITIES
            and not (
                HeuristicPolicy._card_id(card.card_id) == "offering"
                and observation.player.hp <= 10
            )
        ]
        if utility:
            return max(
                utility,
                key=lambda item: (
                    HeuristicPolicy._UTILITY_PRIORITIES[
                        HeuristicPolicy._card_id(item[1].card_id)
                    ]
                    + HeuristicPolicy._target_priority(living.get(item[0].target_id))
                    + HeuristicPolicy._target_threat(living.get(item[0].target_id)),
                    -max(0, item[1].cost),
                ),
            )[0]
        attacks = [
            (
                action,
                card,
                HeuristicPolicy._damage_value(observation, card.card_id, card.upgraded),
            )
            for action, card in playable_cards
            if HeuristicPolicy._damage_value(observation, card.card_id, card.upgraded) > 0
        ]
        if attacks:
            champion = next(
                (
                    enemy
                    for enemy in living.values()
                    if HeuristicPolicy._card_id(enemy.monster_id) == "thechamp"
                ),
                None,
            )
            strength = next(
                (
                    amount
                    for name, amount in observation.player.statuses
                    if name.lower() == "strength"
                ),
                0,
            )
            if (
                champion is not None
                and 0.5 < champion.hp / max(1, champion.max_hp) <= 0.7
                and strength < 8
                and observation.turn < 12
            ):
                return next(
                    (action for action in actions if action.kind is ActionKind.END_TURN),
                    actions[0],
                )

            def attack_score(item: tuple[Action, object, float]) -> tuple[float, ...]:
                action, card, damage = item
                target = living.get(action.target_id)
                threat = (
                    target.intent_damage * max(1, target.intent_hits) / 20.0
                    if target is not None
                    else 0.0
                )
                vulnerable_bonus = 0.0
                if "bash" in HeuristicPolicy._card_id(card.card_id):
                    vulnerable = target is not None and any(
                        name.lower() == "vulnerable" and value > 0
                        for name, value in target.statuses
                    )
                    vulnerable_bonus = 5.0 if not vulnerable else 0.0
                if target is not None and any(
                    name.lower() == "vulnerable" and value > 0
                    for name, value in target.statuses
                ):
                    damage *= 1.5
                normalized = HeuristicPolicy._card_id(card.card_id)
                is_aoe = normalized in HeuristicPolicy._AOE_ATTACKS
                lethal_count = (
                    sum(damage >= enemy.hp for enemy in living.values())
                    if is_aoe
                    else int(target is not None and damage >= target.hp)
                )
                total_damage = damage * len(living) if is_aoe else damage
                healing_bonus = (
                    8.0
                    if normalized in {"bite", "reaper"}
                    and observation.player.hp * 2 < observation.player.max_hp
                    else 0.0
                )
                return (
                    float(lethal_count),
                    float(is_aoe) * len(living),
                    total_damage / max(1, max(0, card.cost)),
                    damage
                    + threat
                    + vulnerable_bonus
                    + healing_bonus
                    + HeuristicPolicy._target_priority(target),
                    -float(target.hp if target is not None else 0),
                    -max(0, card.cost),
                )

            return max(attacks, key=attack_score)[0]
        card_selection = next(
            (action for action in actions if action.kind is ActionKind.CHOOSE_CARD),
            None,
        )
        if card_selection is not None:
            return card_selection
        return next(
            (action for action in actions if action.kind is ActionKind.END_TURN),
            actions[0],
        )

    _BLOCK_VALUES = {
        "armaments": 5,
        "defend": 5,
        "defendr": 5,
        "flamebarrier": 12,
        "ghostlyarmor": 10,
        "impervious": 30,
        "ironwave": 5,
        "panicbutton": 30,
        "powerthrough": 15,
        "shrugitoff": 8,
        "truegrit": 7,
    }
    _DAMAGE_VALUES = {
        "anger": 6,
        "bash": 8,
        "bite": 7,
        "bloodforblood": 18,
        "bludgeon": 32,
        "bodyslam": 0,
        "carnage": 20,
        "clash": 14,
        "cleave": 8,
        "clothesline": 12,
        "dropkick": 5,
        "feed": 10,
        "fiendfire": 20,
        "headbutt": 9,
        "heavyblade": 14,
        "hemokinesis": 15,
        "immolate": 21,
        "ironwave": 5,
        "perfectedstrike": 10,
        "pommelstrike": 9,
        "pummel": 8,
        "rampage": 8,
        "reaper": 4,
        "recklesscharge": 7,
        "searingblow": 12,
        "sever soul": 16,
        "seversoul": 16,
        "strike": 6,
        "striker": 6,
        "swordboomerang": 9,
        "thunderclap": 4,
        "twinstrike": 10,
        "uppercut": 13,
        "wildstrike": 12,
    }
    _POWER_PRIORITIES = {
        "barricade": 4.0,
        "combust": 3.0,
        "demonform": 5.0,
        "evolve": 2.0,
        "feelno pain": 3.0,
        "feelnopain": 3.0,
        "firebreathing": 2.0,
        "inflame": 5.0,
        "juggernaut": 3.0,
        "metallicize": 4.0,
        "rupture": 2.0,
    }
    _UTILITY_PRIORITIES = {
        "battletrance": 5.0,
        "bloodletting": 2.0,
        "disarm": 6.0,
        "dualwield": 2.0,
        "entrench": 3.0,
        "flex": 2.0,
        "intimidate": 4.0,
        "offering": 5.0,
        "seeingred": 3.0,
        "shockwave": 5.0,
        "spotweakness": 4.0,
        "warcry": 2.0,
    }
    _AOE_ATTACKS = {
        "cleave",
        "immolate",
        "reaper",
        "thunderclap",
        "whirlwind",
    }
    _TARGET_PRIORITIES = {
        "bronzeorb": 8.0,
        "byrd": 5.0,
        "chosen": 6.0,
        "cultist": 8.0,
        "dagger": 9.0,
        "deca": 8.0,
        "donu": 10.0,
        "exploder": 12.0,
        "looter": 6.0,
        "mugger": 6.0,
        "mystic": 10.0,
        "orbwalker": 8.0,
        "redslaver": 7.0,
        "reptomancer": 10.0,
        "repulsor": 6.0,
        "shelledparasite": 5.0,
        "snakeplant": 6.0,
        "spiker": -2.0,
        "sphericguardian": 5.0,
        "taskmaster": 8.0,
        "torchhead": 8.0,
    }

    @classmethod
    def _block_value(cls, card_id: str, upgraded: bool) -> float:
        value = cls._BLOCK_VALUES.get(cls._card_id(card_id), 0)
        return float(value + (3 if upgraded and value > 0 else 0))

    @classmethod
    def _effective_block_value(
        cls,
        observation: Observation,
        card_id: str,
        upgraded: bool,
    ) -> float:
        value = cls._block_value(card_id, upgraded)
        if value <= 0:
            return 0.0
        dexterity = next(
            (
                amount
                for name, amount in observation.player.statuses
                if name.lower() == "dexterity"
            ),
            0,
        )
        value = max(0.0, value + dexterity)
        frail = any(
            name.lower() == "frail" and amount > 0
            for name, amount in observation.player.statuses
        )
        return float(int(value * 0.75) if frail else value)

    @classmethod
    def _damage_value(
        cls,
        observation: Observation,
        card_id: str,
        upgraded: bool,
    ) -> float:
        normalized = cls._card_id(card_id)
        if normalized == "bodyslam":
            value = observation.player.block
        elif normalized == "whirlwind":
            value = observation.player.energy * (8 if upgraded else 5)
        elif normalized == "perfectedstrike":
            strike_count = sum(
                count
                for deck_card_id, count in observation.deck
                if "strike" in cls._card_id(deck_card_id)
            )
            value = (6 if not upgraded else 6) + strike_count * (2 if not upgraded else 3)
        else:
            value = cls._DAMAGE_VALUES.get(normalized, 0)
        if upgraded and value > 0:
            value += 3
        strength = next(
            (value for name, value in observation.player.statuses if name.lower() == "strength"),
            0,
        )
        weak = any(
            name.lower() == "weak" and value > 0
            for name, value in observation.player.statuses
        )
        return max(0.0, (value + strength) * (0.75 if weak else 1.0))

    @staticmethod
    def _card_id(card_id: str) -> str:
        return "".join(character for character in card_id.lower() if character.isalnum())

    @classmethod
    def _target_priority(cls, enemy: object | None) -> float:
        if enemy is None:
            return 0.0
        return cls._TARGET_PRIORITIES.get(cls._card_id(enemy.monster_id), 0.0)

    @staticmethod
    def _target_threat(enemy: object | None) -> float:
        if enemy is None:
            return 0.0
        return enemy.intent_damage * max(1, enemy.intent_hits) / 5.0

    @classmethod
    def _map_score(cls, observation: Observation, action: Action) -> float:
        hp_ratio = observation.player.hp / max(1, observation.player.max_hp)
        if observation.act >= 3 and hp_ratio < 0.6:
            scores = {"R": 11.0, "?": 6.0, "$": 2.0, "M": -6.0, "T": 4.0, "E": -16.0}
        elif observation.act >= 3 and hp_ratio < 0.9:
            scores = {"R": 9.0, "?": 6.0, "$": 2.0, "M": -3.0, "T": 4.0, "E": -14.0}
        elif observation.act >= 3:
            scores = {"R": 4.0, "?": 5.0, "$": 2.0, "M": 0.0, "T": 4.0, "E": -12.0}
        elif observation.act >= 2 and hp_ratio < 0.55:
            scores = {"R": 10.0, "?": 5.0, "$": 2.0, "M": -4.0, "T": 4.0, "E": -14.0}
        elif observation.act >= 2 and hp_ratio < 0.85:
            scores = {"R": 8.0, "?": 5.0, "$": 2.0, "M": 0.0, "T": 4.0, "E": -10.0}
        elif observation.act >= 2:
            scores = {"R": 3.0, "?": 4.0, "$": 2.0, "M": 1.0, "T": 4.0, "E": -8.0}
        elif hp_ratio < 0.45:
            scores = {"R": 8.0, "?": 4.0, "$": 2.0, "M": 0.0, "T": 3.0, "E": -10.0}
        elif hp_ratio > 0.8:
            scores = {"E": -2.0, "?": 3.0, "M": 3.0, "T": 4.0, "R": 2.0, "$": 2.0}
        else:
            scores = {"R": 5.0, "?": 3.0, "M": 3.0, "T": 4.0, "$": 2.0, "E": -6.0}
        symbol = action.option_type.upper()
        score = scores.get(symbol, 0.0)
        if symbol == "$" and observation.player.gold < 75:
            score -= 3.0
        nodes = {(node.x, node.y): node for node in observation.map_nodes}
        target = (action.target_x, action.target_y)
        if target in nodes:
            target_node = nodes[target]
            cache: dict[tuple[int, int], float] = {}
            score += 0.75 * max(
                (
                    cls._future_map_score(child, nodes, scores, cache)
                    for child in target_node.children
                    if child in nodes
                ),
                default=0.0,
            )
        return score - 0.001 * max(0, action.target_x)

    @classmethod
    def _future_map_score(
        cls,
        coordinate: tuple[int, int],
        nodes: dict[tuple[int, int], object],
        scores: dict[str, float],
        cache: dict[tuple[int, int], float],
    ) -> float:
        if coordinate in cache:
            return cache[coordinate]
        node = nodes[coordinate]
        child_scores = [
            cls._future_map_score(child, nodes, scores, cache)
            for child in node.children
            if child in nodes
        ]
        future = max(child_scores, default=0.0)
        value = scores.get(node.symbol.upper(), 0.0) + 0.75 * future
        cache[coordinate] = value
        return value

    @classmethod
    def _shop_score(cls, action: Action) -> float:
        priorities = {
            "relic": 5.0,
            "remove_card": 4.0,
            "card": 3.0,
            "potion": 2.0,
        }
        card_bonus = cls._card_action_score(action) if action.option_type == "card" else 0.0
        return priorities.get(action.option_type, 0.0) + card_bonus - action.gold_cost / 1000.0

    @classmethod
    def _boss_relic_score(cls, action: Action) -> float:
        relic_id = cls._card_id(str(action.source_id))
        priorities = {
            "astrolabe": 4.0,
            "blackblood": 5.0,
            "blackstar": 3.0,
            "bustedcrown": -4.0,
            "callingbell": 4.0,
            "coffeedripper": 4.0,
            "cursedkey": 6.0,
            "ectoplasm": -3.0,
            "emptycage": 3.0,
            "fusionhammer": 5.0,
            "markofpain": 2.0,
            "pandorasbox": 4.0,
            "philosophersstone": -2.0,
            "runicdome": -6.0,
            "runicpyramid": 5.0,
            "sacredbark": 3.0,
            "slaverscollar": 4.0,
            "sneckoeye": 2.0,
            "sozu": -1.0,
            "tinyhouse": 1.0,
            "velvetchoker": 2.0,
        }
        return priorities.get(relic_id, 0.0)

    @staticmethod
    def _option_score(action: Action) -> float:
        text = f"{action.label} {action.description}".lower()
        score = -action.hp_cost / 10.0 - action.gold_cost / 100.0
        rewards = {
            "rare relic": 9.0,
            "boss relic": 8.0,
            "common relic": 7.0,
            "remove two": 7.0,
            "transform two": 6.0,
            "250 gold": 6.0,
            "remove a card": 5.0,
            "upgrade a card": 5.0,
            "neow's lament": 5.0,
            "max hp +20": 7.0,
            "max hp +10": 4.0,
            "100 gold": 4.0,
            "transform a card": 4.0,
            "rare card": 4.0,
            "three potions": 3.0,
            "colorless card": 2.0,
            "card to obtain": 2.0,
            "heal": 4.0,
        }
        drawbacks = {
            "curse": -9.0,
            "take 30%": -7.0,
            "max hp -10": -5.0,
            "lose your starter relic": -5.0,
            "lose all gold": -4.0,
        }
        score += sum(value for phrase, value in rewards.items() if phrase in text)
        score += sum(value for phrase, value in drawbacks.items() if phrase in text)
        return score

    @staticmethod
    def _card_action_score(action: Action) -> float:
        card_id = "".join(
            character for character in str(action.source_id).lower() if character.isalnum()
        )
        strong = {
            "anger": 4.0,
            "armaments": 4.0,
            "barricade": 4.0,
            "battletrance": 5.0,
            "bodySlam".lower(): 2.0,
            "carnage": 5.0,
            "cleave": 4.0,
            "clothesline": 3.0,
            "combust": 3.0,
            "corruption": 6.0,
            "darkembrace": 5.0,
            "demonform": 4.0,
            "disarm": 5.0,
            "entrench": 4.0,
            "feelno pain": 5.0,
            "feelnopain": 5.0,
            "fiendfire": 5.0,
            "flamebarrier": 5.0,
            "headbutt": 4.0,
            "hemokinesis": 4.0,
            "impervious": 6.0,
            "inflame": 5.0,
            "ironwave": 4.0,
            "limitbreak": 5.0,
            "metallicize": 5.0,
            "offering": 6.0,
            "perfectedstrike": 4.0,
            "pommelstrike": 4.0,
            "powerthrough": 4.0,
            "reaper": 6.0,
            "secondwind": 5.0,
            "shockwave": 5.0,
            "shrugitoff": 4.0,
            "spotweakness": 4.0,
            "swordboomerang": 3.0,
            "truegrit": 3.0,
            "twinstrike": 3.0,
            "uppercut": 5.0,
            "wildstrike": 2.0,
        }
        weak = {
            "clash": -2.0,
            "flex": -1.0,
            "havoc": -1.0,
            "strike": -3.0,
            "striker": -3.0,
            "defend": -1.0,
            "defendr": -1.0,
        }
        return strong.get(card_id, weak.get(card_id, 1.0))

    @classmethod
    def _card_choice_score(cls, observation: Observation, action: Action) -> float:
        base = cls._card_action_score(action)
        card_id = cls._card_id(str(action.source_id))
        deck_size = sum(count for _, count in observation.deck)
        copies = sum(
            count
            for deck_card_id, count in observation.deck
            if cls._card_id(deck_card_id) == card_id
        )
        size_penalty = max(0, deck_size - 18) * 0.25
        if base >= 5:
            size_penalty *= 0.5
        duplicate_penalty = copies * (0.5 if base >= 5 else 0.75)
        return base - size_penalty - duplicate_penalty


class OneStepSearchPolicy:
    def __init__(self, evaluator: Callable[[Observation], float] | None = None):
        self._evaluator = evaluator or self._default_evaluator

    def select(self, environment: StsEnv) -> Action:
        observation = environment.observation
        if not observation.legal_actions:
            raise ValueError("cannot search a terminal observation")
        scored: list[tuple[float, int, Action]] = []
        for action_index, action in enumerate(observation.legal_actions):
            branch = environment.clone()
            next_observation, reward, terminated, truncated, _ = branch.step(action)
            score = reward * 100.0 + self._evaluator(next_observation)
            if truncated:
                score -= 100.0
            if terminated and reward < 0:
                score -= 100.0
            scored.append((score, -action_index, action))
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def _default_evaluator(observation: Observation) -> float:
        enemy_hp = sum(max(0, enemy.hp) for enemy in observation.enemies)
        incoming = sum(
            enemy.intent_damage * max(1, enemy.intent_hits)
            for enemy in observation.enemies
            if enemy.hp > 0
        )
        return (
            observation.player.hp
            + 0.5 * observation.player.block
            - 1.5 * enemy_hp
            - 0.25 * incoming
        )
