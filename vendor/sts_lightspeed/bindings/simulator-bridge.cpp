#include "simulator-bridge.h"

#include <algorithm>
#include <array>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/stl.h>

#include "constants/Cards.h"
#include "constants/MonsterEncounters.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/Potions.h"
#include "constants/Relics.h"
#include "game/Neow.h"
#include "sim/search/Action.h"
#include "sim/search/GameAction.h"

namespace py = pybind11;

namespace {

    std::uint64_t splitMix64(std::uint64_t value) {
        value += 0x9E3779B97F4A7C15ULL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }

    sts::CardId cardIdFromString(const std::string &value) {
        for (int index = 0; index <= static_cast<int>(sts::CardId::ZAP); ++index) {
            if (value == sts::cardStringIds[index]) {
                return static_cast<sts::CardId>(index);
            }
        }
        throw std::invalid_argument("unknown card id in draw-order constraint: " + value);
    }

    std::string gameOutcomeName(sts::GameOutcome outcome) {
        switch (outcome) {
            case sts::GameOutcome::PLAYER_LOSS:
                return "player_loss";
            case sts::GameOutcome::PLAYER_VICTORY:
                return "player_victory";
            case sts::GameOutcome::UNDECIDED:
            default:
                return "undecided";
        }
    }

    std::string screenStateName(sts::ScreenState state) {
        switch (state) {
            case sts::ScreenState::EVENT_SCREEN:
                return "event";
            case sts::ScreenState::REWARDS:
                return "rewards";
            case sts::ScreenState::BOSS_RELIC_REWARDS:
                return "boss_relic_rewards";
            case sts::ScreenState::CARD_SELECT:
                return "card_select";
            case sts::ScreenState::MAP_SCREEN:
                return "map";
            case sts::ScreenState::TREASURE_ROOM:
                return "treasure";
            case sts::ScreenState::REST_ROOM:
                return "rest";
            case sts::ScreenState::SHOP_ROOM:
                return "shop";
            case sts::ScreenState::BATTLE:
                return "battle";
            case sts::ScreenState::INVALID:
            default:
                return "invalid";
        }
    }

    std::string phaseName(const sts::GameContext &game, bool inBattle) {
        if (game.outcome != sts::GameOutcome::UNDECIDED) {
            return "terminal";
        }
        if (inBattle) {
            return "combat";
        }
        switch (game.screenState) {
            case sts::ScreenState::REWARDS:
            case sts::ScreenState::BOSS_RELIC_REWARDS:
            case sts::ScreenState::CARD_SELECT:
                return "card_reward";
            case sts::ScreenState::MAP_SCREEN:
                return "map";
            case sts::ScreenState::SHOP_ROOM:
                return "shop";
            case sts::ScreenState::REST_ROOM:
                return "rest_site";
            case sts::ScreenState::EVENT_SCREEN:
            case sts::ScreenState::TREASURE_ROOM:
            case sts::ScreenState::INVALID:
            default:
                return "event";
        }
    }

    template<typename Iterator>
    py::list cardCounts(Iterator begin, Iterator end) {
        std::map<std::string, int> counts;
        for (auto it = begin; it != end; ++it) {
            ++counts[sts::getCardStringId(it->getId())];
        }

        py::list result;
        for (const auto &[cardId, count] : counts) {
            result.append(py::make_tuple(cardId, count));
        }
        return result;
    }

    py::list playerStatuses(const sts::Player &player) {
        py::list result;
        for (int index = static_cast<int>(PS::INVALID) + 1;
             index <= static_cast<int>(PS::THE_BOMB);
             ++index) {
            const auto status = static_cast<PlayerStatus>(index);
            if (!player.hasStatusRuntime(status)) {
                continue;
            }

            int value = 1;
            if (status == PS::ARTIFACT) {
                value = player.artifact;
            } else if (status == PS::DEXTERITY) {
                value = player.dexterity;
            } else if (status == PS::FOCUS) {
                value = player.focus;
            } else if (status == PS::STRENGTH) {
                value = player.strength;
            } else {
                const auto found = player.statusMap.find(status);
                if (found != player.statusMap.end()) {
                    value = found->second;
                }
            }

            if (value != 0) {
                result.append(py::make_tuple(playerStatusEnumStrings[index], value));
            }
        }
        return result;
    }

    py::list monsterStatuses(const sts::Monster &monster) {
        py::list result;
        for (int index = 0; index < static_cast<int>(sts::MS::INVALID); ++index) {
            const auto status = static_cast<sts::MonsterStatus>(index);
            const int value = monster.getStatusInternal(status);
            if (value != 0) {
                result.append(py::make_tuple(sts::enemyStatusStrings[index], value));
            }
        }
        return result;
    }

    py::list relicState(const sts::GameContext &game) {
        py::list result;
        for (const auto &relic : game.relics.relics) {
            const auto index = static_cast<int>(relic.id);
            result.append(py::make_tuple(sts::relicIds[index], relic.data));
        }
        return result;
    }

    py::list potionState(const sts::GameContext &game, const std::optional<sts::BattleContext> &battle) {
        py::list result;
        const int capacity = battle ? battle->potionCapacity : game.potionCapacity;
        for (int slot = 0; slot < capacity; ++slot) {
            const auto potion = battle ? battle->potions[slot] : game.potions[slot];
            result.append(py::make_tuple(slot, sts::potionIds[static_cast<int>(potion)]));
        }
        return result;
    }

    py::list mapState(const sts::GameContext &game) {
        py::list result;
        if (!game.map) {
            return result;
        }
        for (int y = 0; y < static_cast<int>(game.map->nodes.size()); ++y) {
            for (int x = 0; x < static_cast<int>(game.map->nodes[y].size()); ++x) {
                const auto &node = game.map->nodes[y][x];
                if (node.edgeCount <= 0) {
                    continue;
                }
                py::dict nodeState;
                nodeState["x"] = node.x;
                nodeState["y"] = node.y;
                nodeState["symbol"] = std::string(1, node.getRoomSymbol());
                nodeState["burning_elite"] =
                    game.map->burningEliteX == node.x && game.map->burningEliteY == node.y;
                py::list children;
                for (int edgeIndex = 0; edgeIndex < node.edgeCount; ++edgeIndex) {
                    children.append(py::make_tuple(
                        node.edges[edgeIndex],
                        node.y == 14 ? 16 : node.y + 1));
                }
                nodeState["children"] = children;
                result.append(nodeState);
            }
        }
        return result;
    }

    int publicCardInstanceId(const sts::CardInstance &card, int handIndex) {
        return (static_cast<int>(card.getUniqueId()) + 1) * 16 + handIndex;
    }

    std::string battleActionDescription(const sts::search::Action &action, const sts::BattleContext &battle) {
        std::ostringstream stream;
        action.printDesc(stream, battle);
        return stream.str();
    }

    std::string gameActionDescription(const sts::search::GameAction &action, const sts::GameContext &game) {
        std::ostringstream stream;
        action.printDesc(stream, game);
        return stream.str();
    }

    py::dict battleActionState(const sts::search::Action &action, const sts::BattleContext &battle) {
        py::dict result;
        result["domain"] = "combat";
        result["token"] = action.bits;
        result["action_type"] = static_cast<int>(action.getActionType());
        result["source_index"] = action.getSourceIdx();
        result["target_index"] = action.getTargetIdx() > 5 ? -1 : action.getTargetIdx();
        result["choice_index"] = action.getSelectIdx();
        result["label"] = battleActionDescription(action, battle);

        if (action.getActionType() == sts::search::ActionType::CARD) {
            result["source_instance_id"] = publicCardInstanceId(
                battle.cards.hand[action.getSourceIdx()], action.getSourceIdx());
        }
        if (action.getActionType() == sts::search::ActionType::POTION) {
            const auto potion = battle.potions[action.getSourceIdx()];
            result["potion_id"] = sts::potionIds[static_cast<int>(potion)];
            result["discard"] = action.getTargetIdx() > 5;
        }
        if (action.getActionType() == sts::search::ActionType::MULTI_CARD_SELECT) {
            py::list selected;
            for (const auto index : action.getSelectedIdxs()) {
                selected.append(index);
            }
            result["selected_indices"] = selected;
        }
        if (action.getActionType() == sts::search::ActionType::SINGLE_CARD_SELECT) {
            const int index = action.getSelectIdx();
            const sts::CardInstance *selectedCard = nullptr;
            switch (battle.cardSelectInfo.cardSelectTask) {
                case sts::CardSelectTask::SECRET_TECHNIQUE:
                case sts::CardSelectTask::SECRET_WEAPON:
                case sts::CardSelectTask::SEEK:
                    if (index >= 0 && index < battle.cards.drawPile.size()) {
                        selectedCard = &battle.cards.drawPile[index];
                    }
                    break;
                default:
                    break;
            }
            if (selectedCard != nullptr) {
                result["selected_card_id"] = sts::getCardStringId(selectedCard->getId());
                result["selected_card_name"] = selectedCard->getName();
                result["selected_card_instance_id"] = static_cast<int>(selectedCard->getUniqueId());
            }
        }
        return result;
    }

    py::dict gameActionState(const sts::search::GameAction &action, const sts::GameContext &game) {
        py::dict result;
        result["domain"] = "game";
        result["token"] = action.bits;
        result["idx1"] = action.getIdx1();
        result["idx2"] = action.getIdx2();
        result["idx3"] = action.getIdx3();
        result["label"] = gameActionDescription(action, game);
        result["screen_state"] = screenStateName(game.screenState);
        result["potion_action"] = action.isPotionAction();
        result["potion_discard"] = action.isPotionDiscard();
        if (action.isPotionAction() && action.getIdx1() < game.potionCapacity) {
            result["potion_id"] = sts::potionIds[static_cast<int>(game.potions[action.getIdx1()])];
        }
        if (game.screenState == sts::ScreenState::EVENT_SCREEN &&
            game.curEvent == sts::Event::NEOW && action.getIdx1() < game.info.neowRewards.size()) {
            const auto &option = game.info.neowRewards[action.getIdx1()];
            const auto bonusIndex = static_cast<int>(option.r);
            const auto drawbackIndex = static_cast<int>(option.d);
            const std::string bonus = sts::Neow::bonusStrings[bonusIndex];
            const std::string drawback = sts::Neow::drawbackStrings[drawbackIndex];
            result["label"] = drawback.empty() ? bonus : bonus + " " + drawback;
            result["description"] = result["label"];
            result["option_type"] = "neow";
            result["item_id"] = "neow_" + std::to_string(bonusIndex) + "_" + std::to_string(drawbackIndex);
            result["neow_bonus"] = bonusIndex;
            result["neow_drawback"] = drawbackIndex;
        }
        if (game.screenState == sts::ScreenState::REWARDS || game.screenState == sts::ScreenState::SHOP_ROOM) {
            const auto rewardType = action.getRewardsActionType();
            result["reward_type"] = static_cast<int>(rewardType);
            if (game.screenState == sts::ScreenState::REWARDS) {
                const auto &rewards = game.info.rewardsContainer;
                if (rewardType == sts::search::GameAction::RewardsActionType::CARD &&
                    action.getIdx1() < rewards.cardRewardCount &&
                    action.getIdx2() < rewards.cardRewards[action.getIdx1()].size()) {
                    const auto card = rewards.cardRewards[action.getIdx1()][action.getIdx2()];
                    result["item_id"] = sts::getCardStringId(card.getId());
                    result["item_name"] = card.getName();
                } else if (rewardType == sts::search::GameAction::RewardsActionType::GOLD &&
                           action.getIdx1() < rewards.goldRewardCount) {
                    result["item_id"] = "gold";
                    result["item_name"] = "gold";
                    result["amount"] = rewards.gold[action.getIdx1()];
                } else if (rewardType == sts::search::GameAction::RewardsActionType::POTION &&
                           action.getIdx1() < rewards.potionCount) {
                    result["item_id"] = sts::potionIds[static_cast<int>(rewards.potions[action.getIdx1()])];
                    result["item_name"] = result["item_id"];
                } else if (rewardType == sts::search::GameAction::RewardsActionType::RELIC &&
                           action.getIdx1() < rewards.relicCount) {
                    result["item_id"] = sts::relicIds[static_cast<int>(rewards.relics[action.getIdx1()])];
                    result["item_name"] = result["item_id"];
                } else if (rewardType == sts::search::GameAction::RewardsActionType::KEY) {
                    result["item_id"] = rewards.sapphireKey ? "sapphire_key" : "emerald_key";
                    result["item_name"] = result["item_id"];
                }
            }
            if (game.screenState == sts::ScreenState::SHOP_ROOM) {
                const auto &shop = game.info.shop;
                if (rewardType == sts::search::GameAction::RewardsActionType::CARD && action.getIdx1() < 7) {
                    const auto card = shop.cards[action.getIdx1()];
                    result["item_id"] = sts::getCardStringId(card.getId());
                    result["item_name"] = card.getName();
                    result["price"] = shop.cardPrice(action.getIdx1());
                } else if (rewardType == sts::search::GameAction::RewardsActionType::POTION && action.getIdx1() < 3) {
                    result["item_id"] = sts::potionIds[static_cast<int>(shop.potions[action.getIdx1()])];
                    result["item_name"] = result["item_id"];
                    result["price"] = shop.potionPrice(action.getIdx1());
                } else if (rewardType == sts::search::GameAction::RewardsActionType::RELIC && action.getIdx1() < 3) {
                    result["item_id"] = sts::relicIds[static_cast<int>(shop.relics[action.getIdx1()])];
                    result["item_name"] = result["item_id"];
                    result["price"] = shop.relicPrice(action.getIdx1());
                } else if (rewardType == sts::search::GameAction::RewardsActionType::CARD_REMOVE) {
                    result["item_id"] = "purge";
                    result["item_name"] = "purge";
                    result["price"] = shop.removeCost;
                }
            }
        }
        if (game.screenState == sts::ScreenState::MAP_SCREEN && game.map) {
            const int targetY = game.curMapNodeY == 14 ? 16 : game.curMapNodeY + 1;
            result["target_x"] = action.getIdx1();
            result["target_y"] = targetY;
            if (targetY < 15) {
                const auto &node = game.map->getNode(action.getIdx1(), targetY);
                result["room_symbol"] = std::string(1, node.getRoomSymbol());
                result["burning_elite"] =
                    game.map->burningEliteX == node.x && game.map->burningEliteY == node.y;
            } else {
                result["room_symbol"] = "B";
                result["burning_elite"] = false;
            }
        }
        if (game.screenState == sts::ScreenState::BOSS_RELIC_REWARDS) {
            if (action.getIdx1() < 3) {
                const auto relic = game.info.bossRelics[action.getIdx1()];
                result["item_id"] = sts::relicIds[static_cast<int>(relic)];
                result["item_name"] = result["item_id"];
            } else {
                result["item_id"] = "skip";
                result["item_name"] = "skip";
            }
        }
        if (game.screenState == sts::ScreenState::CARD_SELECT &&
            action.getIdx1() < game.info.toSelectCards.size()) {
            const auto &card = game.info.toSelectCards[action.getIdx1()].card;
            result["item_id"] = sts::getCardStringId(card.getId());
            result["item_name"] = card.getName();
        }
        if (game.screenState == sts::ScreenState::REST_ROOM) {
            static constexpr const char *restOptions[] = {
                "rest", "smith", "recall", "lift", "toke", "dig", "leave"
            };
            if (action.getIdx1() < 7) {
                result["option_type"] = restOptions[action.getIdx1()];
            }
        }
        return result;
    }

}

sts::py::SimulatorBridge::SimulatorBridge(
    std::uint64_t seed,
    int ascension,
    int neowMode,
    int actOneBossesSeen,
    bool finalActUnlocked)
    : game(
        sts::CharacterClass::IRONCLAD,
        seed,
        ascension,
        actOneBossesSeen,
        finalActUnlocked),
      neowMode(neowMode) {
    if (neowMode < 0 || neowMode > 2) {
        throw std::invalid_argument("neow mode must be full (0), limited (1), or skipped (2)");
    }
    configureNeow();
    syncBattleState();
}

sts::py::SimulatorBridge::SimulatorBridge(const SimulatorBridge &other)
    : game(other.game), battle(other.battle), neowMode(other.neowMode) {
    if (other.game.map) {
        game.map = std::make_shared<sts::Map>(*other.game.map);
    }
}

std::unique_ptr<sts::py::SimulatorBridge> sts::py::SimulatorBridge::clone() const {
    return std::make_unique<SimulatorBridge>(*this);
}

std::unique_ptr<sts::py::SimulatorBridge> sts::py::SimulatorBridge::redeterminizedClone(
    std::uint64_t searchSeed,
    const std::vector<std::string> &knownTop,
    const std::vector<std::string> &knownBottom) const {
    auto result = std::make_unique<SimulatorBridge>(*this);
    result->redeterminizeCombat(searchSeed, knownTop, knownBottom);
    return result;
}

void sts::py::SimulatorBridge::redeterminizeCombat(
    std::uint64_t searchSeed,
    const std::vector<std::string> &knownTop,
    const std::vector<std::string> &knownBottom) {
    if (battle) {
        std::vector<sts::CardInstance> remaining(
            battle->cards.drawPile.begin(), battle->cards.drawPile.end());
        auto takeCard = [&remaining](const std::string &cardId) {
            const auto expected = cardIdFromString(cardId);
            const auto found = std::find_if(
                remaining.begin(), remaining.end(),
                [expected](const sts::CardInstance &card) { return card.getId() == expected; });
            if (found == remaining.end()) {
                throw std::invalid_argument(
                    "draw-order constraint card is absent from the draw pile: " + cardId);
            }
            const auto card = *found;
            remaining.erase(found);
            return card;
        };

        std::vector<sts::CardInstance> bottomCards;
        std::vector<sts::CardInstance> topCards;
        bottomCards.reserve(knownBottom.size());
        topCards.reserve(knownTop.size());
        for (const auto &cardId : knownBottom) {
            bottomCards.push_back(takeCard(cardId));
        }
        for (const auto &cardId : knownTop) {
            topCards.push_back(takeCard(cardId));
        }
        java::Collections::shuffle(
            remaining.begin(), remaining.end(), java::Random(splitMix64(searchSeed)));

        battle->cards.drawPile.clear();
        for (const auto &card : bottomCards) {
            battle->cards.drawPile.push_back(card);
        }
        for (const auto &card : remaining) {
            battle->cards.drawPile.push_back(card);
        }
        for (auto iterator = topCards.rbegin(); iterator != topCards.rend(); ++iterator) {
            battle->cards.drawPile.push_back(*iterator);
        }

        std::array<sts::Random *, 6> battleRngs = {
            &battle->aiRng,
            &battle->cardRandomRng,
            &battle->miscRng,
            &battle->monsterHpRng,
            &battle->potionRng,
            &battle->shuffleRng,
        };
        for (std::size_t index = 0; index < battleRngs.size(); ++index) {
            *battleRngs[index] = sts::Random(splitMix64(searchSeed + index + 1));
        }
    } else if (!knownTop.empty() || !knownBottom.empty()) {
        throw std::invalid_argument(
            "draw-order constraints require a combat state");
    }

    std::array<sts::Random *, 13> gameRngs = {
        &game.aiRng,
        &game.cardRandomRng,
        &game.cardRng,
        &game.eventRng,
        &game.mathUtilRng,
        &game.merchantRng,
        &game.miscRng,
        &game.monsterHpRng,
        &game.monsterRng,
        &game.potionRng,
        &game.relicRng,
        &game.shuffleRng,
        &game.treasureRng,
    };
    for (std::size_t index = 0; index < gameRngs.size(); ++index) {
        *gameRngs[index] = sts::Random(splitMix64(searchSeed + 0x100 + index));
    }
}

pybind11::dict sts::py::SimulatorBridge::observe() const {
    pybind11::dict result;
    const bool inBattle = battle.has_value();
    result["phase"] = phaseName(game, inBattle);
    result["screen_state"] = screenStateName(game.screenState);
    result["outcome"] = gameOutcomeName(game.outcome);
    result["terminated"] = game.outcome != sts::GameOutcome::UNDECIDED;
    result["seed"] = game.seed;
    result["ascension"] = game.ascension;
    result["act"] = game.act;
    result["floor"] = game.floorNum;
    result["map_x"] = game.curMapNodeX;
    result["map_y"] = game.curMapNodeY;
    result["turn"] = inBattle ? battle->turn + 1 : 0;
    result["deck"] = cardCounts(game.deck.cards.begin(), game.deck.cards.end());
    result["relics"] = relicState(game);
    result["potions"] = potionState(game, battle);
    result["potion_capacity"] = battle ? battle->potionCapacity : game.potionCapacity;
    result["map"] = mapState(game);
    result["act_boss"] = monsterEncounterStrings[static_cast<int>(game.boss)];
    result["ruby_key"] = game.redKey;
    result["emerald_key"] = game.greenKey;
    result["sapphire_key"] = game.blueKey;

    pybind11::dict player;
    player["hp"] = inBattle ? battle->player.curHp : game.curHp;
    player["max_hp"] = inBattle ? battle->player.maxHp : game.maxHp;
    player["block"] = inBattle ? battle->player.block : 0;
    player["energy"] = inBattle ? battle->player.energy : 0;
    player["gold"] = inBattle ? battle->player.gold : game.gold;
    player["statuses"] = inBattle ? playerStatuses(battle->player) : pybind11::list();
    result["player"] = player;

    pybind11::list hand;
    pybind11::list enemies;
    pybind11::list drawPile;
    pybind11::list discardPile;
    pybind11::list exhaustPile;

    if (inBattle) {
        for (int handIndex = 0; handIndex < battle->cards.cardsInHand; ++handIndex) {
            const auto &card = battle->cards.hand[handIndex];
            pybind11::dict cardState;
            cardState["instance_id"] = publicCardInstanceId(card, handIndex);
            cardState["card_id"] = sts::getCardStringId(card.getId());
            cardState["name"] = card.getName();
            cardState["cost"] = card.costForTurn;
            cardState["upgraded"] = card.isUpgraded();
            cardState["playable"] = card.canUseOnAnyTarget(*battle);
            cardState["requires_target"] = card.requiresTarget();
            hand.append(cardState);
        }

        for (int monsterIndex = 0; monsterIndex < battle->monsters.monsterCount; ++monsterIndex) {
            const auto &monster = battle->monsters.arr[monsterIndex];
            pybind11::dict monsterState;
            monsterState["enemy_id"] = monster.idx;
            monsterState["monster_id"] = sts::monsterIdStrings[static_cast<int>(monster.id)];
            monsterState["name"] = monster.getName();
            monsterState["hp"] = monster.curHp;
            monsterState["max_hp"] = monster.maxHp;
            monsterState["block"] = monster.block;

            int intentDamage = 0;
            int intentHits = 0;
            if (!battle->player.hasRelic<sts::RelicId::RUNIC_DOME>() &&
                monster.isAttacking()) {
                const auto damage = monster.getMoveBaseDamage(*battle);
                intentDamage = monster.calculateDamageToPlayer(*battle, damage.damage);
                intentHits = damage.attackCount;
            }
            monsterState["intent_damage"] = intentDamage;
            monsterState["intent_hits"] = intentHits;
            monsterState["statuses"] = monsterStatuses(monster);
            enemies.append(monsterState);
        }

        drawPile = cardCounts(battle->cards.drawPile.begin(), battle->cards.drawPile.end());
        discardPile = cardCounts(battle->cards.discardPile.begin(), battle->cards.discardPile.end());
        exhaustPile = cardCounts(battle->cards.exhaustPile.begin(), battle->cards.exhaustPile.end());
    }

    result["hand"] = hand;
    result["enemies"] = enemies;
    result["draw_pile"] = drawPile;
    result["discard_pile"] = discardPile;
    result["exhaust_pile"] = exhaustPile;
    return result;
}

pybind11::list sts::py::SimulatorBridge::legalActions() const {
    pybind11::list result;
    if (battle) {
        for (const auto action : sts::search::Action::getAllActionsInState(*battle)) {
            result.append(battleActionState(action, *battle));
        }
    } else {
        for (const auto action : sts::search::GameAction::getAllActionsInState(game)) {
            if (isGameActionAllowed(action)) {
                result.append(gameActionState(action, game));
            }
        }
    }
    return result;
}

void sts::py::SimulatorBridge::skipCardReward(int rewardIndex) {
    if (battle || game.screenState != sts::ScreenState::REWARDS) {
        throw std::runtime_error("card rewards can only be skipped on the rewards screen");
    }
    auto &rewards = game.info.rewardsContainer;
    if (rewardIndex < 0 || rewardIndex >= rewards.cardRewardCount) {
        throw std::invalid_argument("card reward index is not available");
    }
    rewards.removeCardReward(rewardIndex);
}

void sts::py::SimulatorBridge::step(std::uint32_t actionBits) {
    if (game.outcome != sts::GameOutcome::UNDECIDED) {
        throw std::runtime_error("cannot step a terminal simulator");
    }

    if (battle) {
        const auto actions = sts::search::Action::getAllActionsInState(*battle);
        const auto found = std::find_if(actions.begin(), actions.end(), [actionBits](const auto &action) {
            return action.bits == actionBits;
        });
        if (found == actions.end()) {
            throw std::invalid_argument("combat action token is not legal in the current state");
        }
        found->execute(*battle);
        if (battle->outcome != sts::Outcome::UNDECIDED) {
            battle->exitBattle(game);
            battle.reset();
            syncBattleState();
        }
        return;
    }

    const auto actions = sts::search::GameAction::getAllActionsInState(game);
    const auto found = std::find_if(actions.begin(), actions.end(), [this, actionBits](const auto &action) {
        return action.bits == actionBits && isGameActionAllowed(action);
    });
    if (found == actions.end()) {
        throw std::invalid_argument("game action token is not legal in the current state");
    }
    found->execute(game);
    syncBattleState();
}

void sts::py::SimulatorBridge::configureNeow() {
    if (neowMode == 1) {
        game.info.neowRewards[0] = {
            sts::Neow::Bonus::THREE_ENEMY_KILL,
            sts::Neow::Drawback::NONE,
        };
        game.info.neowRewards[1] = {
            sts::Neow::Bonus::TEN_PERCENT_HP_BONUS,
            sts::Neow::Drawback::NONE,
        };
    } else if (neowMode == 2) {
        game.screenState = sts::ScreenState::MAP_SCREEN;
    }
}

bool sts::py::SimulatorBridge::isGameActionAllowed(const sts::search::GameAction &action) const {
    return neowMode != 1 ||
        game.curEvent != sts::Event::NEOW ||
        action.getIdx1() < 2;
}

void sts::py::SimulatorBridge::syncBattleState() {
    if (game.outcome == sts::GameOutcome::UNDECIDED &&
        game.screenState == sts::ScreenState::BATTLE &&
        !battle) {
        battle.emplace();
        battle->init(game);
        if (battle->outcome != sts::Outcome::UNDECIDED) {
            battle->exitBattle(game);
            battle.reset();
        }
    }
}

void sts::py::bindSimulatorBridge(pybind11::module_ &module) {
    pybind11::class_<SimulatorBridge>(module, "SimulatorBridge")
        .def(
            pybind11::init<std::uint64_t, int, int, int, bool>(),
            pybind11::arg("seed"),
            pybind11::arg("ascension"),
            pybind11::arg("neow_mode") = 0,
            pybind11::arg("act_one_bosses_seen") = 3,
            pybind11::arg("final_act_unlocked") = true)
        .def("clone", &SimulatorBridge::clone)
        .def(
            "redeterminized_clone",
            &SimulatorBridge::redeterminizedClone,
            pybind11::arg("search_seed"),
            pybind11::arg("known_top") = std::vector<std::string>(),
            pybind11::arg("known_bottom") = std::vector<std::string>())
        .def("observe", &SimulatorBridge::observe)
        .def("legal_actions", &SimulatorBridge::legalActions)
        .def("skip_card_reward", &SimulatorBridge::skipCardReward)
        .def("step", &SimulatorBridge::step);
}
