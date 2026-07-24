#ifndef STS_LIGHTSPEED_SIMULATOR_BRIDGE_H
#define STS_LIGHTSPEED_SIMULATOR_BRIDGE_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "combat/BattleContext.h"
#include "game/GameContext.h"

namespace sts::search {
    class GameAction;
}

namespace sts::py {

    class SimulatorBridge {
    public:
        SimulatorBridge(
            std::uint64_t seed,
            int ascension,
            int neowMode = 0,
            int actOneBossesSeen = 3,
            bool finalActUnlocked = true);
        SimulatorBridge(const SimulatorBridge &other);

        [[nodiscard]] std::unique_ptr<SimulatorBridge> clone() const;
        [[nodiscard]] std::unique_ptr<SimulatorBridge> redeterminizedClone(
            std::uint64_t searchSeed,
            const std::vector<std::string> &knownTop,
            const std::vector<std::string> &knownBottom) const;
        [[nodiscard]] pybind11::dict observe() const;
        [[nodiscard]] pybind11::list legalActions() const;
        void skipCardReward(int rewardIndex);
        void step(std::uint32_t actionBits);

    private:
        GameContext game;
        std::optional<BattleContext> battle;
        int neowMode;

        void configureNeow();
        void redeterminizeCombat(
            std::uint64_t searchSeed,
            const std::vector<std::string> &knownTop,
            const std::vector<std::string> &knownBottom);
        [[nodiscard]] bool isGameActionAllowed(const search::GameAction &action) const;
        void syncBattleState();
    };

    void bindSimulatorBridge(pybind11::module_ &module);

}

#endif
