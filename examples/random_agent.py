from __future__ import annotations

import random

from sts_env import StsEnv, ToyCombatBackend


def main() -> None:
    policy_rng = random.Random(0)
    env = StsEnv(ToyCombatBackend())
    observation, info = env.reset(seed=7)
    total_reward = 0.0

    while True:
        action_index = policy_rng.randrange(len(observation.legal_actions))
        action = observation.legal_actions[action_index]
        observation, reward, terminated, truncated, step_info = env.step(action_index)
        total_reward += reward
        print(
            f"turn={observation.turn:02d} action={action.label:<12} "
            f"player_hp={observation.player.hp:02d} enemy_hp={observation.enemies[0].hp:02d} "
            f"reward={reward:+.1f}"
        )
        if terminated or truncated:
            print(f"episode_end={step_info['terminal_reason']} total_reward={total_reward:+.1f}")
            break


if __name__ == "__main__":
    main()

