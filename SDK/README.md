SDK: PettingZoo Multi-Agent Wrapper
==================================

This SDK provides a PettingZoo-compatible AEC environment for the two-player strategy game implemented in this repository.

- Package entry: `SDK/pettingzoo_env.py`
- Import: `from SDK import env, parallel_env`

Quick start:

```
from SDK import env

# Create environment
my_env = env(render_mode="ansi")
my_env.reset()

# Simple random interaction
for _ in range(20):
    agent = my_env.agent_selection
    a = my_env.action_space(agent).sample()
    # bias towards EndTurn to move the game along
    a[0] = 0 if _ % 3 == 0 else a[0]
    my_env.step(a)
    if any(my_env.terminations.values()) or any(my_env.truncations.values()):
        break

print(my_env.render())
```

Action encoding (MultiDiscrete of length 8):

- idx 0: `type` (0..8)
  - 0 EndTurn
  - 1 ArmyMove: [x, y, direction(0..3), num]
  - 2 GeneralMove: [general_id, dest_x, dest_y]
  - 3 LevelUp: [general_id, upgrade_type 1..3]
  - 4 GeneralSkill: [general_id, skill 1..5, x, y]
  - 5 TechUpdate: [tech_type 1..4]
  - 6 SuperWeapon: [weapon_type 1..4, x, y, start_x, start_y]
  - 7 CallGenerals: [x, y]
  - 8 GiveUp

Observation (Dict):

- `board_owner` (row, col) in {-1, 0, 1}
- `board_army` (row, col) >= 0
- `board_terrain` (row, col) in {0, 1, 2}
- `generals_owner` (row, col) in {-1, 0, 1}
- `coins` (2,)
- `rest_moves` (2,)
- `current_player` scalar in {0, 1}

Turn model:

- The same agent retains the turn until it chooses `EndTurn` (type 0) or has no remaining army move steps.
- After player_1 ends a turn, the environment applies `update_round()` to match the original game logic.

Rewards:

- Sparse: +1 for winner, -1 for loser on termination; 0 otherwise. Invalid actions incur a small -0.01 penalty.

Notes:

- This wrapper depends on `pettingzoo` and `gymnasium`. Install in your environment to run training.
```
pip install pettingzoo gymnasium
```

