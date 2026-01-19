"""
PettingZoo AEC wrapper for the Generals-like two-player game in this repo.

Action encoding (MultiDiscrete of length 8):
  [0] type: 0 EndTurn, 1 ArmyMove, 2 GeneralMove, 3 LevelUp,
              4 GeneralSkill, 5 TechUpdate, 6 SuperWeapon, 7 CallGenerals, 8 GiveUp
  [1] a0: generic slot (x or id or type) depending on action type
  [2] a1: generic slot (y or dest_x or sub-type)
  [3] a2: generic slot (direction 0..3, dest_y, or skill index)
  [4] a3: generic slot (num to move, target x, or weapon target x)
  [5] a4: generic slot (target y, or weapon target y)
  [6] a5: generic slot (start x for transmission)
  [7] a6: generic slot (start y for transmission)

Observation (Dict):
  - board_owner: (row, col) int8 in [-1, 1]
  - board_army:  (row, col) int32 >= 0 (clipped to max_army_obs)
  - board_terrain: (row, col) int8 in [0, 2]
  - generals_owner: (row, col) int8 in [-1, 1] (-1 means no general)
  - coins: (2,) int32 >=0
  - rest_moves: (2,) int32 >=0
  - current_player: scalar int8 in [0,1]

Notes:
  - The wrapper keeps the same agent's turn until they choose EndTurn (type==0)
    or have no rest_move_step left for army moves. End of player_1's turn triggers
    an update_round() call matching the original game loop.
  - Rewards are sparse: +1/-1 on terminal for winner/loser, otherwise 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from pettingzoo import AECEnv
    from gymnasium import spaces
except Exception as e:  # pragma: no cover - runtime dep may not be installed locally
    AECEnv = object  # type: ignore
    spaces = None  # type: ignore

from logic.constant import row, col
from logic.gamestate import GameState, init_generals, update_round
from logic.gamedata import CellType
from logic.ai2logic import execute_single_command


try:
    from logic.game_rules import is_game_over
except Exception:
    # Fallback minimal end condition if logic import fails
    def is_game_over(gamestate: GameState) -> int:  # type: ignore
        return -1


@dataclass
class EnvConfig:
    max_generals: int = 64
    max_army_move: int = 100
    max_rounds: int = 500
    initial_coin: int = 40
    max_army_obs: int = 10000


class GeneralsAECEnv(AECEnv):
    metadata = {
        "render_modes": ["ansi"],
        "name": "generals_pettingzoo_v0",
    }

    def __init__(self, render_mode: str | None = None, config: EnvConfig | None = None):
        self.render_mode = render_mode
        self.config = config or EnvConfig()

        # Agent naming follows PettingZoo convention
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents.copy()

        # Internal state
        self._state: GameState | None = None
        self._current: int = 0  # 0 or 1
        self._last_player_ended: int | None = None

        # AEC required dicts
        self.rewards: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self.terminations: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.truncations: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.infos: Dict[str, Dict[str, Any]] = {a: {} for a in self.possible_agents}
        self._cumulative_rewards: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self.agent_selection: str = self.agents[0]

        # Spaces
        self._build_spaces()

    # ---- PettingZoo Core API ----
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        del seed  # unused
        options = options or {}
        self.agents = self.possible_agents.copy()
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self.agent_selection = self.agents[0]
        self._current = 0
        self._last_player_ended = None

        # Game init
        self._state = GameState()
        init_generals(self._state)
        self._state.coin = [self.config.initial_coin, self.config.initial_coin]

    def observe(self, agent: str):
        player = 0 if agent == "player_0" else 1
        return self._get_obs(player)

    def step(self, action: List[int] | np.ndarray):
        if self.terminations[self.agents[0]] or self.terminations[self.agents[1]]:
            return  # Episode over

        assert self._state is not None
        player = self._current
        agent = self.agents[player]
        if agent != self.agent_selection:
            # Order enforcing (lightweight) — ignore stray steps
            return

        # Clear previous rewards for this agent
        self.rewards[agent] = 0.0

        # Normalize action to python list of ints (length >= 1)
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if not isinstance(action, list):
            raise ValueError("action must be list-like (MultiDiscrete vector)")
        if len(action) < 1:
            raise ValueError("action vector is empty")

        atype = int(action[0])
        success = True

        # End turn immediately
        if atype == 0:
            end_turn = True
        elif atype == 8:
            # Give up
            self._state.winner = 1 - player
            end_turn = True
        else:
            end_turn = False
            params = self._decode_params(atype, action)
            success = execute_single_command(player, self._state, atype, params)

        # 更大惩罚：非法操作直接-1
        if not success:
            self.rewards[agent] = -1.0

        # Check for round/end handling
        winner = is_game_over(self._state)
        if winner != -1:
            # Terminal state
            self.terminations = {a: True for a in self.agents}
            self.rewards[self.agents[winner]] = 1.0
            self.rewards[self.agents[1 - winner]] = -1.0
            self.agent_selection = self.agents[1 - player]
            return

        # Truncation by rounds
        if self._state.round > self.config.max_rounds:
            self.truncations = {a: True for a in self.agents}
            self.agent_selection = self.agents[1 - player]
            return

        # Advance turn logic
        if end_turn or self._state.rest_move_step[player] <= 0:
            self._last_player_ended = player
            # If player_1 just ended, close round
            if player == 1:
                update_round(self._state)
            self._current = 1 - player
            self.agent_selection = self.agents[self._current]
        else:
            # Same player continues
            self.agent_selection = self.agents[self._current]

    def render(self):  # simple ANSI renderer
        if self.render_mode != "ansi":
            return None
        if self._state is None:
            return "<uninitialized>"
        lines = [
            f"Round: {self._state.round} | Coins: {self._state.coin} | Rest: {self._state.rest_move_step}",
        ]
        for i in range(row):
            owner_row = []
            army_row = []
            for j in range(col):
                cell = self._state.board[i][j]
                owner_row.append(str(cell.player))
                army_row.append(str(cell.army))
            lines.append("O:" + ",".join(owner_row))
            lines.append("A:" + ",".join(army_row))
        return "\n".join(lines)

    # ---- API helpers ----
    def _build_spaces(self):
        # Action: MultiDiscrete([type, a0, a1, a2, a3, a4, a5, a6])
        type_n = 9  # 0..8
        max_generals = self.config.max_generals
        max_army = self.config.max_army_move
        self._action_space = spaces.MultiDiscrete(
            np.array([
                type_n,        # type
                max_generals,  # a0 (id/type/x)
                max(1, row),   # a1 (y/dest_x)
                max(1, col),   # a2 (direction 0..3 / dest_y / skill)
                max(1, max_army),  # a3 (num / target x)
                max(1, col),   # a4 (target y)
                max(1, row),   # a5 (start x)
                max(1, col),   # a6 (start y)
            ], dtype=np.int64)
        )

        # Observation space
        self._observation_space = spaces.Dict(
            {
                "board_owner": spaces.Box(low=-1, high=1, shape=(row, col), dtype=np.int8),
                "board_army": spaces.Box(low=0, high=self.config.max_army_obs, shape=(row, col), dtype=np.int32),
                "board_terrain": spaces.Box(low=0, high=2, shape=(row, col), dtype=np.int8),
                "generals_owner": spaces.Box(low=-1, high=1, shape=(row, col), dtype=np.int8),
                "coins": spaces.Box(low=0, high=100000, shape=(2,), dtype=np.int32),
                "rest_moves": spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32),
                "current_player": spaces.Discrete(2),
            }
        )

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self._observation_space

    def _get_obs(self, player: int) -> Dict[str, Any]:
        assert self._state is not None
        owner = np.empty((row, col), dtype=np.int8)
        army = np.empty((row, col), dtype=np.int32)
        terrain = np.empty((row, col), dtype=np.int8)
        generals_owner = np.full((row, col), -1, dtype=np.int8)
        for i in range(row):
            for j in range(col):
                cell = self._state.board[i][j]
                owner[i, j] = cell.player
                army[i, j] = min(cell.army, self.config.max_army_obs)
                terrain[i, j] = int(cell.type)
                if cell.generals is not None:
                    generals_owner[i, j] = cell.generals.player

        obs = {
            "board_owner": owner,
            "board_army": army,
            "board_terrain": terrain,
            "generals_owner": generals_owner,
            "coins": np.array(self._state.coin, dtype=np.int32),
            "rest_moves": np.array(self._state.rest_move_step, dtype=np.int32),
            "current_player": int(self._current),
        }
        return obs

    def _decode_params(self, atype: int, action: List[int]) -> List[int]:
        # Coerce defaults and clamp values into legal board range
        def clamp(v, lo, hi):
            return int(max(lo, min(v, hi)))

        a0 = int(action[1]) if len(action) > 1 else 0
        a1 = int(action[2]) if len(action) > 2 else 0
        a2 = int(action[3]) if len(action) > 3 else 0
        a3 = int(action[4]) if len(action) > 4 else 0
        a4 = int(action[5]) if len(action) > 5 else 0
        a5 = int(action[6]) if len(action) > 6 else 0
        a6 = int(action[7]) if len(action) > 7 else 0

        if atype == 1:  # ArmyMove: [x, y, direction(1..4), num]
            x = clamp(a0, 0, row - 1)
            y = clamp(a1, 0, col - 1)
            direction = clamp(a2, 0, 3) + 1  # Direction enum expects 1..4
            num = max(1, int(a3))
            return [x, y, direction, num]
        elif atype == 2:  # GeneralMove: [general_id, dest_x, dest_y]
            gid = clamp(a0, 0, self.config.max_generals - 1)
            dx = clamp(a1, 0, row - 1)
            dy = clamp(a2, 0, col - 1)
            return [gid, dx, dy]
        elif atype == 3:  # LevelUp: [general_id, upgrade_type 1..3]
            gid = clamp(a0, 0, self.config.max_generals - 1)
            up = clamp(a1, 1, 3)
            return [gid, up]
        elif atype == 4:  # GeneralSkill: [general_id, skill(1..5), x, y]
            gid = clamp(a0, 0, self.config.max_generals - 1)
            sk = clamp(a1, 1, 5)
            x = clamp(a2, 0, row - 1)
            y = clamp(a3, 0, col - 1)
            return [gid, sk, x, y]
        elif atype == 5:  # TechUpdate: [tech_type(1..4)]
            tt = clamp(a0, 1, 4)
            return [tt]
        elif atype == 6:  # SuperWeapon: [weapon_type(1..4), x, y, start_x, start_y]
            wt = clamp(a0, 1, 4)
            x = clamp(a1, 0, row - 1)
            y = clamp(a2, 0, col - 1)
            sx = clamp(a3, 0, row - 1)
            sy = clamp(a4, 0, col - 1)
            return [wt, x, y, sx, sy]
        elif atype == 7:  # CallGenerals: [x, y]
            x = clamp(a0, 0, row - 1)
            y = clamp(a1, 0, col - 1)
            return [x, y]
        else:
            return []


def env(render_mode: str | None = None, config: EnvConfig | None = None) -> GeneralsAECEnv:
    return GeneralsAECEnv(render_mode=render_mode, config=config)


def parallel_env(*args, **kwargs):  # pragma: no cover - requires pettingzoo conversion
    try:
        from pettingzoo.utils.conversions import aec_to_parallel
    except Exception as e:
        raise RuntimeError("pettingzoo not installed for parallel conversion") from e
    return aec_to_parallel(GeneralsAECEnv(*args, **kwargs))
