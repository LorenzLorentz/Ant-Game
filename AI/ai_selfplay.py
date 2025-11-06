"""
AI that uses a trained reduced-action policy saved under AI/selfplay/model.pt.

Interface:
  - Exposes function `policy(round_idx, my_seat, state)` returning a command list
    compatible with logic.ai2logic (list of lists), ending with [8] to finish turn.

Weights:
  - Default path: AI/selfplay/model.pt
  - To update: run `python SDK/train_selfplay.py --save AI/selfplay/model.pt`
"""

from typing import List
import os

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

import numpy as np

from logic.constant import row, col
from logic.gamedata import Direction
from logic.gamestate import GameState


MODEL_PATH = os.environ.get("SELFPLAY_MODEL", os.path.join("AI", "selfplay", "model.pt"))


class SmallPolicy(nn.Module):  # type: ignore[misc]
    def __init__(self):
        super().__init__()
        inp = row * col * 4 + 2 + 2 + 1
        hid = 256
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 5)
        )

    def forward(self, x):
        return self.net(x)


def _build_obs(state: GameState, player: int):
    owner = np.empty((row, col), dtype=np.int8)
    army = np.empty((row, col), dtype=np.int32)
    terrain = np.empty((row, col), dtype=np.int8)
    generals_owner = np.full((row, col), -1, dtype=np.int8)
    for i in range(row):
        for j in range(col):
            cell = state.board[i][j]
            owner[i, j] = cell.player
            army[i, j] = cell.army
            terrain[i, j] = int(cell.type)
            if cell.generals is not None:
                generals_owner[i, j] = cell.generals.player
    return {
        "board_owner": owner,
        "board_army": army,
        "board_terrain": terrain,
        "generals_owner": generals_owner,
        "coins": np.array(state.coin, dtype=np.int32),
        "rest_moves": np.array(state.rest_move_step, dtype=np.int32),
        "current_player": int(player),
    }


def _obs_to_tensor(obs) -> np.ndarray:
    owner = obs["board_owner"].astype(np.float32)
    army = np.log1p(obs["board_army"].astype(np.float32))
    terrain = obs["board_terrain"].astype(np.float32) / 2.0
    gens = (obs["generals_owner"].astype(np.float32) + 1.0) / 2.0
    coins = obs["coins"].astype(np.float32) / 100.0
    rest = obs["rest_moves"].astype(np.float32) / 5.0
    cur = np.array([obs["current_player"]], dtype=np.float32)
    feat = np.concatenate(
        [owner.reshape(-1), army.reshape(-1), terrain.reshape(-1), gens.reshape(-1), coins, rest, cur]
    )
    return feat


def _pick_action(policy: SmallPolicy, obs) -> int:
    x = torch.from_numpy(_obs_to_tensor(obs)).float().unsqueeze(0)
    with torch.no_grad():
        logits = policy(x)[0]
        probs = torch.softmax(logits, dim=-1)
        a = torch.distributions.Categorical(probs).sample()
        return int(a.item())


def _main_gen(state: GameState, player: int):
    for g in state.generals:
        if type(g).__name__ == "MainGenerals" and g.player == player:
            return g
    return None


def policy(round_idx: int, my_seat: int, state: GameState) -> List[List[int]]:
    # Fallback if torch or model unavailable
    if torch is None or not os.path.exists(MODEL_PATH):
        return [[8]]
    # Load once and cache on module
    global _POLICY
    try:
        _ = _POLICY  # type: ignore[name-defined]
    except Exception:
        _POLICY = SmallPolicy()  # type: ignore[assignment]
        _POLICY.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _POLICY.eval()
    obs = _build_obs(state, my_seat)
    action_id = _pick_action(_POLICY, obs)  # type: ignore[arg-type]
    # Map to commands
    if action_id == 0:
        return [[8]]
    g = _main_gen(state, my_seat)
    if not g:
        return [[8]]
    dir_map = {1: Direction.UP, 2: Direction.DOWN, 3: Direction.LEFT, 4: Direction.RIGHT}
    d = dir_map.get(action_id, Direction.UP)
    op = [1, g.position[0], g.position[1], int(d) + 1, 1]
    return [op, [8]]

