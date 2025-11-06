import os
import sys
import random
import pytest

# Ensure repo root is importable for 'logic' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.gamestate import GameState
from logic.gamedata import CellType, MainGenerals, SubGenerals, Farmer


@pytest.fixture
def plain_state(tmp_path):
    # Deterministic seed for any incidental randomness
    random.seed(0)
    state = GameState()
    # Ensure replay file exists under tmp dir
    state.replay_file = str(tmp_path / "replay.txt")
    # Normalize the board to be fully plain and neutral
    for i in range(len(state.board)):
        for j in range(len(state.board[0])):
            cell = state.board[i][j]
            cell.type = CellType.PLAIN
            cell.player = -1
            cell.generals = None
            cell.army = 0
    # Generous coins by default
    state.coin = [1000, 1000]
    # Reset movement
    state.rest_move_step = [2, 2]
    # Reset CDs / tech
    state.super_weapon_cd = [-1, -1]
    state.super_weapon_unlocked = [False, False]
    state.active_super_weapon = []
    state.tech_level = [[2, 0, 0, 0], [2, 0, 0, 0]]
    state.generals = []
    state.next_generals_id = 0
    state.round = 1
    return state


def place_general(state: GameState, kind: str, x: int, y: int, player: int):
    if kind == "main":
        gen = MainGenerals(id=state.next_generals_id, player=player, position=[x, y])
    elif kind == "sub":
        gen = SubGenerals(id=state.next_generals_id, player=player, position=[x, y])
    else:
        gen = Farmer(id=state.next_generals_id, player=player, position=[x, y])
    state.next_generals_id += 1
    state.generals.append(gen)
    state.board[x][y].generals = gen
    state.board[x][y].player = player
    return gen
