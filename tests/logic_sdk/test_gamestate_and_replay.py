import json
import pytest

from logic.gamestate import init_generals, update_round
from logic.gamedata import CellType, SuperWeapon, WeaponType


def test_init_generals_counts(plain_state):
    s = plain_state
    # Ensure all cells are plain so placement is possible
    for i in range(len(s.board)):
        for j in range(len(s.board[0])):
            s.board[i][j].type = CellType.PLAIN
    init_generals(s)
    # 2 main + 4 sub + 8 farmer = 14 by constants
    assert len(s.generals) == 14
    # There should be exactly two main generals, each owned by a player
    mains = [g for g in s.generals if type(g).__name__ == "MainGenerals"]
    assert len(mains) == 2
    assert set(g.player for g in mains) == {0, 1}


def test_update_round_mechanics(plain_state, tmp_path):
    s = plain_state
    s.replay_file = str(tmp_path / "rep.txt")
    # Place units/generals
    # Main general at (0,0) gains army per round
    from tests.conftest import place_general

    mg = place_general(s, "main", 0, 0, 0)
    s.board[0][0].army = 0
    # Sub general of player 0 at (0,1) produces army
    sg = place_general(s, "sub", 0, 1, 0)
    s.board[0][1].army = 0
    # Farmer of player 0 at (1,1) produces coin
    fg = place_general(s, "farmer", 1, 1, 0)
    s.board[1][1].army = 0
    s.coin = [0, 0]

    # Bog penalty on (2,2)
    s.board[2][2].type = CellType.BOG
    s.board[2][2].player = 0
    s.board[2][2].army = 1

    # Active nuclear boom affects (3,3)
    s.board[3][3].player = 0
    s.board[3][3].army = 5
    s.active_super_weapon.append(
        SuperWeapon(type=WeaponType.NUCLEAR_BOOM, player=0, cd=0, rest=1, position=[3, 3])
    )
    s.super_weapon_cd = [10, -1]

    # Make this a round divisible by 10 to trigger +1 army
    s.round = 10
    update_round(s)

    # Main and sub produced armies
    assert s.board[0][0].army >= mg.produce_level + 1  # +1 from round bonus
    assert s.board[0][1].army >= sg.produce_level + 1
    # Farmer produced coin
    assert s.coin[0] >= fg.produce_level
    # Bog penalty applied
    assert s.board[2][2].army == 0 and s.board[2][2].player == -1
    # Nuclear boom reduced army around (3,3). +1 from round bonus then -3
    assert s.board[3][3].army == 3
    # CD reduced
    assert s.super_weapon_cd[0] == 9
    # Round incremented
    assert s.round == 11


def test_trans_state_to_init_json_and_replay_cell_type(plain_state):
    s = plain_state
    # Prepare a couple of cells and generals to appear in replay
    from tests.conftest import place_general

    place_general(s, "main", 1, 1, 0)
    place_general(s, "sub", 2, 2, 1)
    # Set some specific cell types to verify Cell_type string generation
    s.board[0][0].type = CellType.BOG
    s.board[0][1].type = CellType.MOUNTAIN

    data = s.trans_state_to_init_json(-1)
    assert "Cell_type" in data
    # First two chars correspond to (0,0) and (0,1)
    assert data["Cell_type"][0] == str(int(CellType.BOG))
    assert data["Cell_type"][1] == str(int(CellType.MOUNTAIN))
