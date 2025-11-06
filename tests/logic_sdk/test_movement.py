import math
import pytest

from logic.movement import calculate_new_pos, army_move, check_general_movement, general_move
from logic.gamedata import Direction, CellType


def test_calculate_new_pos_and_bounds():
    assert calculate_new_pos([5, 5], Direction.UP) == (4, 5)
    assert calculate_new_pos([5, 5], Direction.DOWN) == (6, 5)
    assert calculate_new_pos([5, 5], Direction.LEFT) == (5, 4)
    assert calculate_new_pos([5, 5], Direction.RIGHT) == (5, 6)
    # Out of range returns (-1, -1)
    assert calculate_new_pos([0, 0], Direction.UP) == (-1, -1)
    assert calculate_new_pos([0, 0], Direction.LEFT) == (-1, -1)


def test_army_move_to_neutral_and_friendly(plain_state):
    s = plain_state
    s.board[0][0].player = 0
    s.board[0][0].army = 10

    # Move 3 to neutral (0,1)
    ok = army_move([0, 0], s, 0, Direction.RIGHT, 3)
    assert ok is True
    assert s.board[0][1].player == 0
    # neutral defence = 1, attack = 1, vs = 3 -> ceil(3)=3
    assert s.board[0][1].army == 3
    assert s.board[0][0].army == 7
    assert s.rest_move_step[0] == 1

    # Move 2 to friendly (0,1) from (0,0)
    ok = army_move([0, 0], s, 0, Direction.RIGHT, 2)
    assert ok is True
    assert s.board[0][1].army == 5  # 3 + 2
    assert s.board[0][0].army == 5  # 7 - 2
    assert s.rest_move_step[0] == 0


def test_army_move_vs_enemy_and_general_flip(plain_state):
    s = plain_state
    # origin
    s.board[0][0].player = 0
    s.board[0][0].army = 10
    # enemy dest with a general present
    s.board[0][1].player = 1
    s.board[0][1].army = 2

    # put an enemy general at dest; should flip owner if captured
    from tests.conftest import place_general

    enemy_gen = place_general(s, "sub", 0, 1, 1)
    ok = army_move([0, 0], s, 0, Direction.RIGHT, 3)
    assert ok is True
    assert s.board[0][1].player == 0
    assert enemy_gen.player == 0  # general flipped
    assert s.board[0][0].army == 7


def test_army_move_invalid_cases(plain_state):
    s = plain_state
    # No control on cell
    s.board[1][1].player = -1
    s.board[1][1].army = 5
    assert army_move([1, 1], s, 0, Direction.RIGHT, 1) is False

    # Not enough army to move
    s.board[1][1].player = 0
    s.board[1][1].army = 1
    assert army_move([1, 1], s, 0, Direction.RIGHT, 1) is False

    # Invalid num
    s.board[1][1].army = 5
    assert army_move([1, 1], s, 0, Direction.RIGHT, 0) is False

    # Mountain blocked without mountaineering
    s.board[1][2].type = CellType.MOUNTAIN
    assert army_move([1, 1], s, 0, Direction.RIGHT, 1) is False

    # No remaining move steps
    s.board[2][2].player = 0
    s.board[2][2].army = 5
    s.rest_move_step[0] = 0
    assert army_move([2, 2], s, 0, Direction.RIGHT, 1) is False


def test_army_move_blocked_by_superweapons(plain_state):
    s = plain_state
    s.board[3][3].player = 0
    s.board[3][3].army = 5

    # Transmission stun at same cell by same player
    from logic.gamedata import SuperWeapon, WeaponType

    s.active_super_weapon.append(
        SuperWeapon(type=WeaponType.TRANSMISSION, player=0, cd=1, rest=1, position=[3, 3])
    )
    assert army_move([3, 3], s, 0, Direction.RIGHT, 1) is False

    # Replace with enemy time stop nearby
    s.active_super_weapon.clear()
    s.active_super_weapon.append(
        SuperWeapon(type=WeaponType.TIME_STOP, player=1, cd=1, rest=1, position=[3, 4])
    )
    assert army_move([3, 3], s, 0, Direction.RIGHT, 1) is False


def test_check_and_general_move(plain_state):
    s = plain_state
    # Place friendly corridor
    s.board[5][5].player = 0
    s.board[5][6].player = 0
    s.board[5][7].player = 0
    # Place general at (5,5)
    from tests.conftest import place_general

    gen = place_general(s, "main", 5, 5, 0)
    gen.rest_move = 2

    # Legal move to (5,7) through friendly cells
    assert check_general_movement([5, 5], s, 0, [5, 7]) is True
    assert general_move([5, 5], s, 0, [5, 7]) is True
    assert s.board[5][7].generals is not None
    assert s.board[5][7].generals.id == gen.id
    assert s.board[5][7].generals.rest_move == 0
    assert s.board[5][5].generals is None

    # Occupied destination blocks movement
    s.board[5][6].generals = gen
    s.board[5][7].generals = None
    assert check_general_movement([5, 6], s, 0, [5, 7]) is False

    # Mountain without tech blocks path
    s.board[5][6].generals = None
    s.board[5][6].type = CellType.MOUNTAIN
    s.board[5][6].player = 0
    s.board[5][7].player = 0
    assert check_general_movement([5, 5], s, 0, [5, 7]) is False

