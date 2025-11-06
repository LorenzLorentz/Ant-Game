import pytest

from logic.super_weapons import bomb, strengthen, tp, timestop
from logic.gamedata import SuperWeapon, WeaponType


def unlock_and_ready(state, player=0):
    state.super_weapon_unlocked[player] = True
    state.super_weapon_cd[player] = 0


def test_bomb_effects_on_area(plain_state):
    s = plain_state
    unlock_and_ready(s, 0)

    # Place armies and generals around (5,5)
    from tests.conftest import place_general

    # Main general at center with army
    s.board[5][5].player = 0
    s.board[5][5].army = 10
    mg = place_general(s, "main", 5, 5, 0)

    # Sub general at (5,4)
    s.board[5][4].player = 1
    s.board[5][4].army = 8
    sg = place_general(s, "sub", 5, 4, 1)

    # Other neighbor cell without general
    s.board[4][4].player = 1
    s.board[4][4].army = 3

    assert bomb(s, [5, 5], 0) is True
    # Center main general cell halves army
    assert s.board[5][5].army == 5
    # Sub general cell becomes neutral and general removed
    assert s.board[5][4].player == -1 and s.board[5][4].army == 0
    assert sg not in s.generals
    # Plain neighbor loses army and may become neutral
    assert s.board[4][4].army == 0
    assert s.board[4][4].player == -1


def test_strengthen_and_timestop_activation(plain_state):
    s = plain_state
    unlock_and_ready(s, 0)
    assert strengthen(s, [3, 3], 0) is True
    assert any(w.type == WeaponType.ATTACK_ENHANCE for w in s.active_super_weapon)
    # CD set
    assert s.super_weapon_cd[0] > 0

    # Prepare time stop after cd resets
    s.super_weapon_cd[0] = 0
    assert timestop(s, [2, 2], 0) is True
    assert any(w.type == WeaponType.TIME_STOP for w in s.active_super_weapon)


def test_tp_transfers_army_and_sets_weapon(plain_state):
    s = plain_state
    unlock_and_ready(s, 0)

    s.board[1][1].player = 0
    s.board[1][1].army = 5
    # destination is neutral
    assert tp(s, [1, 1], [1, 2], 0) is True
    assert s.board[1][1].army == 1
    assert s.board[1][2].player == 0
    assert s.board[1][2].army == 4
    assert any(w.type == WeaponType.TRANSMISSION and w.position == [1, 2] for w in s.active_super_weapon)
    assert s.super_weapon_cd[0] > 0

