import pytest

from logic.gamedata import SkillType
from logic.general_skills import skill_activate


def test_surprise_attack_fractional_damage_rounding(plain_state):
    s = plain_state
    from tests.conftest import place_general
    from logic.gamedata import SuperWeapon, WeaponType

    # Place our general at (2,2) with army 5
    gen = place_general(s, "main", 2, 2, 0)
    s.board[2][2].army = 5
    s.coin[0] = 100
    gen.skills_cd[0] = 0

    # Place friendly buffing general near attacker to raise attack to 1.5 via COMMAND
    buff = place_general(s, "sub", 2, 1, 0)
    buff.skill_duration[0] = 1  # command buff increases attack 1.5x

    # Enemy cell with 1 army at destination (within 2 cells)
    s.board[2][3].player = 1
    s.board[2][3].army = 1

    ok = skill_activate(0, [2, 2], [2, 3], s, SkillType.SURPRISE_ATTACK)
    assert ok is True
    # num = 4, attack = 1.5, enemy def = 1.0 => vs = 6 - 1 = 5
    # remaining army should be ceil(5 / 1.5) = 4
    assert s.board[2][3].player == 0
    assert s.board[2][3].army == 4
