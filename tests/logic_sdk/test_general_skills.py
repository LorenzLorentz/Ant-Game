import pytest

from logic.gamedata import SkillType
from logic.general_skills import skill_activate


def test_surprise_attack_success(plain_state):
    s = plain_state
    from tests.conftest import place_general

    # place general with enough coins, enemy nearby
    gen = place_general(s, "main", 2, 2, 0)
    s.board[2][2].army = 5
    s.board[2][3].player = 1
    s.board[2][3].army = 1
    s.coin[0] = 100
    gen.skills_cd[0] = 0

    ok = skill_activate(0, [2, 2], [2, 3], s, SkillType.SURPRISE_ATTACK)
    assert ok is True
    # general moved to destination
    assert s.board[2][2].generals is None
    assert s.board[2][3].generals is not None
    assert s.board[2][3].generals.id == gen.id
    # coins spent and cd applied
    assert s.coin[0] <= 100
    assert s.board[2][3].generals.skills_cd[0] > 0


def test_surprise_attack_invalid(plain_state):
    s = plain_state
    from tests.conftest import place_general
    from logic.gamedata import SuperWeapon, WeaponType

    gen = place_general(s, "main", 3, 3, 0)
    s.board[3][3].army = 2
    # Insufficient coin
    s.coin[0] = 0
    gen.skills_cd[0] = 0
    assert skill_activate(0, [3, 3], [3, 4], s, SkillType.SURPRISE_ATTACK) is False

    # Blocked by enemy time stop
    s.coin[0] = 100
    s.active_super_weapon.append(
        SuperWeapon(type=WeaponType.TIME_STOP, player=1, cd=1, rest=1, position=[3, 3])
    )
    assert skill_activate(0, [3, 3], [3, 4], s, SkillType.SURPRISE_ATTACK) is False


def test_rout_breakthrough(plain_state):
    s = plain_state
    from tests.conftest import place_general

    gen = place_general(s, "main", 4, 4, 0)
    s.coin[0] = 100
    gen.skills_cd[1] = 0
    s.board[4][5].player = 1
    s.board[4][5].army = 25

    ok = skill_activate(0, [4, 4], [4, 5], s, SkillType.ROUT)
    assert ok is True
    assert s.board[4][5].army == 5  # 25 - 20
    assert gen.skills_cd[1] > 0


def test_command_defence_weaken(plain_state):
    s = plain_state
    from tests.conftest import place_general

    gen = place_general(s, "main", 5, 5, 0)
    s.coin[0] = 1000
    gen.skills_cd = [0, 0, 0, 0, 0]

    assert skill_activate(0, [5, 5], [-1, -1], s, SkillType.COMMAND) is True
    assert gen.skill_duration[0] > 0 and gen.skills_cd[2] > 0

    assert skill_activate(0, [5, 5], [-1, -1], s, SkillType.DEFENCE) is True
    assert gen.skill_duration[1] > 0 and gen.skills_cd[3] > 0

    assert skill_activate(0, [5, 5], [-1, -1], s, SkillType.WEAKEN) is True
    assert gen.skill_duration[2] > 0 and gen.skills_cd[4] > 0

