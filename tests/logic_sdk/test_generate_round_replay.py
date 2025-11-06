from logic.generate_round_replay import get_single_round_replay
from logic.gamedata import MainGenerals, SubGenerals, Farmer, SuperWeapon, WeaponType


def test_generate_round_replay_fields_and_levels(plain_state):
    s = plain_state
    # Coins and tech
    s.coin = [123, 456]
    s.tech_level = [[3, 1, 1, 0], [5, 0, 0, 0]]
    # Active weapon to include
    s.active_super_weapon.append(
        SuperWeapon(type=WeaponType.ATTACK_ENHANCE, player=0, cd=2, rest=2, position=[0, 0])
    )
    # Add generals with specific levels
    from tests.conftest import place_general

    mg = place_general(s, "main", 0, 0, 0)
    mg.produce_level = 2  # maps to 2
    mg.mobility_level = 2  # maps to 2
    sg = place_general(s, "sub", 0, 1, 1)
    sg.defense_level = 2
    fg = place_general(s, "farmer", 1, 1, 0)
    fg.defense_level = 1.5  # special mapping case

    cells = [[0, 0], [0, 1]]
    rep = get_single_round_replay(s, cells, player=0, action=[8])

    assert rep["Round"] == s.round
    assert rep["Player"] == 0
    assert rep["Action"] == [8]
    assert rep["Weapon_cds"] == s.super_weapon_cd
    assert rep["Coins"] == s.coin

    # Tech mapping for mobility index (0): (val+1)//2
    assert rep["Tech_level"][0][0] == (s.tech_level[0][0] + 1) // 2
    assert rep["Tech_level"][1][0] == (s.tech_level[1][0] + 1) // 2

    # Generals entries and type mapping 1/2/3
    types = {tuple(g["Position"]): g["Type"] for g in rep["Generals"]}
    assert types[(0, 0)] == 1
    assert types[(0, 1)] == 2
    assert types[(1, 1)] == 3

    # Level mapping for farmer defence special case
    def_levels = {tuple(g["Position"]): g["Level"][1] for g in rep["Generals"]}
    assert def_levels[(1, 1)] == 2

