import pytest

from logic.upgrade import production_up, defence_up, movement_up, tech_update
from logic.gamedata import MainGenerals, SubGenerals, Farmer


def test_production_up_farmer_and_generals(plain_state):
    s = plain_state
    # Farmer upgrades 1 -> 2 -> 4 -> 6
    from tests.conftest import place_general

    farmer = place_general(s, "farmer", 1, 1, 0)
    s.coin[0] = 100
    assert production_up([1, 1], s, 0) is True
    assert farmer.produce_level == 2
    assert production_up([1, 1], s, 0) is True
    assert farmer.produce_level == 4
    assert production_up([1, 1], s, 0) is True
    assert farmer.produce_level == 6
    # No further upgrades
    assert production_up([1, 1], s, 0) is False

    # Main general halves cost path
    main = place_general(s, "main", 2, 2, 0)
    main.produce_level = 1
    s.coin[0] = 200
    assert production_up([2, 2], s, 0) is True
    assert main.produce_level == 2
    assert production_up([2, 2], s, 0) is True
    assert main.produce_level == 4
    assert production_up([2, 2], s, 0) is False

    # Sub general full cost path
    sub = place_general(s, "sub", 3, 3, 0)
    sub.produce_level = 1
    s.coin[0] = 200
    assert production_up([3, 3], s, 0) is True
    assert sub.produce_level == 2
    assert production_up([3, 3], s, 0) is True
    assert sub.produce_level == 4
    assert production_up([3, 3], s, 0) is False


def test_defence_up_all_types(plain_state):
    s = plain_state
    from tests.conftest import place_general

    # Farmer: 1 -> 1.5 -> 2 -> 3
    farmer = place_general(s, "farmer", 1, 1, 0)
    s.coin[0] = 200
    assert defence_up([1, 1], s, 0) is True
    assert farmer.defense_level == 1.5
    assert defence_up([1, 1], s, 0) is True
    assert farmer.defense_level == 2
    assert defence_up([1, 1], s, 0) is True
    assert farmer.defense_level == 3
    assert defence_up([1, 1], s, 0) is False

    # Main general 1 -> 2 -> 3
    main = place_general(s, "main", 2, 2, 0)
    s.coin[0] = 200
    main.defense_level = 1
    assert defence_up([2, 2], s, 0) is True
    assert main.defense_level == 2
    assert defence_up([2, 2], s, 0) is True
    assert main.defense_level == 3
    assert defence_up([2, 2], s, 0) is False

    # Sub general 1 -> 2 -> 3
    sub = place_general(s, "sub", 3, 3, 0)
    s.coin[0] = 200
    sub.defense_level = 1
    assert defence_up([3, 3], s, 0) is True
    assert sub.defense_level == 2
    assert defence_up([3, 3], s, 0) is True
    assert sub.defense_level == 3
    assert defence_up([3, 3], s, 0) is False


def test_movement_up_main_and_sub(plain_state):
    s = plain_state
    from tests.conftest import place_general

    main = place_general(s, "main", 1, 1, 0)
    main.mobility_level = 1
    s.coin[0] = 200
    assert movement_up([1, 1], s, 0) is True
    assert main.mobility_level == 2 and main.rest_move >= 1
    assert movement_up([1, 1], s, 0) is True
    assert main.mobility_level == 4
    assert movement_up([1, 1], s, 0) is False

    sub = place_general(s, "sub", 2, 2, 0)
    sub.mobility_level = 1
    s.coin[0] = 200
    assert movement_up([2, 2], s, 0) is True
    assert sub.mobility_level == 2
    assert movement_up([2, 2], s, 0) is True
    assert sub.mobility_level == 4
    assert movement_up([2, 2], s, 0) is False


def test_tech_update_paths(plain_state):
    s = plain_state
    s.coin = [1000, 1000]

    # Mobility: 2 -> 3 -> 5
    assert tech_update(0, s, 0) is True
    assert s.tech_level[0][0] == 3 and s.rest_move_step[0] >= 3
    assert tech_update(0, s, 0) is True
    assert s.tech_level[0][0] == 5 and s.rest_move_step[0] >= 5
    assert tech_update(0, s, 0) is False

    # Mountaineering
    assert s.tech_level[0][1] == 0
    assert tech_update(1, s, 0) is True
    assert s.tech_level[0][1] == 1
    assert tech_update(1, s, 0) is False

    # Swamp immunity
    assert s.tech_level[0][2] == 0
    assert tech_update(2, s, 0) is True
    assert s.tech_level[0][2] == 1
    assert tech_update(2, s, 0) is False

    # Unlock super weapon
    assert s.tech_level[0][3] == 0
    assert tech_update(3, s, 0) is True
    assert s.tech_level[0][3] == 1
    assert s.super_weapon_unlocked[0] is True
    assert s.super_weapon_cd[0] >= 0
    assert tech_update(3, s, 0) is False

