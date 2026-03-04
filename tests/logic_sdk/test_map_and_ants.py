import pytest

from logic.map import Map, MAP_SIZE
from logic.ant import Ant
from logic.gamestate import GameState
from logic.gamedata import MainGenerals


def test_map_validity():
    m = Map()
    # points inside bounds but listed as invalid should be rejected
    assert not m.is_valid(6, 1)
    # corners of rectangular region may be invalid as well
    assert not m.is_valid(-1, 0)
    assert not m.is_valid(MAP_SIZE, 0)
    # a known valid location
    assert m.is_valid(0, 0)


def test_call_generals_respects_map():
    s = GameState()
    # pick an invalid coordinate for tower building
    from logic.call_generals import call_generals

    # negative coordinates should be rejected without throwing
    ok = call_generals(s, 0, [-1, -1])
    assert ok is False
    # using a valid index inside the board but outside the hex shape
    bad = [6, 1]
    ok = call_generals(s, 0, bad)
    assert ok is False


def test_ant_path_and_status():
    ant = Ant(player=0, id=0, x=0, y=0, level=0)
    # target farther away, no path yet
    assert ant.get_status(((2, 2), (5, 5))) == Ant.Status.Alive
    ant.move(0)
    assert ant.pos != (0, 0)
    ant.hp = -1
    assert ant.get_status(((2, 2), (5, 5))) == Ant.Status.Fail
