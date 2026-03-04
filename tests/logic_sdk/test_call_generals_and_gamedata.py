import pytest

from logic.gamestate import GameState
from logic.call_generals import call_generals, downgrade_tower
from logic.gamedata import SubGenerals
from logic.upgrade import production_up
import logic.constant as constant


def _reset_plain(state: GameState):
    for i in range(len(state.board)):
        for j in range(len(state.board[0])):
            c = state.board[i][j]
            c.type = type(c.type)(0)
            c.player = -1
            c.generals = None
            c.army = 0


def test_call_generals_success_and_coin_deduction(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "rep.json")
    _reset_plain(s)
    s.coin = [100, 100]
    # Claim a cell for player 0 without general
    pos = [2, 2]
    s.board[pos[0]][pos[1]].player = 0

    ok = call_generals(s, 0, pos)
    assert ok is True
    assert isinstance(s.board[pos[0]][pos[1]].generals, SubGenerals)
    # cost formula = 15 * 2^i where i=0 -> 15
    assert s.coin[0] == 85  # 100 - 15


def test_downgrade_and_refund(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "rep.json")
    _reset_plain(s)
    s.coin = [1000, 0]
    # prepare a player0 cell and build two towers
    pos1 = [1, 1]
    pos2 = [2, 2]
    s.board[pos1[0]][pos1[1]].player = 0
    s.board[pos2[0]][pos2[1]].player = 0
    assert call_generals(s, 0, pos1) is True
    assert call_generals(s, 0, pos2) is True
    tid1 = s.generals[-2].id
    # removal refund after first build: remaining count 1 -> refund 24
    assert downgrade_tower(s, 0, tid1) is True
    assert s.coin[0] == 1000 - 15 - 30 + 24
    # upgrade remaining tower to level3 then downgrade stepwise
    tid2 = s.generals[-1].id
    assert production_up(s.generals[-1].position, s, 0) is True
    assert production_up(s.generals[-1].position, s, 0) is True
    assert s.generals[-1].produce_level == 4
    oldc = s.coin[0]
    assert downgrade_tower(s, 0, tid2) is True
    assert s.generals[-1].produce_level == 2
    assert s.coin[0] == oldc + int(constant.lieutenant_production_T2 * 0.8)
    oldc = s.coin[0]
    assert downgrade_tower(s, 0, tid2) is True
    assert s.generals[-1].produce_level == 1
    assert s.coin[0] == oldc + int(constant.lieutenant_production_T1 * 0.8)
    oldc = s.coin[0]
    assert downgrade_tower(s, 0, tid2) is True
    assert all(g.id != tid2 for g in s.generals)
    assert s.coin[0] == oldc + int(0.8 * 15 * 1)

def test_call_generals_fails_without_enough_coin(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "rep.json")
    _reset_plain(s)
    # First tower costs 15, so 14 should be insufficient.
    s.coin = [14, 100]
    pos = [1, 1]
    s.board[pos[0]][pos[1]].player = 0
    assert call_generals(s, 0, pos) is False


def test_call_generals_fails_on_wrong_owner(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "rep.json")
    _reset_plain(s)
    s.coin = [100, 100]
    pos = [1, 2]
    s.board[pos[0]][pos[1]].player = 1  # owned by opponent
    assert call_generals(s, 0, pos) is False


def test_call_generals_fails_if_cell_has_general(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "rep.json")
    _reset_plain(s)
    s.coin = [100, 100]
    pos = [0, 0]
    s.board[pos[0]][pos[1]].player = 0
    # Pretend a sub general already present
    s.board[pos[0]][pos[1]].generals = SubGenerals(id=0, player=0, position=pos[:])
    assert call_generals(s, 0, pos) is False
