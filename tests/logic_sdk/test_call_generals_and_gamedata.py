import pytest

from logic.gamestate import GameState
from logic.call_generals import call_generals
from logic.gamedata import SubGenerals


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
    assert s.coin[0] == 50  # 100 - 50


def test_call_generals_fails_without_enough_coin(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "rep.json")
    _reset_plain(s)
    s.coin = [49, 100]
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

