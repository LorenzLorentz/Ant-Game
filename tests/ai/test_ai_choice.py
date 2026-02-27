import pytest

from logic.gamestate import GameState
from logic.gamedata import CellType, MainGenerals, Direction

from AI.ai_greedy import policy as greedy_policy
from AI.ai_random_safe import policy as random_safe_policy


def _reset_plain(state: GameState):
    for i in range(len(state.board)):
        for j in range(len(state.board[0])):
            c = state.board[i][j]
            c.type = CellType.PLAIN
            c.player = -1
            c.generals = None
            c.army = 0


def _place_main(state: GameState, player: int, x: int, y: int, army: int = 3):
    g = MainGenerals(id=state.next_generals_id, player=player, position=[x, y])
    state.next_generals_id += 1
    state.generals.append(g)
    cell = state.board[x][y]
    cell.player = player
    cell.generals = g
    cell.army = army


def test_greedy_prefers_only_plain_neighbor():
    s = GameState()
    _reset_plain(s)
    x, y = 5, 5
    _place_main(s, 0, x, y, army=5)
    s.rest_move_step = [2, 2]

    # Block all directions except UP with mountains
    s.board[x + 1][y].type = CellType.MOUNTAIN     # DOWN
    s.board[x][y - 1].type = CellType.MOUNTAIN     # LEFT
    s.board[x][y + 1].type = CellType.MOUNTAIN     # RIGHT
    s.board[x - 1][y].type = CellType.PLAIN        # UP remains plain neutral

    ops = greedy_policy(1, 0, s)
    # First op should be a move from (x,y) to UP
    assert ops[0][0] == 1
    assert ops[0][1] == x and ops[0][2] == y
    assert ops[0][3] == int(Direction.UP) + 1
    assert ops[-1] == [8]


def test_random_safe_direction_controlled(monkeypatch):
    s = GameState()
    _reset_plain(s)
    x, y = 3, 3
    _place_main(s, 0, x, y, army=4)
    s.rest_move_step = [1, 1]

    # Force RNG to pick RIGHT (3)
    monkeypatch.setattr("AI.ai_random_safe.random.randint", lambda a, b: 3)
    ops = random_safe_policy(1, 0, s)
    assert ops[0][0] == 1
    assert ops[0][1] == x and ops[0][2] == y
    assert ops[0][3] == int(Direction.RIGHT) + 1
    assert ops[-1] == [8]


def test_random_safe_pass_when_no_move():
    s = GameState()
    _reset_plain(s)
    x, y = 4, 4
    _place_main(s, 0, x, y, army=1)  # only 1 army -> cannot move
    s.rest_move_step = [1, 1]
    ops = random_safe_policy(1, 0, s)
    assert ops == [[8]]

