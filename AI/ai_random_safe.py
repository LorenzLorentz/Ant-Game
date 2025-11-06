import random
from typing import List

from logic.gamedata import Direction
from logic.gamestate import GameState


def move_army_op(position: list[int], direction: Direction, num: int) -> List[int]:
    return [1, position[0], position[1], int(direction) + 1, num]


def _find_main(state: GameState, player: int):
    for g in state.generals:
        if type(g).__name__ == "MainGenerals" and g.player == player:
            return g
    return None


def policy(round_idx: int, my_seat: int, state: GameState) -> list[list[int]]:
    """Safe random baseline: only moves if legal and has >1 army, else end."""
    g = _find_main(state, my_seat)
    if not g:
        return [[8]]
    x, y = g.position
    if state.board[x][y].army <= 1 or state.rest_move_step[my_seat] <= 0:
        return [[8]]
    d = Direction(random.randint(0, 3))
    num = 1
    return [move_army_op([x, y], d, num), [8]]

