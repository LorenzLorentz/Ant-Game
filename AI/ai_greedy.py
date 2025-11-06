from typing import List

from logic.gamedata import Direction
from logic.gamestate import GameState
from logic.constant import row, col
from logic.computation import compute_attack, compute_defence


def move_army_op(position: list[int], direction: Direction, num: int) -> List[int]:
    return [1, position[0], position[1], int(direction) + 1, num]


def _find_main(state: GameState, player: int):
    for g in state.generals:
        if type(g).__name__ == "MainGenerals" and g.player == player:
            return g
    return None


def policy(round_idx: int, my_seat: int, state: GameState) -> list[list[int]]:
    """Greedy adjacent capture baseline."""
    ops: list[list[int]] = []
    g = _find_main(state, my_seat)
    if not g:
        return [[8]]
    x, y = g.position
    moves = min(2, state.rest_move_step[my_seat])
    dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    for _ in range(moves):
        best = None
        best_vs = 0.0
        army_here = state.board[x][y].army
        if army_here <= 1:
            break
        for d in dirs:
            nx = x + (-1 if d == Direction.UP else 1 if d == Direction.DOWN else 0)
            ny = y + (-1 if d == Direction.LEFT else 1 if d == Direction.RIGHT else 0)
            if nx < 0 or nx >= row or ny < 0 or ny >= col:
                continue
            dest = state.board[nx][ny]
            if int(dest.type) == 2:  # avoid mountains (no tech)
                continue
            atk = compute_attack(state.board[x][y], state)
            dfn = compute_defence(dest, state)
            for num in range(1, min(3, army_here - 1) + 1):
                vs = num * atk - dest.army * dfn
                if dest.player != my_seat and vs > best_vs:
                    best_vs = vs
                    best = (d, num)
            if dest.player == -1 and dest.army == 0 and best is None:
                best = (d, 1)
                best_vs = 1.0
        if best is None:
            break
        d, num = best
        ops.append(move_army_op([x, y], d, int(num)))
        break
    ops.append([8])
    return ops

