from logic.constant import row, col
from logic.gamedata import MainGenerals
from logic.gamestate import GameState


def is_game_over(state: GameState) -> int:
    """
    Determine winner.
    - Returns -1 if ongoing
    - Returns 0 or 1 if that player wins
    Rules:
      1) If a player's main general is eliminated, the other wins
      2) If round <= 500, game continues
      3) Otherwise tiebreaker: total army > cells owned > coins
    """
    main_alive = {0: 0, 1: 0}
    for g in state.generals:
        if isinstance(g, MainGenerals):
            main_alive[g.player] = 1
    if main_alive[0] == 0 and main_alive[1] == 1:
        return 1
    if main_alive[1] == 0 and main_alive[0] == 1:
        return 0

    if state.round <= 500:
        return -1

    a0 = a1 = 0
    c0 = c1 = 0
    for i in range(row):
        for j in range(col):
            cell = state.board[i][j]
            if cell.player == 0:
                a0 += cell.army
                c0 += 1
            elif cell.player == 1:
                a1 += cell.army
                c1 += 1

    if a0 != a1:
        return 0 if a0 > a1 else 1
    if c0 != c1:
        return 0 if c0 > c1 else 1
    if state.coin[0] != state.coin[1]:
        return 0 if state.coin[0] > state.coin[1] else 1
    return 0


def tiebreak_now(state: GameState) -> int:
    """Apply tiebreak immediately without waiting for round > 500."""
    current = state.round
    state.round = 501
    try:
        return is_game_over(state)
    finally:
        state.round = current

