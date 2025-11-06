from logic.game_rules import is_game_over, tiebreak_now
from logic.constant import row, col


def test_is_game_over_by_main_general_death(plain_state):
    s = plain_state
    from tests.conftest import place_general

    g0 = place_general(s, "main", 0, 0, 0)
    g1 = place_general(s, "main", 0, 1, 1)
    # Remove player 1 main general from the game
    s.generals.remove(g1)
    assert is_game_over(s) == 0


def test_tiebreak_army_cells_coins(plain_state):
    s = plain_state
    # No mains present; add them to avoid immediate draw assumptions
    from tests.conftest import place_general

    place_general(s, "main", 0, 0, 0)
    place_general(s, "main", 0, 1, 1)

    # Configure ownership and armies
    # Player 0 has more army
    s.board[1][1].player = 0
    s.board[1][1].army = 10
    s.board[2][2].player = 1
    s.board[2][2].army = 5
    s.coin = [0, 100]

    assert tiebreak_now(s) == 0

    # Equalize armies, give player 1 more cells
    s.board[1][1].army = 5
    s.board[3][3].player = 1
    s.board[3][3].army = 1
    assert tiebreak_now(s) == 1

    # Equalize cells too, decide by coins
    s.board[3][3].player = -1
    s.coin = [10, 20]
    assert tiebreak_now(s) == 1

