import random

from logic.gamestate import GameState
from logic.ai2logic import execute_single_command
from logic.gamedata import MainGenerals
from logic.gamestate import update_round

from AI.ai_random_safe import policy as random_safe_policy
from AI.ai_greedy import policy as greedy_policy


def _place_main(state: GameState, player: int, x: int, y: int) -> None:
    g = MainGenerals(id=state.next_generals_id, player=player, position=[x, y])
    state.next_generals_id += 1
    state.generals.append(g)
    c = state.board[x][y]
    c.player = player
    c.generals = g
    c.army = 5


def _ops_all_legal(player: int, state: GameState, ops: list[list[int]]) -> bool:
    for op in ops:
        if not op:
            continue
        if op[0] == 8:
            break
        ok = execute_single_command(player, state, op[0], op[1:])
        if not ok:
            return False
    return True


def test_ai_moves_are_legal_over_several_rounds(tmp_path):
    random.seed(0)
    # Prepare a simple empty state: all plain cells, generous coins, free moves
    s = GameState()
    s.replay_file = str(tmp_path / "rep.json")
    for i in range(len(s.board)):
        for j in range(len(s.board[0])):
            cell = s.board[i][j]
            cell.type = type(cell.type)(0)
            cell.player = -1
            cell.generals = None
            cell.army = 0
    s.coin = [1000, 1000]
    s.rest_move_step = [2, 2]

    # Place main generals for both players with some army so they can move
    _place_main(s, 0, 3, 3)
    _place_main(s, 1, 5, 5)

    # Run a few rounds, ensuring each AI produces only legal ops
    rounds = 5
    for r in range(1, rounds + 1):
        ops0 = greedy_policy(r, 0, s)
        assert _ops_all_legal(0, s, ops0)
        ops1 = random_safe_policy(r, 1, s)
        assert _ops_all_legal(1, s, ops1)
        update_round(s)

