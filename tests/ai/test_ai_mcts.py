from logic.ai2logic import execute_single_command
from logic.gamedata import CellType, MainGenerals
from logic.gamestate import GameState, update_round

from AI.ai_mcts import policy as mcts_policy


def _reset_plain(state: GameState) -> None:
    for i in range(len(state.board)):
        for j in range(len(state.board[0])):
            cell = state.board[i][j]
            cell.type = CellType.PLAIN
            cell.player = -1
            cell.army = 0
            cell.generals = None


def _place_main(state: GameState, player: int, x: int, y: int, army: int = 8) -> None:
    general = MainGenerals(id=state.next_generals_id, player=player, position=[x, y])
    state.next_generals_id += 1
    state.generals.append(general)
    state.board[x][y].player = player
    state.board[x][y].army = army
    state.board[x][y].generals = general


def _ops_all_legal(player: int, state: GameState, ops: list[list[int]]) -> bool:
    for op in ops:
        if not op:
            continue
        if op[0] == 8:
            break
        if not execute_single_command(player, state, op[0], op[1:]):
            return False
    return True


def test_mcts_policy_stays_legal_over_multiple_rounds():
    state = GameState()
    _reset_plain(state)
    _place_main(state, 0, 5, 5, army=10)
    _place_main(state, 1, 13, 13, army=10)
    state.coin = [120, 120]
    state.rest_move_step = [2, 2]

    for round_idx in range(1, 4):
        ops0 = mcts_policy(round_idx, 0, state)
        assert ops0[-1] == [8]
        assert _ops_all_legal(0, state, ops0)

        ops1 = mcts_policy(round_idx, 1, state)
        assert ops1[-1] == [8]
        assert _ops_all_legal(1, state, ops1)

        update_round(state)
