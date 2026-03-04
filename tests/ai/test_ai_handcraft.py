from logic.ai2logic import execute_single_command
from logic.gamedata import CellType, MainGenerals
from logic.gamestate import GameState, update_round
from logic.map import PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP

from AI.ai_handcraft import policy as handcraft_policy


def _reset_plain(state: GameState) -> None:
    for i in range(len(state.board)):
        for j in range(len(state.board[0])):
            c = state.board[i][j]
            c.type = CellType.PLAIN
            c.player = -1
            c.generals = None
            c.army = 0


def _place_main(state: GameState, player: int, x: int, y: int, army: int = 5) -> None:
    g = MainGenerals(id=state.next_generals_id, player=player, position=[x, y])
    state.next_generals_id += 1
    state.generals.append(g)
    cell = state.board[x][y]
    cell.player = player
    cell.generals = g
    cell.army = army


def _ops_all_legal(player: int, state: GameState, ops: list[list[int]]) -> bool:
    for op in ops:
        if not op:
            continue
        if op[0] == 8:
            break
        if not execute_single_command(player, state, op[0], op[1:]):
            return False
    return True


def test_handcraft_builds_base_defender_when_affordable():
    s = GameState()
    _reset_plain(s)
    _place_main(s, 0, 5, 5, army=1)
    _place_main(s, 1, 13, 13, army=6)
    s.coin = [30, 20]
    bx, by = PLAYER_0_BASE_CAMP
    s.board[bx][by].player = 0
    s.board[bx][by].army = 0

    ops = handcraft_policy(1, 0, s)

    assert ops[0] == [7, bx, by]
    assert ops[-1] == [8]


def test_handcraft_policy_stays_legal_over_multiple_rounds(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "rep.json")
    _reset_plain(s)
    _place_main(s, 0, 5, 5, army=8)
    _place_main(s, 1, 13, 13, army=8)
    s.coin = [120, 120]
    s.rest_move_step = [2, 2]
    bx0, by0 = PLAYER_0_BASE_CAMP
    bx1, by1 = PLAYER_1_BASE_CAMP
    s.board[bx0][by0].player = 0
    s.board[bx1][by1].player = 1

    for round_idx in range(1, 6):
        ops0 = handcraft_policy(round_idx, 0, s)
        assert _ops_all_legal(0, s, ops0)
        ops1 = handcraft_policy(round_idx, 1, s)
        assert _ops_all_legal(1, s, ops1)
        update_round(s)
