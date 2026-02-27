import json
import pytest

from logic.gamestate import GameState, update_round
from logic.gamedata import CellType, MainGenerals


def _reset_plain(state: GameState):
    for i in range(len(state.board)):
        for j in range(len(state.board[0])):
            c = state.board[i][j]
            c.type = CellType.PLAIN
            c.player = -1
            c.generals = None
            c.army = 0


def _place_main(state: GameState, player: int, x: int, y: int, army: int = 3) -> None:
    g = MainGenerals(id=state.next_generals_id, player=player, position=[x, y])
    state.next_generals_id += 1
    state.generals.append(g)
    cell = state.board[x][y]
    cell.player = player
    cell.generals = g
    cell.army = army


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_transition_and_ant_replay_logging(tmp_path, seed):
    s = GameState()
    s.replay_file = str(tmp_path / "ant_replay.json")
    _reset_plain(s)

    # Place mains to populate towers/ants and enable production
    _place_main(s, 0, 1, 1, army=2)
    _place_main(s, 1, 3, 3, army=4)

    # Prepare a bog tile that should lose army and become neutral
    s.board[2][2].type = CellType.BOG
    s.board[2][2].player = 0
    s.board[2][2].army = 1

    # Open Ant replay and set synthetic ops for this full round
    s.replay_open(seed)
    s.set_last_ops(0, [[1, 0, 0, 4, 1], [8]])  # op schema only for logging mapping
    s.set_last_ops(1, [[8]])

    # Ensure not a special round; we only check basic transitions
    s.round = 1
    update_round(s)
    s.replay_close()

    # Basic state transitions asserted on board
    # Bog penalty removed army and neutralized cell
    assert s.board[2][2].army == 0 and s.board[2][2].player == -1
    # Round incremented
    assert s.round == 2

    # Verify Ant replay JSON content against an exact expected answer
    data = json.loads(open(s.replay_file, "r", encoding="utf-8").read())
    assert isinstance(data, list) and len(data) >= 1
    first = data[0]
    # Seed present on first frame
    assert first.get("seed") == seed
    # Exact ops mapping
    assert first.get("op0") == [
        {"args": -1, "id": 0, "pos": {"x": 0, "y": 0}, "type": 1},
        {"args": -1, "id": -1, "pos": {"x": -1, "y": -1}, "type": 8},
    ]
    assert first.get("op1") == [
        {"args": -1, "id": -1, "pos": {"x": -1, "y": -1}, "type": 8},
    ]

    # Build the exact expected round_state for this scenario
    expected_ants = [
        {
            "age": 0,
            "hp": 3,  # main (1,1) army 2 -> +1 production
            "id": 0,
            "level": 0,
            "move": -1,
            "player": 0,
            "pos": {"x": 1, "y": 1},
            "status": 0,
        },
        {
            "age": 0,
            "hp": 5,  # main (3,3) army 4 -> +1 production
            "id": 1,
            "level": 0,
            "move": -1,
            "player": 1,
            "pos": {"x": 3, "y": 3},
            "status": 0,
        },
    ]
    expected_towers = [
        {"cd": 0, "id": 0, "player": 0, "pos": {"x": 1, "y": 1}, "type": 0},
        {"cd": 0, "id": 1, "player": 1, "pos": {"x": 3, "y": 3}, "type": 0},
    ]

    rs = first.get("round_state")
    assert rs["coins"] == [0, 0]
    assert rs["camps"] == [3, 5]
    assert rs["speedLv"] == [2, 2]
    assert rs["anthpLv"] == [0, 0]
    assert rs["winner"] == -1
    # Exact lists (order matters)
    assert rs["ants"] == expected_ants
    assert rs["towers"] == expected_towers
    # Pheromone has only two non-zero cells matching army on owned tiles
    ph0 = rs["pheromone"][0]
    ph1 = rs["pheromone"][1]
    assert ph0[1][1] == 3 and ph0[3][3] == 0
    assert ph1[3][3] == 5 and ph1[1][1] == 0
    # Sanity: sums match visible ants' hp per player
    assert sum(sum(row) for row in ph0) == 3
    assert sum(sum(row) for row in ph1) == 5
