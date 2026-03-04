import json
import pytest

from logic.gamestate import GameState, update_round
from logic.gamedata import CellType, MainGenerals
from logic.ant import Ant


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
    s.coin = [0, 0]

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

    # Round 4 is the first spawn round for level-1 main generals.
    s.round = 4
    update_round(s)
    s.replay_close()

    # Basic state transitions asserted on board
    # Bog penalty removed army and neutralized cell
    assert s.board[2][2].army == 0 and s.board[2][2].player == -1
    # Round incremented
    assert s.round == 5

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

    # Build the expected state after one spawn round of the ant game logic.
    # Newly spawned ants are aged once before the round ends.
    from logic.map import PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP
    expected_ants = [
        {
            "age": 1,
            "hp": 10,
            "id": 0,
            "level": 0,
            "move": -1,
            "player": 0,
            "pos": {"x": PLAYER_0_BASE_CAMP[0], "y": PLAYER_0_BASE_CAMP[1]},
            "status": Ant.Status.Alive,
        },
        {
            "age": 1,
            "hp": 10,
            "id": 1,
            "level": 0,
            "move": -1,
            "player": 1,
            "pos": {"x": PLAYER_1_BASE_CAMP[0], "y": PLAYER_1_BASE_CAMP[1]},
            "status": Ant.Status.Alive,
        },
    ]
    expected_towers = [
        {"cd": 0, "id": 0, "player": 0, "pos": {"x": 1, "y": 1}, "type": 0},
        {"cd": 0, "id": 1, "player": 1, "pos": {"x": 3, "y": 3}, "type": 0},
    ]

    rs = first.get("round_state")
    assert rs["coins"] == [1, 1]
    # base camps are independent values now
    assert rs["camps"] == [50, 50]
    assert rs["speedLv"] == [2, 2]
    assert rs["anthpLv"] == [0, 0]
    assert rs["winner"] == -1
    # Exact lists (order matters)
    assert rs["ants"] == expected_ants
    assert rs["towers"] == expected_towers
    # Pheromone grid is padded for frontend use and carries seeded values.
    ph0 = rs["pheromone"][0]
    ph1 = rs["pheromone"][1]
    assert len(ph0) >= 19 and len(ph0[0]) >= 19
    assert len(ph1) >= 19 and len(ph1[0]) >= 19
