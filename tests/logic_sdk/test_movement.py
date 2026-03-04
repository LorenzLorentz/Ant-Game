import math
import pytest

from logic.gamestate import update_round
from logic.movement import calculate_new_pos, army_move, check_general_movement, general_move
from logic.gamedata import Direction, CellType, MainGenerals


def test_calculate_new_pos_and_bounds():
    assert calculate_new_pos([5, 5], Direction.UP) == (4, 5)
    assert calculate_new_pos([5, 5], Direction.DOWN) == (6, 5)
    assert calculate_new_pos([5, 5], Direction.LEFT) == (5, 4)
    assert calculate_new_pos([5, 5], Direction.RIGHT) == (5, 6)
    # Out of range returns (-1, -1)
    assert calculate_new_pos([0, 0], Direction.UP) == (-1, -1)
    assert calculate_new_pos([0, 0], Direction.LEFT) == (-1, -1)


def test_army_move_to_neutral_and_friendly(plain_state):
    s = plain_state
    s.board[0][0].player = 0
    s.board[0][0].army = 10

    # Move 3 to neutral (0,1)
    ok = army_move([0, 0], s, 0, Direction.RIGHT, 3)
    assert ok is True
    assert s.board[0][1].player == 0
    # neutral defence = 1, attack = 1, vs = 3 -> ceil(3)=3
    assert s.board[0][1].army == 3
    assert s.board[0][0].army == 7
    assert s.rest_move_step[0] == 1

    # Move 2 to friendly (0,1) from (0,0)
    ok = army_move([0, 0], s, 0, Direction.RIGHT, 2)
    assert ok is True
    assert s.board[0][1].army == 5  # 3 + 2
    assert s.board[0][0].army == 5  # 7 - 2
    assert s.rest_move_step[0] == 0


def test_army_move_vs_enemy_and_general_flip(plain_state):
    s = plain_state
    # origin
    s.board[0][0].player = 0
    s.board[0][0].army = 10
    # enemy dest with a general present
    s.board[0][1].player = 1
    s.board[0][1].army = 2

    # put an enemy general at dest; should flip owner if captured
    from tests.conftest import place_general

    enemy_gen = place_general(s, "sub", 0, 1, 1)
    ok = army_move([0, 0], s, 0, Direction.RIGHT, 3)
    assert ok is True
    assert s.board[0][1].player == 0
    assert enemy_gen.player == 0  # general flipped
    assert s.board[0][0].army == 7


def test_army_move_invalid_cases(plain_state):
    s = plain_state
    # No control on cell
    s.board[1][1].player = -1
    s.board[1][1].army = 5
    assert army_move([1, 1], s, 0, Direction.RIGHT, 1) is False

    # Not enough army to move
    s.board[1][1].player = 0
    s.board[1][1].army = 1
    assert army_move([1, 1], s, 0, Direction.RIGHT, 1) is False

    # Invalid num
    s.board[1][1].army = 5
    assert army_move([1, 1], s, 0, Direction.RIGHT, 0) is False

    # Mountain blocked without mountaineering
    s.board[1][2].type = CellType.MOUNTAIN
    assert army_move([1, 1], s, 0, Direction.RIGHT, 1) is False

    # No remaining move steps
    s.board[2][2].player = 0
    s.board[2][2].army = 5
    s.rest_move_step[0] = 0
    assert army_move([2, 2], s, 0, Direction.RIGHT, 1) is False


def test_army_move_blocked_by_superweapons(plain_state):
    s = plain_state
    s.board[3][3].player = 0
    s.board[3][3].army = 5

    # Transmission stun at same cell by same player
    from logic.gamedata import SuperWeapon, WeaponType

    s.active_super_weapon.append(
        SuperWeapon(type=WeaponType.TRANSMISSION, player=0, cd=1, rest=1, position=[3, 3])
    )
    assert army_move([3, 3], s, 0, Direction.RIGHT, 1) is False

    # Replace with enemy time stop nearby
    s.active_super_weapon.clear()
    s.active_super_weapon.append(
        SuperWeapon(type=WeaponType.TIME_STOP, player=1, cd=1, rest=1, position=[3, 4])
    )
    assert army_move([3, 3], s, 0, Direction.RIGHT, 1) is False


def test_check_and_general_move(plain_state):
    s = plain_state
    # Place friendly corridor
    s.board[5][5].player = 0
    s.board[5][6].player = 0
    s.board[5][7].player = 0
    # Place general at (5,5)
    from tests.conftest import place_general

    gen = place_general(s, "main", 5, 5, 0)
    gen.rest_move = 2

    # Legal move to (5,7) through friendly cells
    assert check_general_movement([5, 5], s, 0, [5, 7]) is True
    assert general_move([5, 5], s, 0, [5, 7]) is True
    assert s.board[5][7].generals is not None
    assert s.board[5][7].generals.id == gen.id
    assert s.board[5][7].generals.rest_move == 0
    assert s.board[5][5].generals is None

    # Occupied destination blocks movement
    s.board[5][6].generals = gen
    s.board[5][7].generals = None
    assert check_general_movement([5, 6], s, 0, [5, 7]) is False

    # Mountain without tech blocks path
    s.board[5][6].generals = None
    s.board[5][6].type = CellType.MOUNTAIN
    s.board[5][6].player = 0
    s.board[5][7].player = 0
    assert check_general_movement([5, 5], s, 0, [5, 7]) is False


def test_movement_bounds_and_ant_logging(plain_state):
    s = plain_state
    # moving army off board should fail and not record a move
    s.board[0][0].player = 0
    s.board[0][0].army = 5
    assert army_move([0, 0], s, 0, Direction.UP, 1) is False
    assert getattr(s, "_ant_moves", {}) == {}

    # legal move should be recorded
    s.board[0][1].player = 0
    s.board[0][1].army = 1
    s.rest_move_step[0] = 1
    ok = army_move([0, 0], s, 0, Direction.RIGHT, 2)
    assert ok is True
    # after an army move there is no ant creation yet
    ants = s._build_ants()
    assert ants == []


def test_ant_kill_count_increment():
    """When an ant steps on an enemy general it counts as a kill."""
    from logic.gamestate import GameState
    from logic.map import PLAYER_0_BASE_CAMP
    from logic.ant import Ant
    from logic.gamedata import MainGenerals

    s = GameState()
    # give player0 a general at base camp
    s.board[PLAYER_0_BASE_CAMP[0]][PLAYER_0_BASE_CAMP[1]].generals = MainGenerals(player=0, id=0, position=PLAYER_0_BASE_CAMP)
    s.board[PLAYER_0_BASE_CAMP[0]][PLAYER_0_BASE_CAMP[1]].player = 0
    # spawn an ant belonging to player1 directly on that tile
    ant = Ant(player=1, id=0, x=PLAYER_0_BASE_CAMP[0], y=PLAYER_0_BASE_CAMP[1], level=0)
    s.ants.append(ant)
    # perform attack and manage phases
    s._ant_attack()
    s._ant_manage()
    assert s.kill_count[0] == 1


def test_ant_generation_and_movement():
    # when update_round is called main generals should spawn ants and the
    # ants should move toward the opponent base on the subsequent round.
    # Level-1 main generals generate ants on rounds divisible by 4.
    from logic.gamestate import GameState
    from logic.map import PLAYER_0_BASE_CAMP
    from tests.conftest import place_general

    s = GameState()
    place_general(s, "main", 0, 0, 0)
    place_general(s, "main", 1, 1, 1)
    s.round = 4

    # round 4 generates ants at the predefined base camp coords
    update_round(s)
    assert len(s.ants) >= 1
    pos0 = [ant.pos for ant in s.ants if ant.player == 0][0]
    assert pos0 == PLAYER_0_BASE_CAMP

    # second round the ant should move away from the base camp
    update_round(s)
    pos0_new = [ant.pos for ant in s.ants if ant.player == 0][0]
    assert pos0_new != PLAYER_0_BASE_CAMP


def test_ant_no_trail_or_duplicates():
    """After multiple rounds the number of ants should stay constant and
    no ant id should show up twice in the JSON snapshot.  This guards
    against backend leaking old positions into the state (which the Unity
    client was exhibiting as the trail-of-ants bug)."""
    from logic.gamestate import GameState
    from tests.conftest import place_general

    s = GameState()
    # give each player a main general so that ants spawn
    place_general(s, "main", 0, 0, 0)
    place_general(s, "main", 1, 1, 1)
    s.round = 4

    seen_positions = {}
    # simulate 5 rounds and record snapshots
    for _ in range(5):
        update_round(s)
        ants_json = s._build_round_state()["ants"]
        ids = {a["id"] for a in ants_json}
        # every id should be unique per snapshot
        assert len(ids) == len(ants_json)
        # compare against previous positions
        for a in ants_json:
            aid = a["id"]
            pos = (a["pos"]["x"], a["pos"]["y"])
            if aid in seen_positions:
                # ensure position actually changed (or ant died)
                assert pos != seen_positions[aid]
            seen_positions[aid] = pos
        # also make sure no internal path data survived the snapshot
        for ant in s.ants:
            assert ant.path == []


def test_pheromone_seed_and_size(tmp_path):
    from logic.gamestate import GameState
    from logic.constant import row, col
    s = GameState()
    # before replay_open pheromone may be None
    assert s._pheromone is None
    s.replay_file = str(tmp_path / "dummy.json")
    s.replay_open(12345)
    # _pheromone should now be a 2×row×col structure
    assert s._pheromone is not None
    assert len(s._pheromone) == 2
    assert len(s._pheromone[0]) == row
    assert len(s._pheromone[0][0]) == col
    # values should be >0 because of +8 constant
    assert all(v > 0 for plane in s._pheromone for line in plane for v in line)
    # building pheromone returns a padded grid at least 19×19
    ph = s._build_pheromone()
    assert len(ph[0]) >= 19 and len(ph[0][0]) >= 19
