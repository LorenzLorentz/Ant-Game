import json

from logic.gamestate import GameState
from logic.gamedata import CellType, MainGenerals, SubGenerals, SkillType
from logic.general_skills import skill_activate
from logic.upgrade import tech_update, production_up
from logic.call_generals import call_generals, downgrade_tower
import logic.constant as constant


def _reset_plain(state: GameState):
    for i in range(len(state.board)):
        for j in range(len(state.board[0])):
            c = state.board[i][j]
            c.type = CellType.PLAIN
            c.player = -1
            c.generals = None
            c.army = 0


def _place_main(state: GameState, player: int, x: int, y: int, army: int = 2) -> MainGenerals:
    g = MainGenerals(id=state.next_generals_id, player=player, position=[x, y])
    state.next_generals_id += 1
    state.generals.append(g)
    cell = state.board[x][y]
    cell.player = player
    cell.generals = g
    cell.army = army
    return g


def _read_ant_frames(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def test_skill_consumes_coins_and_updates_tower_cd(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "ant.json")
    _reset_plain(s)
    g0 = _place_main(s, 0, 2, 2)
    _place_main(s, 1, 6, 6)
    # Plenty of coins
    s.coin = [1000, 1000]

    # Open replay and write baseline frame (no changes yet)
    s.replay_open(seed=42)
    s.append_ant_replay_frame(force=True)

    # Use COMMAND skill (leadership cost=30) at (2,2)
    assert skill_activate(0, [2, 2], [-1, -1], s, SkillType.COMMAND) is True
    assert s.coin[0] == 1000 - 30

    # Append a frame after the skill to capture towers delta
    s.append_ant_replay_frame(force=True)
    s.replay_close()

    frames = _read_ant_frames(s.replay_file)
    last = frames[-1]
    towers = last["round_state"]["towers"]
    # Round state carries the full tower list; the acted-upon main should
    # now expose a cooldown while the other main remains unchanged.
    assert len(towers) == 2
    t = next(t for t in towers if t["id"] == g0.id)
    assert t == {"cd": 10, "id": g0.id, "player": 0, "pos": {"x": 2, "y": 2}, "type": 0}


def test_tech_update_mobility_cost_and_speed_levels(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "ant.json")
    _reset_plain(s)
    _place_main(s, 0, 1, 1)
    _place_main(s, 1, 3, 3)
    s.coin = [1000, 1000]

    # Baseline
    s.replay_open(7)
    s.append_ant_replay_frame(force=True)

    # Upgrade army movement tech twice for player 0
    assert tech_update(0, s, 0) is True
    assert s.coin[0] == 1000 - 80
    assert s.rest_move_step[0] == 3  # +1
    assert tech_update(0, s, 0) is True
    assert s.coin[0] == 1000 - 80 - 150
    assert s.rest_move_step[0] == 5  # +2 more

    # Write frame and verify speedLv and coins in round_state
    s.append_ant_replay_frame(force=True)
    s.replay_close()

    last = _read_ant_frames(s.replay_file)[-1]
    rs = last["round_state"]
    assert rs["speedLv"] == [5, 2]
    assert rs["coins"] == [s.coin[0], s.coin[1]]
    assert len(rs["towers"]) == 2


def test_call_generals_adds_tower_and_deducts_coin(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "ant.json")
    _reset_plain(s)
    _place_main(s, 0, 0, 0)
    _place_main(s, 1, 4, 4)
    s.coin = [200, 200]
    # Prepare a player-0 owned tile without general
    pos = [2, 3]
    s.board[pos[0]][pos[1]].player = 0

    s.replay_open(9)
    s.append_ant_replay_frame(force=True)

    ok = call_generals(s, 0, pos)
    assert ok is True
    assert isinstance(s.board[pos[0]][pos[1]].generals, SubGenerals)
    # first tower costs 15 coins
    assert s.coin[0] == 185  # 200 - 15

    s.append_ant_replay_frame(force=True)
    s.replay_close()

    last = _read_ant_frames(s.replay_file)[-1]
    towers = last["round_state"]["towers"]
    assert len(towers) == 3
    t = next(t for t in towers if t["type"] == 1 and t["pos"] == {"x": pos[0], "y": pos[1]})
    assert t["player"] == 0 and t["type"] == 1
    assert t["pos"] == {"x": pos[0], "y": pos[1]}
    assert t["cd"] == 0

    # Try placing a second tower; cost should double (30 coins from remaining 185 -> 155)
    pos2 = [3, 3]
    s.board[pos2[0]][pos2[1]].player = 0
    ok = call_generals(s, 0, pos2)
    assert ok is True
    assert s.coin[0] == 155  # 185 - 30


def test_tower_downgrade_and_removal_refund(tmp_path):
    s = GameState()
    s.replay_file = str(tmp_path / "ant.json")
    _reset_plain(s)
    # place main generals to enable coin
    _place_main(s, 0, 0, 0)
    _place_main(s, 1, 6, 6)
    s.coin = [1000, 1000]

    # build two towers for player0
    pos1 = [2, 3]
    pos2 = [3, 3]
    s.board[pos1[0]][pos1[1]].player = 0
    s.board[pos2[0]][pos2[1]].player = 0
    assert call_generals(s, 0, pos1) is True
    assert call_generals(s, 0, pos2) is True
    assert s.coin[0] == 1000 - 15 - 30
    tid = s.generals[-2].id

    # remove first tower (count after removal 1)
    assert downgrade_tower(s, 0, tid) is True
    assert s.coin[0] == 1000 - 15 - 30 + int(0.8 * 15 * 2)

    # upgrade remaining tower to level3 and downgrade stepwise
    tid2 = s.generals[-1].id
    assert production_up(s.generals[-1].position, s, 0) is True
    assert production_up(s.generals[-1].position, s, 0) is True
    assert s.generals[-1].produce_level == 4
    oldc = s.coin[0]
    assert downgrade_tower(s, 0, tid2) is True
    assert s.generals[-1].produce_level == 2
    assert s.coin[0] == oldc + int(constant.lieutenant_production_T2 * 0.8)
    oldc = s.coin[0]
    assert downgrade_tower(s, 0, tid2) is True
    assert s.generals[-1].produce_level == 1
    assert s.coin[0] == oldc + int(constant.lieutenant_production_T1 * 0.8)
    oldc = s.coin[0]
    assert downgrade_tower(s, 0, tid2) is True
    assert all(g.id != tid2 for g in s.generals)
    assert s.coin[0] == oldc + int(0.8 * 15 * 1)
