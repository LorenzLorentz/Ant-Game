# 本文件定义了游戏状态类，以及负责初始化将军，更新回合的函数
import random
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from logic.call_generals import call_generals
from logic.constant import *
from logic.gamedata import (
    Cell,
    CellType,
    Direction,
    Farmer,
    Generals,
    MainGenerals,
    SkillType,
    SubGenerals,
    SuperWeapon,
    WeaponType,
    init_coin,
)
from logic.general_skills import skill_activate
from logic.generate_round_replay import get_single_round_replay
from logic.movement import army_move, general_move
from logic.super_weapons import *
from logic.upgrade import *


# =========================
# AntWar-style replay writer (JSON array)
# - seed only appears in the first element
# =========================
class AntReplayWriter:
    def __init__(self, path: str, seed: int):
        self.fp = open(path, "w", encoding="utf-8")
        self.first = True
        self.seed = int(seed)
        self.seed_written = False
        self.fp.write("[\n")

    def append(self, frame: dict) -> None:
        if not self.seed_written:
            frame = dict(frame)
            frame["seed"] = self.seed
            self.seed_written = True

        if not self.first:
            self.fp.write(",\n")
        self.fp.write(json.dumps(frame, ensure_ascii=False))
        self.first = False
        self.fp.flush()

    def close(self) -> None:
        self.fp.write("\n]\n")
        self.fp.close()


@dataclass
class GameState:
    replay_file: str = "default_replay.json"
    round: int = 1  # 当前游戏回合数
    generals: list[Generals] = field(default_factory=list)  # 游戏中的将军列表，用于通信
    coin: list[int] = field(default_factory=lambda: [init_coin() for p in range(2)])  # 每个玩家的金币数量列表
    active_super_weapon: list[SuperWeapon] = field(default_factory=list)
    super_weapon_unlocked: list[bool] = field(default_factory=lambda: [False, False])
    super_weapon_cd: list[int] = field(default_factory=lambda: [-1, -1])
    tech_level: list[list[int]] = field(default_factory=lambda: [[2, 0, 0, 0], [2, 0, 0, 0]])
    # 科技等级列表，第一层对应玩家一，玩家二，第二层分别对应行动力，攀岩，免疫沼泽，超级武器
    rest_move_step: list[int] = field(default_factory=lambda: [2, 2])

    board: list[list[Cell]] = field(default_factory=lambda: [[Cell(position=[i, j]) for j in range(col)] for i in range(row)])
    changed_cells: list[list[int]] = field(default_factory=lambda: [])
    next_generals_id: int = 0
    winner: int = -1

    # ---- replay support ----
    replay_seed: int = 0
    _ant_replay: Optional[AntReplayWriter] = field(default=None, init=False, repr=False)

    # store last ops of each player for the current full round
    _last_ops: list[list[list[int]]] = field(default_factory=lambda: [[], []], init=False, repr=False)

    # tower delta tracking (so towers list can be "delta-like")
    _prev_towers_snapshot: Dict[int, tuple] = field(default_factory=dict, init=False, repr=False)

    def replay_open(self, seed: int) -> None:
        self.replay_seed = int(seed)
        self._ant_replay = AntReplayWriter(self.replay_file, self.replay_seed)

    def replay_close(self) -> None:
        if self._ant_replay is not None:
            self._ant_replay.close()
            self._ant_replay = None

    def set_last_ops(self, player: int, ops: list[list[int]]) -> None:
        if player in (0, 1):
            self._last_ops[player] = ops

    def find_general_position_by_id(self, general_id: int):
        for gen in self.generals:
            if gen.id == general_id:
                return gen.position
        return None

    def trans_state_to_init_json(self, player):
        """
        这是你现有 AI/前端使用的 JSON rep。
        保持不变。
        """
        result = get_single_round_replay(
            self, [[int(i / col), i % col] for i in range(row * col)], player, [8]
        )
        cell_type = ""
        for i in range(row * col):
            cell_type += str(int(self.board[int(i / col)][i % col].type))
        result["Cell_type"] = cell_type
        return result

    # =========================
    # AntWar replay: op + round_state mapping
    # =========================
    def _op_to_ant_op(self, op: list[int]) -> dict:
        """
        AntWar replay op schema requires:
          {"args":..., "id":..., "pos":{"x","y"}, "type":...}
        这里做 best-effort 映射：
        - 若 op 至少有 3 个数，则把 op[1],op[2] 当作 pos
        - 若 op 至少有 2 个数，则把 op[1] 当作 id
        """
        t = int(op[0]) if op else -1
        args = -1
        tid = -1
        pos = {"x": -1, "y": -1}

        if len(op) >= 2:
            tid = int(op[1])
        if len(op) >= 3:
            pos["x"], pos["y"] = int(op[1]), int(op[2])

        return {"args": args, "id": tid, "pos": pos, "type": t}

    def _build_pheromone(self) -> list:
        """
        pheromone: [2][row][col]
        用“占领格 army 强度”派生：己方占领则为 army，否则 0。
        """
        pheromone = []
        for p in (0, 1):
            grid = []
            for i in range(row):
                line = []
                for j in range(col):
                    c = self.board[i][j]
                    line.append(int(c.army) if c.player == p else 0)
                grid.append(line)
            pheromone.append(grid)
        return pheromone

    def _build_ants(self) -> list:
        """
        ants: list of dict (full list)
        用“所有被占领且 army>0 的格子”派生 ants。
        """
        ants = []
        aid = 0
        for i in range(row):
            for j in range(col):
                c = self.board[i][j]
                if c.player != -1 and c.army > 0:
                    ants.append({
                        "age": 0,
                        "hp": int(c.army),
                        "id": aid,
                        "level": 0,
                        "move": -1,
                        "player": int(c.player),
                        "pos": {"x": int(i), "y": int(j)},
                        "status": 0,
                    })
                    aid += 1
        return ants

    def _base_hp_from_main_general(self) -> list[int]:
        """
        camps: 用各自 MainGenerals 所在格的 army 作为“基地血量”
        """
        camps = [50, 50]
        for p in (0, 1):
            for g in self.generals:
                if isinstance(g, MainGenerals) and g.player == p:
                    x, y = g.position
                    camps[p] = int(self.board[x][y].army)
                    break
        return camps

    def _build_towers_delta(self) -> list:
        """
        towers: 文档说是“回合内新建/变化的塔”(delta)。
        我们把 generals 映射为 towers，并做一次 delta 比较输出。
        """
        current_snapshot: Dict[int, tuple] = {}
        delta = []

        for g in self.generals:
            if g.player == -1:
                continue
            x, y = g.position
            ttype = 0 if isinstance(g, MainGenerals) else (1 if isinstance(g, SubGenerals) else 2)

            # cd: 取最小正 cd（否则 0）
            cd = 0
            try:
                positives = [int(v) for v in getattr(g, "skills_cd", []) if int(v) > 0]
                cd = min(positives) if positives else 0
            except Exception:
                cd = 0

            snap = (int(g.player), int(x), int(y), int(ttype), int(cd))
            gid = int(g.id)
            current_snapshot[gid] = snap

            if gid not in self._prev_towers_snapshot or self._prev_towers_snapshot[gid] != snap:
                delta.append({
                    "cd": int(cd),
                    "id": int(gid),
                    "player": int(g.player),
                    "pos": {"x": int(x), "y": int(y)},
                    "type": int(ttype),
                })

        # update snapshot
        self._prev_towers_snapshot = current_snapshot
        return delta

    def _build_round_state(self) -> dict:
        """
        用你现有 GameState 派生 AntWar round_state 必需字段集合。
        """
        coins = [int(self.coin[0]), int(self.coin[1])]
        camps = self._base_hp_from_main_general()

        # tech_level: [行动力, 攀岩, 免疫沼泽, 超级武器]
        speedLv = [int(self.tech_level[0][0]), int(self.tech_level[1][0])]
        anthpLv = [int(self.tech_level[0][1]), int(self.tech_level[1][1])]

        rs = {
            "anthpLv": anthpLv,
            "ants": self._build_ants(),
            "camps": camps,
            "coins": coins,
            "error": "",
            "message": "[,]",
            "pheromone": self._build_pheromone(),
            "speedLv": speedLv,
            "towers": self._build_towers_delta(),
            "winner": int(self.winner),
        }
        return rs

    def append_ant_replay_frame(self, force: bool = False) -> None:
        """
        写入一帧 AntWar replay:
          {seed?, op0, op1, round_state}
        - 正常情况下由 update_round 在每个完整回合末调用
        - force=True: 即使某一方 ops 缺失也强制写一帧（用于异常/终局补帧）
        """
        if self._ant_replay is None:
            return

        op0 = [self._op_to_ant_op(op) for op in (self._last_ops[0] or []) if op]
        op1 = [self._op_to_ant_op(op) for op in (self._last_ops[1] or []) if op]

        if (not force) and (len(op0) == 0 and len(op1) == 0):
            return

        frame = {
            "op0": op0 if op0 else [{"args": -1, "id": -1, "pos": {"x": -1, "y": -1}, "type": 8}],
            "op1": op1 if op1 else [{"args": -1, "id": -1, "pos": {"x": -1, "y": -1}, "type": 8}],
            "round_state": self._build_round_state(),
        }
        self._ant_replay.append(frame)

        # clear ops after write
        self._last_ops = [[], []]


def init_generals(gamestate: GameState):
    # init random position
    positions = []
    for i in range(row):
        for j in range(col):
            if gamestate.board[i][j].type == CellType(0):
                positions.append([i, j])
    random.shuffle(positions)

    # generate main generals
    for player in range(2):
        gen = MainGenerals(player=player, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].player = player

    # generate sub generals
    for player in range(subgen_num):
        gen = SubGenerals(player=-1, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].army = random.randint(10, 20)

    # generate farmer
    for i in range(farmer_num):
        gen = Farmer(player=-1, produce_level=1, id=gamestate.next_generals_id)
        gamestate.next_generals_id += 1
        pos = positions.pop()
        gen.position[0] = pos[0]
        gen.position[1] = pos[1]
        gamestate.generals.append(gen)
        gamestate.board[pos[0]][pos[1]].generals = gen
        gamestate.board[pos[0]][pos[1]].army = random.randint(3, 5)


def update_round(gamestate: GameState):
    changed = set()
    for i in range(row):
        for j in range(col):
            # 将军
            if gamestate.board[i][j].generals is not None:
                gamestate.board[i][j].generals.rest_move = gamestate.board[i][j].generals.mobility_level

            if isinstance(gamestate.board[i][j].generals, MainGenerals):
                gamestate.board[i][j].army += gamestate.board[i][j].generals.produce_level
                changed.add(i * col + j)
            elif isinstance(gamestate.board[i][j].generals, SubGenerals):
                if gamestate.board[i][j].generals.player != -1:
                    gamestate.board[i][j].army += gamestate.board[i][j].generals.produce_level
                    changed.add(i * col + j)
            elif isinstance(gamestate.board[i][j].generals, Farmer):
                if gamestate.board[i][j].generals.player != -1:
                    gamestate.coin[gamestate.board[i][j].generals.player] += gamestate.board[i][j].generals.produce_level

            # 每10回合增兵
            if gamestate.round % 10 == 0:
                if gamestate.board[i][j].player != -1:
                    gamestate.board[i][j].army += 1
                    changed.add(i * col + j)

            # 沼泽减兵
            if (
                gamestate.board[i][j].type == CellType(1)
                and gamestate.board[i][j].player != -1
                and gamestate.board[i][j].army > 0
            ):
                if gamestate.tech_level[gamestate.board[i][j].player][2] == 0:
                    gamestate.board[i][j].army -= 1
                    if gamestate.board[i][j].army == 0 and gamestate.board[i][j].generals is None:
                        gamestate.board[i][j].player = -1
                    # FIX: 原来是 i * row + j，应该是 i * col + j
                    changed.add(i * col + j)

    # 超级武器判定
    for weapon in gamestate.active_super_weapon:
        if weapon.type == WeaponType(0):
            for _i in range(max(0, weapon.position[0] - 1), min(row, weapon.position[0] + 2)):
                for _j in range(max(0, weapon.position[1] - 1), min(col, weapon.position[1] + 2)):
                    if gamestate.board[_i][_j].army > 0:
                        gamestate.board[_i][_j].army = max(0, gamestate.board[_i][_j].army - 3)
                        gamestate.board[_i][_j].player = (
                            -1
                            if (gamestate.board[_i][_j].army == 0 and gamestate.board[_i][_j].generals is None)
                            else gamestate.board[_i][_j].player
                        )
                        changed.add(_i * col + _j)

    # 更新超级武器信息
    gamestate.super_weapon_cd = [i - 1 if i > 0 else i for i in gamestate.super_weapon_cd]
    for weapon in gamestate.active_super_weapon:
        weapon.rest -= 1

    # cd和duration 减少
    for gen in gamestate.generals:
        gen.skills_cd = [i - 1 if i > 0 else i for i in gen.skills_cd]
        gen.skill_duration = [i - 1 if i > 0 else i for i in gen.skill_duration]

    # 移动步数恢复
    gamestate.rest_move_step = [gamestate.tech_level[0][0], gamestate.tech_level[1][0]]

    # 你原来的增量 replay dict 仍然可以生成（供 AI/调试），但不再直接写入 replay_file
    _ = get_single_round_replay(gamestate, [[int(i / col), i % col] for i in changed], -1, [8])

    gamestate.active_super_weapon = list(filter(lambda x: (x.rest > 0), gamestate.active_super_weapon))

    # 在回合数+1之前写入 AntWar replay frame
    gamestate.append_ant_replay_frame(force=False)

    gamestate.round += 1
