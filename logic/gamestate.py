# 本文件定义了游戏状态类，以及负责初始化将军，更新回合的函数
import random
from dataclasses import dataclass, field

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


@dataclass
class GameState:
    replay_file: str = "default_replay.json"
    round: int = 1  # 当前游戏回合数
    generals: list[Generals] = field(default_factory=list)  # 游戏中的将军列表，用于通信
    coin: list[int] = field(
        default_factory=lambda: [init_coin() for p in range(2)]
    )  # 每个玩家的金币数量列表，分别对应玩家1，玩家2
    active_super_weapon: list[SuperWeapon] = field(default_factory=list)
    super_weapon_unlocked: list[bool] = field(
        default_factory=lambda: [False, False]
    )  # 超级武器是否解锁的列表，解锁了是true，分别对应玩家1，玩家2

    super_weapon_cd: list[int] = field(
        default_factory=lambda: [-1, -1]
    )  # 超级武器的冷却回合数列表，分别对应玩家1，玩家2

    tech_level: list[list[int]] = field(
        default_factory=lambda: [[2, 0, 0, 0], [2, 0, 0, 0]]
    )
    # 科技等级列表，第一层对应玩家一，玩家二，第二层分别对应行动力，攀岩，免疫沼泽，超级武器

    rest_move_step: list[int, int] = field(default_factory=lambda: [2, 2])

    board: list[list[Cell]] = field(
        default_factory=lambda: [
            [Cell(position=[i, j]) for j in range(col)] for i in range(row)
        ]
    )  # 游戏棋盘的二维列表，每个元素是一个Cell对象

    changed_cells: list[list[int, int]] = field(default_factory=lambda: [])
    next_generals_id: int = 0
    winner: int = -1

    def find_general_position_by_id(self, general_id: int):
        for gen in self.generals:
            if gen.id == general_id:
                return gen.position
        return None

    def trans_state_to_init_json(self, player):
        result = get_single_round_replay(
            self, [[int(i / col), i % col] for i in range(row * col)], player, [8]
        )
        cell_type = ""
        for i in range(row * col):
            cell_type += str(int(self.board[int(i / col)][i % col].type))
        result["Cell_type"] = cell_type
        return result


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
            if gamestate.board[i][j].generals != None:
                gamestate.board[i][j].generals.rest_move = gamestate.board[i][
                    j
                ].generals.mobility_level
            if isinstance(gamestate.board[i][j].generals, MainGenerals):
                gamestate.board[i][j].army += gamestate.board[i][
                    j
                ].generals.produce_level
                changed.add(i * col + j)
            elif isinstance(gamestate.board[i][j].generals, SubGenerals):
                if gamestate.board[i][j].generals.player != -1:
                    gamestate.board[i][j].army += gamestate.board[i][
                        j
                    ].generals.produce_level
                    changed.add(i * col + j)
            elif isinstance(gamestate.board[i][j].generals, Farmer):
                if gamestate.board[i][j].generals.player != -1:
                    gamestate.coin[
                        gamestate.board[i][j].generals.player
                    ] += gamestate.board[i][j].generals.produce_level
            # 每25回合增兵
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
                    if (
                        gamestate.board[i][j].army == 0
                        and gamestate.board[i][j].generals == None
                    ):
                        gamestate.board[i][j].player = -1
                    changed.add(i * row + j)

    # 超级武器判定
    for weapon in gamestate.active_super_weapon:
        if weapon.type == WeaponType(0):
            for _i in range(
                max(0, weapon.position[0] - 1), min(row, weapon.position[0] + 2)
            ):
                for _j in range(
                    max(0, weapon.position[1] - 1), min(col, weapon.position[1] + 2)
                ):
                    if gamestate.board[_i][_j].army > 0:
                        gamestate.board[_i][_j].army = max(
                            0, gamestate.board[_i][_j].army - 3
                        )
                        gamestate.board[_i][_j].player = (
                            -1
                            if (
                                gamestate.board[_i][_j].army == 0
                                and gamestate.board[_i][_j].generals == None
                            )
                            else gamestate.board[_i][_j].player
                        )
                        changed.add(_i * col + _j)

    # 更新超级武器信息
    gamestate.super_weapon_cd = [
        i - 1 if i > 0 else i for i in gamestate.super_weapon_cd
    ]
    for weapon in gamestate.active_super_weapon:
        weapon.rest -= 1
    # cd和duration 减少
    for gen in gamestate.generals:
        gen.skills_cd = [i - 1 if i > 0 else i for i in gen.skills_cd]
        gen.skill_duration = [i - 1 if i > 0 else i for i in gen.skill_duration]
    # 移动步数恢复
    gamestate.rest_move_step = [gamestate.tech_level[0][0], gamestate.tech_level[1][0]]

    # 生成回放
    replay = get_single_round_replay(
        gamestate, [[int(i / col), i % col] for i in changed], -1, [8]
    )
    with open(gamestate.replay_file, "a") as f:
        f.write(str(replay).replace("'", '"') + "\n")
    f.close()

    gamestate.active_super_weapon = list(
        filter(lambda x: (x.rest > 0), gamestate.active_super_weapon)
    )

    gamestate.round += 1
