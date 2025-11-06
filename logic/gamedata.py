# 本文件中定义了游戏中主要的数据结构（不包含游戏状态）
import random
from dataclasses import dataclass, field
from enum import IntEnum

import logic.constant as constant


class SkillType(IntEnum):
    SURPRISE_ATTACK = 0
    ROUT = 1
    COMMAND = 2
    DEFENCE = 3
    WEAKEN = 4
    # 玩家技能种类


class QualityType(IntEnum):
    PRODUCTION = 0
    DEFENCE = 1
    MOBILITY = 2
    # 玩家技能种类


class WeaponType(IntEnum):
    NUCLEAR_BOOM = 0
    ATTACK_ENHANCE = 1
    TRANSMISSION = 2
    TIME_STOP = 3
    # 超级武器种类


class CellType(IntEnum):
    PLAIN = 0
    BOG = 1
    MOUNTAIN = 2
    # 地形种类
    # 注：将军不算地形，而属于地形的元素


class TechType(IntEnum):
    MOBILITY = 0
    CLIMB = 1
    IMMUNE = 2
    UNLOCK = 3


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def init_coin():
    return 0


def init_cell_type():
    seed = random.random()
    if seed <= constant.bog_percent:
        return CellType(1)
    if seed > 1 - constant.mountain_percent:
        return CellType(2)
    return CellType(0)


# 随机生成cell的种类


@dataclass
class Skill:
    type: SkillType = SkillType(0)
    cd: int = 0


@dataclass
class SuperWeapon:
    type: WeaponType = WeaponType(0)
    player: int = -1
    cd: int = 0
    rest: int = 0
    position: list = field(default_factory=lambda: [0, 0])


@dataclass
class Generals:
    id: int = 0
    player: int = -1
    position: list = field(default_factory=lambda: [0, 0])
    skills_cd: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    skill_duration: list[int] = field(default_factory=lambda: [0, 0, 0])
    rest_move: int = 1


# 注意，xxx_level代表具体数值，例如produce_level=1，意味着每回合生产1


@dataclass
class Farmer(Generals):
    produce_level: int = 1
    defense_level: int = 1
    mobility_level: int = 0  # 注意，此属性不可升级，仅用于调用


@dataclass
class MainGenerals(Generals):
    produce_level: int = 1
    defense_level: int = 1
    mobility_level: int = 1
    skills: list[Skill] = field(
        default_factory=lambda: [
            Skill(SkillType(0), 5),
            Skill(SkillType(1), 10),
            Skill(SkillType(2), 10),
            Skill(SkillType(3), 10),
            Skill(SkillType(4), 10),
        ]
    )


@dataclass
class SubGenerals(Generals):
    produce_level: int = 1
    defense_level: int = 1
    mobility_level: int = 1
    skills: list[Skill] = field(
        default_factory=lambda: [
            Skill(SkillType(0), 5),
            Skill(SkillType(1), 10),
            Skill(SkillType(2), 10),
            Skill(SkillType(3), 10),
            Skill(SkillType(4), 10),
        ]
    )


@dataclass
class Cell:
    position: list[int, int] = field(default_factory=lambda: [0, 0])
    type: CellType = field(default_factory=init_cell_type)
    player: int = -1
    generals: Generals = None
    weapon_activate: list[SuperWeapon] = field(default_factory=list)
    army: int = 0  # 格子上的军队数目
