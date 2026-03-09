from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

MAX_ROUND = 512
EDGE = 10
MAP_SIZE = 2 * EDGE - 1
PLAYER_COUNT = 2
BASE_HP = 50

PLAYER_BASES = ((2, EDGE - 1), (MAP_SIZE - 3, EDGE - 1))

OFFSET = (
    ((0, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (1, 1)),
    ((-1, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (0, 1)),
)


class Terrain(IntEnum):
    VOID = -1
    PATH = 0
    BARRIER = 1
    PLAYER0_HIGHLAND = 2
    PLAYER1_HIGHLAND = 3


MAP_PROPERTY = (
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, 0, 0, 1, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, -1, -1, -1, -1),
    (-1, -1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1),
    (0, 0, 2, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0),
    (0, 0, 0, 2, 0, 0, 2, 2, 0, 2, 0, 2, 2, 0, 0, 2, 0, 0, 0),
    (0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 2, 2, 0),
    (0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0),
    (0, 1, 3, 0, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 3, 0, 3, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0),
    (0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3, 0),
    (0, 3, 0, 0, 0, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0, 3, 0),
    (0, 0, 3, 3, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3, 3, 0, 0),
    (-1, 0, 0, 3, 0, 1, 1, 0, 0, 3, 0, 0, 1, 1, 0, 3, 0, 0, -1),
    (-1, -1, -1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, -1, -1, -1),
    (-1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 1, 1, 0, 0, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
)

PATH_CELLS = tuple(
    (x, y)
    for x in range(MAP_SIZE)
    for y in range(MAP_SIZE)
    if MAP_PROPERTY[x][y] == Terrain.PATH
)
HIGHLAND_CELLS = {
    0: tuple(
        (x, y)
        for x in range(MAP_SIZE)
        for y in range(MAP_SIZE)
        if MAP_PROPERTY[x][y] == Terrain.PLAYER0_HIGHLAND
    ),
    1: tuple(
        (x, y)
        for x in range(MAP_SIZE)
        for y in range(MAP_SIZE)
        if MAP_PROPERTY[x][y] == Terrain.PLAYER1_HIGHLAND
    ),
}
VALID_CELLS = PATH_CELLS + HIGHLAND_CELLS[0] + HIGHLAND_CELLS[1] + PLAYER_BASES


class AntStatus(IntEnum):
    ALIVE = 0
    SUCCESS = 1
    FAIL = 2
    TOO_OLD = 3
    FROZEN = 4


class AntBehavior(IntEnum):
    DEFAULT = 0
    CONSERVATIVE = 1
    RANDOM = 2
    BEWITCHED = 3
    CONTROL_FREE = 4


class TowerType(IntEnum):
    BASIC = 0
    HEAVY = 1
    QUICK = 2
    MORTAR = 3
    HEAVY_PLUS = 11
    ICE = 12
    CANNON = 13
    QUICK_PLUS = 21
    DOUBLE = 22
    SNIPER = 23
    MORTAR_PLUS = 31
    PULSE = 32
    MISSILE = 33


class SuperWeaponType(IntEnum):
    LIGHTNING_STORM = 1
    EMP_BLASTER = 2
    DEFLECTOR = 3
    EMERGENCY_EVASION = 4


class OperationType(IntEnum):
    BUILD_TOWER = 11
    UPGRADE_TOWER = 12
    DOWNGRADE_TOWER = 13
    USE_LIGHTNING_STORM = 21
    USE_EMP_BLASTER = 22
    USE_DEFLECTOR = 23
    USE_EMERGENCY_EVASION = 24
    UPGRADE_GENERATION_SPEED = 31
    UPGRADE_GENERATED_ANT = 32


@dataclass(frozen=True)
class TowerStats:
    damage: int
    speed: float
    attack_range: int


@dataclass(frozen=True)
class WeaponStats:
    duration: int
    attack_range: int
    cooldown: int
    cost: int


TOWER_STATS = {
    TowerType.BASIC: TowerStats(5, 2.0, 2),
    TowerType.HEAVY: TowerStats(20, 2.0, 2),
    TowerType.QUICK: TowerStats(6, 1.0, 3),
    TowerType.MORTAR: TowerStats(16, 4.0, 3),
    TowerType.HEAVY_PLUS: TowerStats(35, 2.0, 3),
    TowerType.ICE: TowerStats(15, 2.0, 2),
    TowerType.CANNON: TowerStats(10, 3.0, 3),
    TowerType.QUICK_PLUS: TowerStats(8, 0.5, 3),
    TowerType.DOUBLE: TowerStats(7, 1.0, 4),
    TowerType.SNIPER: TowerStats(15, 2.0, 6),
    TowerType.MORTAR_PLUS: TowerStats(35, 4.0, 4),
    TowerType.PULSE: TowerStats(12, 3.0, 2),
    TowerType.MISSILE: TowerStats(45, 6.0, 5),
}

TOWER_UPGRADE_TREE = {
    TowerType.BASIC: (TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR),
    TowerType.HEAVY: (TowerType.HEAVY_PLUS, TowerType.ICE, TowerType.CANNON),
    TowerType.QUICK: (TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER),
    TowerType.MORTAR: (TowerType.MORTAR_PLUS, TowerType.PULSE, TowerType.MISSILE),
}

SUPER_WEAPON_STATS = {
    SuperWeaponType.LIGHTNING_STORM: WeaponStats(20, 3, 100, 150),
    SuperWeaponType.EMP_BLASTER: WeaponStats(20, 3, 100, 150),
    SuperWeaponType.DEFLECTOR: WeaponStats(10, 3, 50, 100),
    SuperWeaponType.EMERGENCY_EVASION: WeaponStats(1, 3, 50, 100),
}

ANT_MAX_HP = (10, 25, 50)
ANT_KILL_REWARD = (3, 5, 7)
ANT_AGE_LIMIT = 32
ANT_GENERATION_CYCLE = (4, 2, 1)
BASE_UPGRADE_COST = (200, 250)
TOWER_BUILD_BASE_COST = 15
TOWER_BUILD_RATIO = 2
LEVEL2_TOWER_UPGRADE_COST = 60
LEVEL3_TOWER_UPGRADE_COST = 200
TOWER_DOWNGRADE_REFUND_RATIO = 0.8
BASIC_INCOME = 1
INITIAL_COINS = 50
# Pheromone: stored as int, real_value = pheromone_int / PHEROMONE_SCALE
PHEROMONE_SCALE = 10000
PHEROMONE_INIT = 10.0
PHEROMONE_ATTENUATION = 0.97
PHEROMONE_FLOOR = 0.0
PHEROMONE_SUCCESS_BONUS = 10.0
PHEROMONE_FAIL_BONUS = -5.0
PHEROMONE_TOO_OLD_BONUS = -3.0
# Integer versions for deterministic computation
PHEROMONE_INIT_INT = 80000  # base for LCG init
PHEROMONE_SUCCESS_BONUS_INT = 100000
PHEROMONE_FAIL_BONUS_INT = -50000
PHEROMONE_TOO_OLD_BONUS_INT = -30000
LAMBDA_NUM = 97
LAMBDA_DENOM = 100
TAU_BASE_ADD_INT = 3000  # 0.03 * 10 * PHEROMONE_SCALE
MAX_ACTIONS = 96
DEFAULT_MOVE_TEMPERATURE = 4.0
BEWITCH_MOVE_TEMPERATURE = 1.5
CROWDING_PENALTY = 1.25
RANDOM_ANT_DECAY_TURNS = 5
ANT_TELEPORT_INTERVAL = 10
ANT_TELEPORT_RATIO = 0.2

SPAWN_BEHAVIOR_WEIGHTS = (
    (AntBehavior.DEFAULT, 0.5),
    (AntBehavior.CONSERVATIVE, 0.2),
    (AntBehavior.RANDOM, 0.15),
    (AntBehavior.CONTROL_FREE, 0.15),
)

# These anchors are adapted from the curated high-ground order used by the expert bot,
# but exposed with descriptive names rather than opaque slot codes.
STRATEGIC_BUILD_ORDER = {
    0: (
        (2, 9), (4, 9), (5, 9), (5, 7), (6, 9), (5, 11), (5, 6), (6, 7), (6, 11),
        (5, 12), (4, 3), (5, 3), (7, 8), (7, 10), (4, 15), (5, 15), (4, 2), (6, 4),
        (7, 5), (8, 7), (8, 11), (7, 13), (6, 14), (4, 16), (6, 1), (6, 2), (6, 16),
        (6, 17), (7, 1), (8, 4), (8, 14), (7, 17), (8, 2), (8, 16), (3, 9),
    ),
    1: (
        (16, 9), (14, 9), (13, 9), (13, 7), (12, 9), (13, 11), (12, 6), (12, 7), (12, 11),
        (12, 12), (14, 3), (13, 3), (10, 8), (10, 10), (14, 15), (13, 15), (13, 2), (11, 4),
        (11, 5), (10, 7), (10, 11), (11, 13), (11, 14), (13, 16), (12, 1), (11, 2), (11, 16),
        (12, 17), (11, 1), (9, 4), (9, 14), (11, 17), (9, 2), (9, 16), (15, 9),
    ),
}

CENTERLINE_WEIGHTS = {
    (2, 9): 1.0,
    (4, 9): 1.1,
    (5, 9): 1.15,
    (6, 9): 1.2,
    (16, 9): 1.0,
    (14, 9): 1.1,
    (13, 9): 1.15,
    (12, 9): 1.2,
}
