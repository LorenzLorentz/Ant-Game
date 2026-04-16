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


class AntKind(IntEnum):
    WORKER = 0
    COMBAT = 1


class TowerType(IntEnum):
    BASIC = 0
    HEAVY = 1
    QUICK = 2
    MORTAR = 3
    PRODUCER = 4
    HEAVY_PLUS = 11
    ICE = 12
    BEWITCH = 13
    QUICK_PLUS = 21
    DOUBLE = 22
    SNIPER = 23
    MORTAR_PLUS = 31
    PULSE = 32
    MISSILE = 33
    PRODUCER_FAST = 41
    PRODUCER_SIEGE = 42
    PRODUCER_MEDIC = 43


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
    max_hp: int
    spawn_interval: int = 0
    support_interval: int = 0
    support_range: int = 0
    siege_spawn_chance: float = 0.0
    heal_amount: int = 0


@dataclass(frozen=True)
class MoveWeights:
    progress: float
    pheromone: float
    crowding: float
    expected_damage: float
    control_risk: float
    tower_pull: float
    effect_pull: float


@dataclass(frozen=True)
class WeaponStats:
    duration: int
    attack_range: int
    cooldown: int
    cost: int


TOWER_STATS = {
    TowerType.BASIC: TowerStats(5, 2.0, 1, 10),
    TowerType.HEAVY: TowerStats(12, 2.0, 1, 15),
    TowerType.QUICK: TowerStats(6, 1.0, 1, 15),
    TowerType.MORTAR: TowerStats(12, 4.0, 2, 15),
    TowerType.PRODUCER: TowerStats(0, 0.0, 0, 15, spawn_interval=10),
    TowerType.HEAVY_PLUS: TowerStats(24, 2.0, 1, 15),
    TowerType.ICE: TowerStats(12, 2.0, 2, 15),
    TowerType.BEWITCH: TowerStats(14, 2.0, 2, 15),
    TowerType.QUICK_PLUS: TowerStats(6, 0.5, 1, 15),
    TowerType.DOUBLE: TowerStats(6, 2.0, 3, 15),
    TowerType.SNIPER: TowerStats(10, 2.0, 4, 15),
    TowerType.MORTAR_PLUS: TowerStats(18, 4.0, 2, 15),
    TowerType.PULSE: TowerStats(14, 4.0, 2, 15),
    TowerType.MISSILE: TowerStats(18, 6.0, 3, 15),
    TowerType.PRODUCER_FAST: TowerStats(0, 0.0, 0, 15, spawn_interval=8),
    TowerType.PRODUCER_SIEGE: TowerStats(0, 0.0, 0, 15, spawn_interval=10, siege_spawn_chance=0.25),
    TowerType.PRODUCER_MEDIC: TowerStats(0, 0.0, 0, 15, spawn_interval=10, support_interval=4),
}

TOWER_UPGRADE_TREE = {
    TowerType.BASIC: (TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR, TowerType.PRODUCER),
    TowerType.HEAVY: (TowerType.HEAVY_PLUS, TowerType.ICE, TowerType.BEWITCH),
    TowerType.QUICK: (TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER),
    TowerType.MORTAR: (TowerType.MORTAR_PLUS, TowerType.PULSE, TowerType.MISSILE),
    TowerType.PRODUCER: (TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC),
}

SUPER_WEAPON_STATS = {
    SuperWeaponType.LIGHTNING_STORM: WeaponStats(15, 3, 35, 90),
    SuperWeaponType.EMP_BLASTER: WeaponStats(10, 3, 45, 135),
    SuperWeaponType.DEFLECTOR: WeaponStats(10, 3, 25, 60),
    SuperWeaponType.EMERGENCY_EVASION: WeaponStats(1, 3, 25, 60),
}

LIGHTNING_STORM_ANT_DAMAGE = 20
LIGHTNING_STORM_TOWER_DAMAGE = 3
LIGHTNING_STORM_TOWER_INTERVAL = 5
DEFLECTOR_PATH_ATTRACTION = 1.0
EMERGENCY_EVASION_PATH_ATTRACTION = 1.35

ANT_MAX_HP = (20, 25, 25)
COMBAT_ANT_HP = 30
ANT_KILL_REWARD = (6, 10, 14)
COMBAT_ANT_KILL_REWARD = 18
ANT_BREACH_REWARD = 10
ANT_AGE_LIMIT = 64
ANT_GENERATION_CYCLE = (4.5, 4.0, 3.5)
ANT_GENERATION_SCHEDULE = ((9, 2), (4, 1), (7, 2))
BASE_UPGRADE_COST = (200, 250)
TOWER_BUILD_BASE_COST = 15
# Legacy compatibility constant. Build costs now alternate x2 and x1.5 growth.
TOWER_BUILD_RATIO = 2
LEVEL2_TOWER_UPGRADE_COST = 60
LEVEL3_TOWER_UPGRADE_COST = 200
TOWER_DOWNGRADE_REFUND_RATIO = 0.9
BASIC_INCOME = 3
BASIC_INCOME_INTERVAL = 2
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
DEFAULT_MOVE_TEMPERATURE = 1.75
BEWITCH_MOVE_TEMPERATURE = 1.5
CROWDING_PENALTY = 1.25
RANDOM_ANT_DECAY_TURNS = 5
SPECIAL_BEHAVIOR_DECAY_TURNS = 5
ANT_TELEPORT_INTERVAL = 10
ANT_TELEPORT_RATIO = 0.1
STALL_MOVE_PENALTY = 0.35
RETREAT_MOVE_PENALTY = 0.8
TARGET_PULL_DISTANCE_SCALE = 0.18
WORKER_TOWER_ATTACK_DAMAGE = (1, 2, 4)
COMBAT_TOWER_ATTACK_DAMAGE = 5
COMBAT_SELF_DESTRUCT_DAMAGE = 10
COMBAT_SELF_DESTRUCT_RANGE = 1
COMBAT_INITIAL_EVASION = 3
WORKER_RISK_FIELD_DISTANCE_DECAY = 0.9
COMBAT_RISK_FIELD_DISTANCE_DECAY = 0.7

# Legacy compatibility constant. Combat ants no longer self-destruct based on tower HP.
TOWER_KAMIKAZE_HP_THRESHOLD = 5

MOVE_PROFILE_WEIGHTS = {
    AntKind.WORKER: MoveWeights(
        progress=1.05,
        pheromone=0.15,
        crowding=0.4,
        expected_damage=2.0,
        control_risk=1.15,
        tower_pull=0.45,
        effect_pull=0.55,
    ),
    AntKind.COMBAT: MoveWeights(
        progress=1.3,
        pheromone=0.05,
        crowding=0.15,
        expected_damage=1.1,
        control_risk=0.45,
        tower_pull=1.75,
        effect_pull=0.35,
    ),
}

SPAWN_PROFILE_WEIGHTS = (
    (AntKind.WORKER, AntBehavior.DEFAULT, 0.4),
    (AntKind.WORKER, AntBehavior.CONSERVATIVE, 0.35),
    (AntKind.WORKER, AntBehavior.RANDOM, 0.10),
    (AntKind.COMBAT, AntBehavior.DEFAULT, 0.15),
)

SPAWN_BEHAVIOR_WEIGHTS = (
    (AntBehavior.DEFAULT, 0.4),
    (AntBehavior.CONSERVATIVE, 0.3),
    (AntBehavior.RANDOM, 0.15),
)

# These anchors are adapted from the curated high-ground order used by the greedy bot,
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


def tower_build_cost_for_count(tower_count: int) -> int:
    tower_count = max(int(tower_count), 0)
    cost = TOWER_BUILD_BASE_COST * (3 ** (tower_count // 2))
    if tower_count % 2 == 1:
        cost *= 2
    return cost
