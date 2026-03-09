from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

MAX_ROUND = 512
EDGE = 10
MAP_SIZE = 2 * EDGE - 1


class BuildingType(IntEnum):
    EMPTY = 0
    TOWER = 1
    BASE = 2


class AntState(IntEnum):
    ALIVE = 0
    SUCCESS = 1
    FAIL = 2
    TOO_OLD = 3
    FROZEN = 4


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
    COUNT = 5


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

OFFSET = (
    ((0, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (1, 1)),
    ((-1, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (0, 1)),
)

COIN_INIT = 50
BASIC_INCOME = 1
TOWER_BUILD_PRICE_BASE = 15
TOWER_BUILD_PRICE_RATIO = 2
LEVEL2_TOWER_UPGRADE_PRICE = 60
LEVEL3_TOWER_UPGRADE_PRICE = 200
TOWER_DOWNGRADE_REFUND_RATIO = 0.8
LEVEL2_BASE_UPGRADE_PRICE = 200
LEVEL3_BASE_UPGRADE_PRICE = 250
# Pheromone: stored as int, real_value = pheromone_int / PHEROMONE_SCALE
PHEROMONE_SCALE = 10000
PHEROMONE_INIT = 10.0
PHEROMONE_INIT_INT = 80000
PHEROMONE_MIN = 0
PHEROMONE_ATTENUATING_RATIO = 0.97
LAMBDA_NUM = 97
LAMBDA_DENOM = 100
TAU_BASE_ADD_INT = 3000

TOWER_INFO = {
    TowerType.BASIC: (5, 2.0, 2),
    TowerType.HEAVY: (20, 2.0, 2),
    TowerType.QUICK: (6, 1.0, 3),
    TowerType.MORTAR: (16, 4.0, 3),
    TowerType.HEAVY_PLUS: (35, 2.0, 3),
    TowerType.ICE: (15, 2.0, 2),
    TowerType.CANNON: (50, 3.0, 3),
    TowerType.QUICK_PLUS: (8, 0.5, 3),
    TowerType.DOUBLE: (7, 1.0, 4),
    TowerType.SNIPER: (15, 2.0, 6),
    TowerType.MORTAR_PLUS: (35, 4.0, 4),
    TowerType.PULSE: (30, 3.0, 2),
    TowerType.MISSILE: (45, 6.0, 5),
}

SUPER_WEAPON_INFO = {
    SuperWeaponType.LIGHTNING_STORM: (20, 3, 100, 150),
    SuperWeaponType.EMP_BLASTER: (20, 3, 100, 150),
    SuperWeaponType.DEFLECTOR: (10, 3, 50, 100),
    SuperWeaponType.EMERGENCY_EVASION: (1, 3, 50, 100),
}

PLAYER_BASES = ((2, EDGE - 1), ((MAP_SIZE - 1) - 2, EDGE - 1))
GENERATION_CYCLE = (4, 2, 1)
ANT_MAX_HP = (10, 25, 50)
ANT_REWARD = (3, 5, 7)
ANT_AGE_LIMIT = 32
PHEROMONE_BONUS_INT = {
    AntState.SUCCESS: 100000,
    AntState.FAIL: -50000,
    AntState.TOO_OLD: -30000,
}
ETA_SCALED = (12500, 10000, 7500)  # 1.25, 1.0, 0.75
ETA_OFFSET = 1


def _as_int(value) -> int:
    if isinstance(value, bytes):
        return value[0]
    if isinstance(value, str):
        return ord(value)
    return int(value)


def _arg_value(value) -> int:
    number = _as_int(value)
    if number == 255:
        return -1
    return number


def hex_distance(x0: int, y0: int, x1: int, y1: int) -> int:
    dy = abs(y0 - y1)
    if dy % 2:
        if x0 > x1:
            dx = max(0, abs(x0 - x1) - dy // 2 - (y0 % 2))
        else:
            dx = max(0, abs(x0 - x1) - dy // 2 - (1 - (y0 % 2)))
    else:
        dx = max(0, abs(x0 - x1) - dy // 2)
    return dx + dy


def is_valid_pos(x: int, y: int) -> bool:
    return 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and MAP_PROPERTY[x][y] >= 0


def is_path(x: int, y: int) -> bool:
    return 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and MAP_PROPERTY[x][y] == 0


def is_highland(player: int, x: int, y: int) -> bool:
    if not is_valid_pos(x, y):
        return False
    return MAP_PROPERTY[x][y] == (2 if player == 0 else 3)


def get_direction(x0: int, y0: int, x1: int, y1: int) -> int:
    dx = x1 - x0
    dy = y1 - y0
    for index, (ox, oy) in enumerate(OFFSET[y0 % 2]):
        if ox == dx and oy == dy:
            return index
    return -1


@dataclass(slots=True)
class Operation:
    op_type: OperationType
    arg0: int = -1
    arg1: int = -1

    def to_protocol_tokens(self) -> list[int]:
        op_type = int(self.op_type)
        if self.arg0 == -1:
            return [op_type]
        if self.arg1 == -1:
            return [op_type, self.arg0]
        return [op_type, self.arg0, self.arg1]


@dataclass(slots=True)
class Ant:
    id: int
    player: int
    x: int
    y: int
    hp: int
    level: int
    age: int
    state: AntState
    evasion: int = 0
    deflector: bool = False
    path: list[int] = field(default_factory=list)

    def clone(self) -> Ant:
        return Ant(
            id=self.id,
            player=self.player,
            x=self.x,
            y=self.y,
            hp=self.hp,
            level=self.level,
            age=self.age,
            state=self.state,
            evasion=self.evasion,
            deflector=self.deflector,
            path=list(self.path),
        )

    def move(self, direction: int) -> None:
        self.path.append(direction)
        self.x += OFFSET[self.y % 2][direction][0]
        self.y += OFFSET[self.y % 2][direction][1]

    def max_hp(self) -> int:
        return ANT_MAX_HP[self.level]

    def reward(self) -> int:
        return ANT_REWARD[self.level]

    def is_alive(self) -> bool:
        return self.state in (AntState.ALIVE, AntState.FROZEN)

    def is_in_range(self, x: int, y: int, attack_range: int) -> bool:
        return hex_distance(self.x, self.y, x, y) <= attack_range

    def is_attackable_from(self, player: int, x: int, y: int, attack_range: int) -> bool:
        return self.player != player and self.is_alive() and self.is_in_range(x, y, attack_range)


@dataclass(slots=True)
class Tower:
    id: int
    player: int
    x: int
    y: int
    type: TowerType = TowerType.BASIC
    cd: int = -2
    emp: bool = False
    damage: int = 0
    attack_range: int = 0
    speed: float = 0.0

    def __post_init__(self) -> None:
        initial_cd = self.cd
        self.type = TowerType(self.type)
        self.upgrade(self.type)
        if initial_cd != -2:
            self.cd = int(initial_cd)

    @property
    def range(self) -> int:
        return self.attack_range

    @range.setter
    def range(self, value: int) -> None:
        self.attack_range = value

    def clone(self) -> Tower:
        tower = Tower(self.id, self.player, self.x, self.y, self.type, self.cd)
        tower.emp = self.emp
        tower.damage = self.damage
        tower.attack_range = self.attack_range
        tower.speed = self.speed
        return tower

    def attack(self, ants: list[Ant]) -> list[int]:
        attacked_idxs: list[int] = []
        if self.cd > 0:
            self.cd -= 1
        if self.cd <= 0:
            repeat = 1 if self.speed >= 1 else int(1 / self.speed)
            target_num = 2 if self.type == TowerType.DOUBLE else 1
            while repeat > 0:
                repeat -= 1
                target_idxs = self.find_targets(ants, target_num)
                attackable_idxs = self.find_attackable(ants, target_idxs)
                for idx in attackable_idxs:
                    self.action(ants[idx])
                attacked_idxs.extend(attackable_idxs)
            attacked_idxs = sorted(set(attacked_idxs))
            if attacked_idxs:
                self.reset_cd()
        return attacked_idxs

    def find_targets(self, ants: list[Ant], target_num: int) -> list[int]:
        idxs = self.get_attackable_ants(ants, self.x, self.y, self.attack_range)
        idxs.sort(key=lambda idx: (hex_distance(ants[idx].x, ants[idx].y, self.x, self.y), idx))
        return idxs[:target_num]

    def find_attackable(self, ants: list[Ant], target_idxs: list[int]) -> list[int]:
        attackable_idxs: list[int] = []
        for idx in target_idxs:
            if self.type in (TowerType.MORTAR, TowerType.MORTAR_PLUS):
                found = self.get_attackable_ants(ants, ants[idx].x, ants[idx].y, 1)
            elif self.type == TowerType.PULSE:
                found = self.get_attackable_ants(ants, self.x, self.y, self.attack_range)
            elif self.type == TowerType.MISSILE:
                found = self.get_attackable_ants(ants, ants[idx].x, ants[idx].y, 2)
            else:
                found = [idx]
            attackable_idxs.extend(found)
        return attackable_idxs

    def action(self, ant: Ant) -> None:
        if ant.evasion > 0:
            ant.evasion -= 1
            return
        if ant.deflector and self.damage < ant.max_hp() / 2:
            return
        ant.hp -= self.damage
        if self.type == TowerType.ICE and ant.hp > 0:
            ant.state = AntState.FROZEN
        if ant.hp <= 0:
            ant.state = AntState.FAIL

    def get_attackable_ants(self, ants: list[Ant], x: int, y: int, attack_range: int) -> list[int]:
        return [idx for idx, ant in enumerate(ants) if ant.is_attackable_from(self.player, x, y, attack_range)]

    def is_ready(self) -> bool:
        return self.cd <= 0

    def reset_cd(self) -> None:
        self.cd = int(self.speed if self.speed > 1 else 1)

    def upgrade(self, new_type: TowerType) -> None:
        new_type = TowerType(new_type)
        self.type = new_type
        damage, speed, attack_range = TOWER_INFO[new_type]
        self.damage = int(damage)
        self.speed = float(speed)
        self.attack_range = int(attack_range)
        self.reset_cd()

    def is_upgrade_type_valid(self, target: int | TowerType) -> bool:
        try:
            target_type = TowerType(target)
        except ValueError:
            return False
        if self.type == TowerType.BASIC:
            return target_type in (TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR)
        if self.type == TowerType.HEAVY:
            return target_type in (TowerType.HEAVY_PLUS, TowerType.ICE, TowerType.CANNON)
        if self.type == TowerType.QUICK:
            return target_type in (TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER)
        if self.type == TowerType.MORTAR:
            return target_type in (TowerType.MORTAR_PLUS, TowerType.PULSE, TowerType.MISSILE)
        return False

    def downgrade(self) -> None:
        self.upgrade(TowerType(int(self.type) // 10))

    def is_downgrade_valid(self) -> bool:
        return self.type != TowerType.BASIC


@dataclass(slots=True)
class Base:
    player: int
    hp: int = 50
    gen_speed_level: int = 0
    ant_level: int = 0
    x: int = field(init=False)
    y: int = field(init=False)

    def __post_init__(self) -> None:
        self.x, self.y = PLAYER_BASES[self.player]

    def clone(self) -> Base:
        base = Base(self.player, self.hp, self.gen_speed_level, self.ant_level)
        return base

    def generate_ant(self, ant_id: int, round_index: int) -> Ant | None:
        if round_index % GENERATION_CYCLE[self.gen_speed_level] != 0:
            return None
        return Ant(ant_id, self.player, self.x, self.y, ANT_MAX_HP[self.ant_level], self.ant_level, 0, AntState.ALIVE)

    def upgrade_generation_speed(self) -> None:
        self.gen_speed_level += 1

    def upgrade_generated_ant(self) -> None:
        self.ant_level += 1


@dataclass(slots=True)
class SuperWeapon:
    type: SuperWeaponType
    player: int
    x: int
    y: int
    left_time: int = field(init=False)
    range: int = field(init=False)

    def __post_init__(self) -> None:
        self.type = SuperWeaponType(self.type)
        duration, attack_range, _, _ = SUPER_WEAPON_INFO[self.type]
        self.left_time = int(duration + 1)
        self.range = int(attack_range)

    def clone(self) -> SuperWeapon:
        weapon = SuperWeapon(self.type, self.player, self.x, self.y)
        weapon.left_time = self.left_time
        weapon.range = self.range
        return weapon

    def is_in_range(self, x: int, y: int) -> bool:
        return hex_distance(self.x, self.y, x, y) <= self.range


class RandomSource:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def get(self) -> int:
        self.seed = (25214903917 * self.seed) & ((1 << 48) - 1)
        return self.seed


@dataclass(slots=True)
class GameInfo:
    round: int = 0
    towers: list[Tower] = field(default_factory=list)
    ants: list[Ant] = field(default_factory=list)
    bases: list[Base] = field(default_factory=lambda: [Base(0), Base(1)])
    coins: list[int] = field(default_factory=lambda: [COIN_INIT, COIN_INIT])
    pheromone: list[list[list[int]]] = field(
        default_factory=lambda: [[[PHEROMONE_INIT_INT for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)] for _ in range(2)]
    )
    building_tag: list[list[BuildingType]] = field(
        default_factory=lambda: [[BuildingType.EMPTY for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
    )
    super_weapons: list[SuperWeapon] = field(default_factory=list)
    super_weapon_cd: list[list[int]] = field(default_factory=lambda: [[0] * int(SuperWeaponType.COUNT) for _ in range(2)])
    old_count: list[int] = field(default_factory=lambda: [0, 0])
    die_count: list[int] = field(default_factory=lambda: [0, 0])
    next_ant_id: int = 0
    next_tower_id: int = 0

    def __init__(self, seed: int = 0) -> None:
        self.round = 0
        self.towers = []
        self.ants = []
        self.bases = [Base(0), Base(1)]
        self.coins = [COIN_INIT, COIN_INIT]
        rng = RandomSource(seed)
        self.pheromone = [
            [
                [PHEROMONE_INIT_INT + (rng.get() * 10000 >> 46) for _ in range(MAP_SIZE)]
                for _ in range(MAP_SIZE)
            ]
            for _ in range(2)
        ]
        self.building_tag = [[BuildingType.EMPTY for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
        for player, (x, y) in enumerate(PLAYER_BASES):
            self.building_tag[x][y] = BuildingType.BASE
        self.super_weapons = []
        self.super_weapon_cd = [[0] * int(SuperWeaponType.COUNT) for _ in range(2)]
        self.old_count = [0, 0]
        self.die_count = [0, 0]
        self.next_ant_id = 0
        self.next_tower_id = 0

    def clone(self) -> GameInfo:
        copy = GameInfo(0)
        copy.round = self.round
        copy.towers = [tower.clone() for tower in self.towers]
        copy.ants = [ant.clone() for ant in self.ants]
        copy.bases = [base.clone() for base in self.bases]
        copy.coins = list(self.coins)
        copy.pheromone = [[list(row) for row in plane] for plane in self.pheromone]
        copy.building_tag = [[cell for cell in row] for row in self.building_tag]
        copy.super_weapons = [weapon.clone() for weapon in self.super_weapons]
        copy.super_weapon_cd = [list(row) for row in self.super_weapon_cd]
        copy.old_count = list(self.old_count)
        copy.die_count = list(self.die_count)
        copy.next_ant_id = self.next_ant_id
        copy.next_tower_id = self.next_tower_id
        return copy

    def tower_at(self, x: int, y: int) -> Tower | None:
        for tower in self.towers:
            if tower.x == x and tower.y == y:
                return tower
        return None

    def tower_of_id(self, tower_id: int) -> Tower | None:
        for tower in self.towers:
            if tower.id == tower_id:
                return tower
        return None

    def tower_num_of_player(self, player_id: int) -> int:
        return sum(1 for tower in self.towers if tower.player == player_id)

    def build_tower(self, tower_id: int, player: int, x: int, y: int, tower_type: TowerType = TowerType.BASIC) -> None:
        self.towers.append(Tower(tower_id, player, x, y, tower_type))
        self.building_tag[x][y] = BuildingType.TOWER

    def upgrade_tower(self, tower_id: int, tower_type: TowerType) -> None:
        tower = self.tower_of_id(tower_id)
        if tower is not None:
            tower.upgrade(tower_type)

    def downgrade_or_destroy_tower(self, tower_id: int) -> None:
        for index, tower in enumerate(self.towers):
            if tower.id != tower_id:
                continue
            if tower.is_downgrade_valid():
                tower.downgrade()
            else:
                self.building_tag[tower.x][tower.y] = BuildingType.EMPTY
                self.towers.pop(index)
            return

    def upgrade_generation_speed(self, player_id: int) -> None:
        self.bases[player_id].upgrade_generation_speed()

    def upgrade_generated_ant(self, player_id: int) -> None:
        self.bases[player_id].upgrade_generated_ant()

    def set_coin(self, player_id: int, value: int) -> None:
        self.coins[player_id] = int(value)

    def update_coin(self, player_id: int, change: int) -> None:
        self.coins[player_id] += int(change)

    def set_base_hp(self, player_id: int, value: int) -> None:
        self.bases[player_id].hp = int(value)

    def update_base_hp(self, player_id: int, change: int) -> None:
        self.bases[player_id].hp += int(change)

    def clear_dead_and_succeeded_ants(self) -> None:
        alive: list[Ant] = []
        for ant in self.ants:
            if ant.state == AntState.FAIL:
                self.die_count[ant.player] += 1
            elif ant.state == AntState.TOO_OLD:
                self.old_count[ant.player] += 1
            if ant.state not in (AntState.SUCCESS, AntState.FAIL, AntState.TOO_OLD):
                alive.append(ant)
        self.ants = alive

    def update_pheromone_for_ants(self) -> None:
        for ant in self.ants:
            self.update_pheromone(ant)

    def update_pheromone(self, ant: Ant) -> None:
        if ant.state in (AntState.ALIVE, AntState.FROZEN):
            return
        tau = PHEROMONE_BONUS_INT.get(ant.state)
        if tau is None:
            return
        player = ant.player
        x, y = PLAYER_BASES[player]
        visited = [[False for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
        enemy_base = PLAYER_BASES[player ^ 1]
        for move in ant.path:
            if not 0 <= move < len(OFFSET[y % 2]):
                return
            if not visited[x][y]:
                visited[x][y] = True
                self.pheromone[player][x][y] += tau
                if self.pheromone[player][x][y] < PHEROMONE_MIN:
                    self.pheromone[player][x][y] = PHEROMONE_MIN
            next_x = x + OFFSET[y % 2][move][0]
            next_y = y + OFFSET[y % 2][move][1]
            if not is_valid_pos(next_x, next_y):
                return
            if (next_x, next_y) != enemy_base and not is_path(next_x, next_y):
                return
            x = next_x
            y = next_y
        if not visited[x][y]:
            self.pheromone[player][x][y] = max(PHEROMONE_MIN, self.pheromone[player][x][y] + tau)

    @staticmethod
    def sanitize_ant_path(ant: Ant) -> None:
        x, y = PLAYER_BASES[ant.player]
        enemy_base = PLAYER_BASES[ant.player ^ 1]
        for move in ant.path:
            if not 0 <= move < len(OFFSET[y % 2]):
                ant.path.clear()
                return
            x += OFFSET[y % 2][move][0]
            y += OFFSET[y % 2][move][1]
            if not is_valid_pos(x, y):
                ant.path.clear()
                return
            if (x, y) != enemy_base and not is_path(x, y):
                ant.path.clear()
                return
        if (x, y) != (ant.x, ant.y):
            ant.path.clear()

    @staticmethod
    def sanitize_ant_path(ant: Ant) -> None:
        x, y = PLAYER_BASES[ant.player]
        enemy_base = PLAYER_BASES[ant.player ^ 1]
        for move in ant.path:
            if not 0 <= move < len(OFFSET[y % 2]):
                ant.path.clear()
                return
            x += OFFSET[y % 2][move][0]
            y += OFFSET[y % 2][move][1]
            if not is_valid_pos(x, y):
                ant.path.clear()
                return
            if (x, y) != enemy_base and not is_path(x, y):
                ant.path.clear()
                return
        if (x, y) != (ant.x, ant.y):
            ant.path.clear()

    def global_pheromone_attenuation(self) -> None:
        for player in range(2):
            for x in range(MAP_SIZE):
                for y in range(MAP_SIZE):
                    if MAP_PROPERTY[x][y] >= 0:
                        p = self.pheromone[player][x][y]
                        self.pheromone[player][x][y] = max(
                            PHEROMONE_MIN,
                            (LAMBDA_NUM * p + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM,
                        )

    def is_operation_valid(self, player_id: int, ops_or_op, new_op: Operation | None = None) -> bool:
        if new_op is None:
            op = coerce_operation(ops_or_op)
            if op.op_type == OperationType.BUILD_TOWER:
                return is_valid_pos(op.arg0, op.arg1) and is_highland(player_id, op.arg0, op.arg1) and not self.is_shielded_by_emp(player_id, op.arg0, op.arg1)
            if op.op_type == OperationType.UPGRADE_TOWER:
                tower = self.tower_of_id(op.arg0)
                return tower is not None and tower.player == player_id and tower.is_upgrade_type_valid(op.arg1) and not self.is_shielded_by_emp(tower)
            if op.op_type == OperationType.DOWNGRADE_TOWER:
                tower = self.tower_of_id(op.arg0)
                return tower is not None and tower.player == player_id and not self.is_shielded_by_emp(tower)
            if op.op_type in (
                OperationType.USE_LIGHTNING_STORM,
                OperationType.USE_EMP_BLASTER,
                OperationType.USE_DEFLECTOR,
                OperationType.USE_EMERGENCY_EVASION,
            ):
                weapon_index = int(op.op_type) % 10
                return is_valid_pos(op.arg0, op.arg1) and self.super_weapon_cd[player_id][weapon_index] <= 0
            if op.op_type == OperationType.UPGRADE_GENERATION_SPEED:
                return self.bases[player_id].gen_speed_level < 2
            if op.op_type == OperationType.UPGRADE_GENERATED_ANT:
                return self.bases[player_id].ant_level < 2
            return False

        ops = [coerce_operation(op) for op in ops_or_op]
        candidate = coerce_operation(new_op)
        if candidate.op_type == OperationType.BUILD_TOWER:
            collide = any(op.op_type == OperationType.BUILD_TOWER and op.arg0 == candidate.arg0 and op.arg1 == candidate.arg1 for op in ops)
        elif candidate.op_type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER):
            collide = any(op.op_type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER) and op.arg0 == candidate.arg0 for op in ops)
        elif candidate.op_type in (OperationType.UPGRADE_GENERATED_ANT, OperationType.UPGRADE_GENERATION_SPEED):
            collide = any(op.op_type in (OperationType.UPGRADE_GENERATED_ANT, OperationType.UPGRADE_GENERATION_SPEED) for op in ops)
        elif candidate.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            collide = any(op.op_type == candidate.op_type for op in ops)
        else:
            return False
        if collide or not self.is_operation_valid(player_id, candidate):
            return False
        return self.check_affordable(player_id, ops + [candidate])

    def get_operation_income(self, player_id: int, op: Operation) -> int:
        if op.op_type == OperationType.BUILD_TOWER:
            return -self.build_tower_cost(self.tower_num_of_player(player_id))
        if op.op_type == OperationType.UPGRADE_TOWER:
            return -self.upgrade_tower_cost(op.arg1)
        if op.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_of_id(op.arg0)
            if tower is None:
                return 0
            if tower.type == TowerType.BASIC:
                return self.destroy_tower_income(self.tower_num_of_player(player_id))
            return self.downgrade_tower_income(tower.type)
        if op.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            return -self.use_super_weapon_cost(int(op.op_type) % 10)
        if op.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            return -self.upgrade_base_cost(self.bases[player_id].gen_speed_level)
        if op.op_type == OperationType.UPGRADE_GENERATED_ANT:
            return -self.upgrade_base_cost(self.bases[player_id].ant_level)
        return 0

    def check_affordable(self, player_id: int, ops: Iterable[Operation]) -> bool:
        income = 0
        tower_num = self.tower_num_of_player(player_id)
        for op in ops:
            op = coerce_operation(op)
            if op.op_type == OperationType.BUILD_TOWER:
                income -= self.build_tower_cost(tower_num)
                tower_num += 1
            elif op.op_type == OperationType.DOWNGRADE_TOWER:
                tower = self.tower_of_id(op.arg0)
                if tower is None:
                    continue
                if tower.type == TowerType.BASIC:
                    income += self.destroy_tower_income(tower_num)
                    tower_num -= 1
                else:
                    income += self.downgrade_tower_income(tower.type)
            else:
                income += self.get_operation_income(player_id, op)
        return income + self.coins[player_id] >= 0

    def is_current_and_around_empty(self, x: int, y: int) -> bool:
        if self.building_tag[x][y] != BuildingType.EMPTY:
            return False
        for ox, oy in OFFSET[y % 2]:
            nx = x + ox
            ny = y + oy
            if is_valid_pos(nx, ny) and self.building_tag[nx][ny] != BuildingType.EMPTY:
                return False
        return True

    def apply_operation(self, player_id: int, op) -> None:
        operation = coerce_operation(op)
        self.update_coin(player_id, self.get_operation_income(player_id, operation))
        if operation.op_type == OperationType.BUILD_TOWER:
            self.build_tower(self.next_tower_id, player_id, operation.arg0, operation.arg1)
            self.next_tower_id += 1
        elif operation.op_type == OperationType.UPGRADE_TOWER:
            self.upgrade_tower(operation.arg0, TowerType(operation.arg1))
        elif operation.op_type == OperationType.DOWNGRADE_TOWER:
            self.downgrade_or_destroy_tower(operation.arg0)
        elif operation.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            self.use_super_weapon(SuperWeaponType(int(operation.op_type) % 10), player_id, operation.arg0, operation.arg1)
        elif operation.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            self.upgrade_generation_speed(player_id)
        elif operation.op_type == OperationType.UPGRADE_GENERATED_ANT:
            self.upgrade_generated_ant(player_id)

    def next_move(self, ant: Ant) -> int:
        target_x, target_y = PLAYER_BASES[ant.player ^ 1]
        current_distance = hex_distance(ant.x, ant.y, target_x, target_y)
        best_index = 0
        best_weighted = -1
        best_raw = -1
        for index, (ox, oy) in enumerate(OFFSET[ant.y % 2]):
            nx = ant.x + ox
            ny = ant.y + oy
            if (ant.path and ant.path[-1] == (index + 3) % 6) or not is_path(nx, ny):
                continue
            next_distance = hex_distance(nx, ny, target_x, target_y)
            eta = ETA_SCALED[next_distance - current_distance + ETA_OFFSET]
            raw = self.pheromone[ant.player][nx][ny]
            weighted = raw * eta // PHEROMONE_SCALE
            if (weighted, raw, -index) > (best_weighted, best_raw, -best_index):
                best_index = index
                best_weighted = weighted
                best_raw = raw
        return best_index

    @staticmethod
    def destroy_tower_income(tower_num: int) -> int:
        return int(GameInfo.build_tower_cost(tower_num - 1) * TOWER_DOWNGRADE_REFUND_RATIO)

    @staticmethod
    def downgrade_tower_income(tower_type: int | TowerType) -> int:
        return int(GameInfo.upgrade_tower_cost(int(tower_type)) * TOWER_DOWNGRADE_REFUND_RATIO)

    @staticmethod
    def build_tower_cost(tower_num: int) -> int:
        return int(TOWER_BUILD_PRICE_BASE * (TOWER_BUILD_PRICE_RATIO ** tower_num))

    @staticmethod
    def upgrade_tower_cost(tower_type: int | TowerType) -> int:
        tower_value = int(tower_type)
        if tower_value in (int(TowerType.HEAVY), int(TowerType.QUICK), int(TowerType.MORTAR)):
            return LEVEL2_TOWER_UPGRADE_PRICE
        if tower_value in {
            int(TowerType.HEAVY_PLUS),
            int(TowerType.ICE),
            int(TowerType.CANNON),
            int(TowerType.QUICK_PLUS),
            int(TowerType.DOUBLE),
            int(TowerType.SNIPER),
            int(TowerType.MORTAR_PLUS),
            int(TowerType.PULSE),
            int(TowerType.MISSILE),
        }:
            return LEVEL3_TOWER_UPGRADE_PRICE
        return -1

    @staticmethod
    def upgrade_base_cost(level: int) -> int:
        if level == 0:
            return LEVEL2_BASE_UPGRADE_PRICE
        if level == 1:
            return LEVEL3_BASE_UPGRADE_PRICE
        return -1

    @staticmethod
    def use_super_weapon_cost(weapon_type: int | SuperWeaponType) -> int:
        return SUPER_WEAPON_INFO[SuperWeaponType(int(weapon_type))][3]

    def use_super_weapon(self, weapon_type: SuperWeaponType, player: int, x: int, y: int) -> None:
        weapon = SuperWeapon(weapon_type, player, x, y)
        if weapon.type == SuperWeaponType.EMERGENCY_EVASION:
            for ant in self.ants:
                if ant.player == weapon.player and weapon.is_in_range(ant.x, ant.y):
                    ant.evasion = 2
        else:
            self.super_weapons.append(weapon)
        self.super_weapon_cd[player][int(weapon.type)] = SUPER_WEAPON_INFO[weapon.type][2]

    def is_shielded_by_emp(self, target, x: int | None = None, y: int | None = None) -> bool:
        if x is None and y is None and hasattr(target, 'x') and hasattr(target, 'y'):
            player_id = _as_int(target.player)
            x = _as_int(target.x)
            y = _as_int(target.y)
        else:
            player_id = int(target)
            assert x is not None and y is not None
        return any(
            weapon.type == SuperWeaponType.EMP_BLASTER and weapon.player != player_id and weapon.is_in_range(x, y)
            for weapon in self.super_weapons
        )

    def is_shielded_by_deflector(self, ant: Ant) -> bool:
        return any(
            weapon.type == SuperWeaponType.DEFLECTOR and weapon.player == ant.player and weapon.is_in_range(ant.x, ant.y)
            for weapon in self.super_weapons
        )

    def count_down_super_weapons_left_time(self, player_id: int) -> None:
        survivors: list[SuperWeapon] = []
        for weapon in self.super_weapons:
            if weapon.player == player_id:
                weapon.left_time -= 1
            if weapon.left_time > 0:
                survivors.append(weapon)
        self.super_weapons = survivors

    def count_down_super_weapons_cd(self) -> None:
        for player in range(2):
            for weapon in range(1, 5):
                self.super_weapon_cd[player][weapon] = max(self.super_weapon_cd[player][weapon] - 1, 0)


@dataclass(slots=True)
class Simulator:
    info: GameInfo
    operations: list[list[Operation]] = field(default_factory=lambda: [[], []])

    def clone(self) -> Simulator:
        return Simulator(self.info.clone(), [[coerce_operation(op) for op in side] for side in self.operations])

    def add_operation_of_player(self, player_id: int, op) -> bool:
        operation = coerce_operation(op)
        if self.info.is_operation_valid(player_id, self.operations[player_id], operation):
            self.operations[player_id].append(operation)
            return True
        return False

    def apply_operations_of_player(self, player_id: int) -> None:
        for operation in self.operations[player_id]:
            self.info.apply_operation(player_id, operation)

    def fast_next_round(self, side: int) -> bool:
        if self.info.round >= MAX_ROUND:
            return False

        updated_weapons: list[SuperWeapon] = []
        for weapon in self.info.super_weapons:
            weapon.left_time -= 1
            if weapon.left_time > 0:
                updated_weapons.append(weapon)
        self.info.super_weapons = updated_weapons

        self.info.ants = [ant for ant in self.info.ants if ant.player != side]
        self.info.towers = [tower for tower in self.info.towers if tower.player == side]

        for ant in self.info.ants:
            ant.deflector = False
        for tower in self.info.towers:
            tower.emp = False
        for weapon in self.info.super_weapons:
            if weapon.type == SuperWeaponType.LIGHTNING_STORM and weapon.player == side:
                for ant in self.info.ants:
                    if weapon.is_in_range(ant.x, ant.y):
                        ant.hp = 0
                        ant.state = AntState.FAIL
                        self.info.coins[weapon.player] += ant.reward()
            elif weapon.type == SuperWeaponType.DEFLECTOR and weapon.player == (side ^ 1):
                for ant in self.info.ants:
                    if weapon.is_in_range(ant.x, ant.y):
                        ant.deflector = True
            elif weapon.type == SuperWeaponType.EMP_BLASTER and weapon.player == (side ^ 1):
                for tower in self.info.towers:
                    if weapon.is_in_range(tower.x, tower.y):
                        tower.emp = True

        for tower in self.info.towers:
            if tower.emp:
                continue
            targets = tower.attack(self.info.ants)
            for idx in targets:
                if self.info.ants[idx].state == AntState.FAIL:
                    self.info.coins[tower.player] += self.info.ants[idx].reward()

        for ant in self.info.ants:
            ant.age += 1
            if ant.state == AntState.FAIL:
                continue
            if ant.age > ANT_AGE_LIMIT:
                ant.state = AntState.TOO_OLD
            if ant.state == AntState.ALIVE:
                ant.move(self.info.next_move(ant))
            enemy_base_x, enemy_base_y = PLAYER_BASES[ant.player ^ 1]
            if ant.x == enemy_base_x and ant.y == enemy_base_y:
                ant.state = AntState.SUCCESS
                self.info.bases[ant.player ^ 1].hp -= 1
                self.info.coins[ant.player] += 5
                if self.info.bases[ant.player ^ 1].hp <= 0:
                    return False
            if ant.state == AntState.FROZEN:
                ant.state = AntState.ALIVE

        enemy = side ^ 1
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                if MAP_PROPERTY[x][y] >= 0:
                    p = self.info.pheromone[enemy][x][y]
                    self.info.pheromone[enemy][x][y] = max(
                        PHEROMONE_MIN,
                        (LAMBDA_NUM * p + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM,
                    )
        for ant in self.info.ants:
            self.info.update_pheromone(ant)
        self.info.clear_dead_and_succeeded_ants()

        base = self.info.bases[enemy]
        if self.info.round % GENERATION_CYCLE[base.gen_speed_level] == 0:
            self.info.ants.append(Ant(self.info.next_ant_id, enemy, base.x, base.y, ANT_MAX_HP[base.ant_level], base.ant_level, 0, AntState.ALIVE))
            self.info.next_ant_id += 1

        self.info.coins[0] += 1
        self.info.coins[1] += 1
        if self.info.round % 3 != 0:
            self.info.coins[enemy] += 1

        self.info.round += 1
        for player in range(2):
            for weapon in range(1, 5):
                if self.info.super_weapon_cd[player][weapon] > 0:
                    self.info.super_weapon_cd[player][weapon] -= 1
        self.operations[side].clear()
        return True


@dataclass(slots=True)
class GreedyController:
    self_player_id: int
    info: GameInfo

    @classmethod
    def start(cls, self_player_id: int, seed: int) -> GreedyController:
        return cls(self_player_id=self_player_id, info=GameInfo(seed))

    def apply_self_operations(self, operations: Iterable[Operation]) -> None:
        player = self.self_player_id
        self.info.count_down_super_weapons_left_time(player)
        for operation in operations:
            self.info.apply_operation(player, operation)

    def apply_opponent_operations(self, operations: Iterable[Operation]) -> None:
        player = self.self_player_id ^ 1
        self.info.count_down_super_weapons_left_time(player)
        for operation in operations:
            self.info.apply_operation(player, operation)

    def sync_public_round_state(self, public_round_state) -> None:
        for weapon in self.info.super_weapons:
            if weapon.type != SuperWeaponType.LIGHTNING_STORM:
                continue
            for ant in self.info.ants:
                if weapon.player != ant.player and weapon.is_in_range(ant.x, ant.y):
                    ant.hp = 0
                    ant.state = AntState.FAIL
                    self.info.update_coin(weapon.player, ant.reward())

        for ant in self.info.ants:
            ant.deflector = self.info.is_shielded_by_deflector(ant)
        for tower in self.info.towers:
            if self.info.is_shielded_by_emp(tower):
                continue
            targets = tower.attack(self.info.ants)
            for idx in targets:
                if self.info.ants[idx].state == AntState.FAIL:
                    self.info.update_coin(tower.player, self.info.ants[idx].reward())
        for ant in self.info.ants:
            ant.deflector = False

        self._update_towers(public_round_state.towers)
        self._update_ants(public_round_state.ants)
        self.info.global_pheromone_attenuation()
        self.info.update_pheromone_for_ants()
        self.info.clear_dead_and_succeeded_ants()
        self.info.set_coin(0, public_round_state.coins[0])
        self.info.set_coin(1, public_round_state.coins[1])
        self.info.set_base_hp(0, public_round_state.camps_hp[0])
        self.info.set_base_hp(1, public_round_state.camps_hp[1])
        self.info.round = int(public_round_state.round_index)
        self.info.count_down_super_weapons_cd()

    def _update_towers(self, tower_rows) -> None:
        for tower in self.info.towers:
            self.info.building_tag[tower.x][tower.y] = BuildingType.EMPTY
        self.info.towers = []
        for row in tower_rows:
            tower_id, player, x, y, tower_type, cooldown = row
            tower = Tower(int(tower_id), int(player), int(x), int(y), TowerType(int(tower_type)), int(cooldown))
            self.info.towers.append(tower)
            self.info.building_tag[tower.x][tower.y] = BuildingType.TOWER
        self.info.next_tower_id = max((tower.id for tower in self.info.towers), default=-1) + 1

    def _update_ants(self, ant_rows) -> None:
        for row in ant_rows:
            self._update_ant(Ant(int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), AntState(int(row[7]))))
        self.info.next_ant_id = max((ant.id for ant in self.info.ants), default=-1) + 1

    def _update_ant(self, ant: Ant) -> None:
        for known in self.info.ants:
            if known.id != ant.id:
                continue
            if (known.x, known.y) != (ant.x, ant.y):
                move = get_direction(known.x, known.y, ant.x, ant.y)
                if move >= 0:
                    known.path.append(move)
                else:
                    known.path.clear()
            known.x = ant.x
            known.y = ant.y
            known.hp = ant.hp
            known.level = ant.level
            known.age = ant.age
            known.state = ant.state
            known.evasion = 0
            known.deflector = False
            self.info.sanitize_ant_path(known)
            return
        self.info.sanitize_ant_path(ant)
        self.info.ants.append(ant)


def coerce_operation(operation) -> Operation:
    if isinstance(operation, Operation):
        return operation
    op_type = getattr(operation, 'op_type', None)
    if op_type is None:
        op_type = getattr(operation, 'type')
    return Operation(OperationType(_as_int(op_type)), _arg_value(getattr(operation, 'arg0', -1)), _arg_value(getattr(operation, 'arg1', -1)))


def info_from_state(state, player: int = 0, seed: int = 0) -> GameInfo:
    if hasattr(state, 'native'):
        state = state
    info = GameInfo(seed)
    if hasattr(state, 'round_index'):
        info.round = int(state.round_index)
    elif hasattr(state, 'info') and hasattr(state.info, 'round'):
        info.round = _as_int(state.info.round)

    info.coins = [int(state.coins[0]), int(state.coins[1])] if hasattr(state, 'coins') else list(info.coins)
    if hasattr(state, 'old_count'):
        info.old_count = [int(state.old_count[0]), int(state.old_count[1])]
    if hasattr(state, 'die_count'):
        info.die_count = [int(state.die_count[0]), int(state.die_count[1])]

    if hasattr(state, 'bases'):
        info.bases = []
        for base in state.bases:
            built = Base(int(getattr(base, 'player', 0)), int(getattr(base, 'hp')), int(getattr(base, 'gen_speed_level', getattr(base, 'generation_level', 0))), int(getattr(base, 'ant_level', 0)))
            info.bases.append(built)
    if len(info.bases) < 2:
        info.bases = [Base(0), Base(1)]

    info.towers = []
    if hasattr(state, 'towers'):
        for tower in state.towers:
            tower_type = getattr(tower, 'type', getattr(tower, 'tower_type'))
            cooldown = getattr(tower, 'cd', None)
            if cooldown is None:
                display = getattr(tower, 'display_cooldown', None)
                cooldown = display() if callable(display) else getattr(tower, 'cooldown_clock', 0)
            info.towers.append(Tower(int(getattr(tower, 'id', getattr(tower, 'tower_id'))), int(tower.player), int(tower.x), int(tower.y), TowerType(int(tower_type)), int(cooldown)))
    info.building_tag = [[BuildingType.EMPTY for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
    for base in info.bases:
        info.building_tag[base.x][base.y] = BuildingType.BASE
    for tower in info.towers:
        info.building_tag[tower.x][tower.y] = BuildingType.TOWER

    info.ants = []
    if hasattr(state, 'ants'):
        for ant in state.ants:
            status = getattr(ant, 'state', getattr(ant, 'status'))
            path = list(getattr(ant, 'path', []))
            info.ants.append(
                Ant(
                    int(getattr(ant, 'id', getattr(ant, 'ant_id'))),
                    int(ant.player),
                    int(ant.x),
                    int(ant.y),
                    int(ant.hp),
                    int(ant.level),
                    int(ant.age),
                    AntState(int(status)),
                    int(getattr(ant, 'evasion', getattr(ant, 'shield', 0))),
                    bool(getattr(ant, 'deflector', False)),
                    path,
                )
            )
            info.sanitize_ant_path(info.ants[-1])

    if hasattr(state, 'active_effects'):
        info.super_weapons = []
        for effect in state.active_effects:
            weapon_type = getattr(effect, 'type', getattr(effect, 'weapon_type'))
            weapon = SuperWeapon(SuperWeaponType(int(weapon_type)), int(effect.player), int(effect.x), int(effect.y))
            weapon.left_time = int(getattr(effect, 'left_time', getattr(effect, 'remaining_turns', weapon.left_time)))
            info.super_weapons.append(weapon)

    if hasattr(state, 'weapon_cooldowns'):
        info.super_weapon_cd = [[int(value) for value in row] for row in state.weapon_cooldowns]
    elif hasattr(state, 'info') and hasattr(state.info, 'super_weapon_cd'):
        info.super_weapon_cd = [[int(value) for value in row] for row in state.info.super_weapon_cd]

    if hasattr(state, 'pheromone'):
        info.pheromone = [
            [[int(state.pheromone[player_index, x, y]) for y in range(MAP_SIZE)] for x in range(MAP_SIZE)]
            for player_index in range(2)
        ]

    if hasattr(state, 'next_ant_id'):
        info.next_ant_id = int(state.next_ant_id)
    else:
        info.next_ant_id = max((ant.id for ant in info.ants), default=-1) + 1
    if hasattr(state, 'next_tower_id'):
        info.next_tower_id = int(state.next_tower_id)
    else:
        info.next_tower_id = max((tower.id for tower in info.towers), default=-1) + 1

    if info.round > 0 and not hasattr(state, 'pheromone'):
        # Direct conversion from a public-only snapshot cannot reconstruct pheromone history exactly.
        # Keep the seeded field so decision making still runs, but protocol-driven usage should prefer GreedyController.
        pass
    return info
