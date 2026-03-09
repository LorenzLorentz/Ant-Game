from __future__ import annotations

import struct
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from math import pow
from typing import Deque, Iterable, List, Optional, Sequence, Tuple


MAX_ROUND = 512
EDGE = 10
MAP_SIZE = 2 * EDGE - 1

COIN_INIT = 50
BASIC_INCOME = 1
TOWER_BUILD_PRICE_BASE = 15
TOWER_BUILD_PRICE_RATIO = 2
LEVEL2_TOWER_UPGRADE_PRICE = 60
LEVEL3_TOWER_UPGRADE_PRICE = 200
TOWER_DOWNGRADE_REFUND_RATIO = 0.8
LEVEL2_BASE_UPGRADE_PRICE = 200
LEVEL3_BASE_UPGRADE_PRICE = 250

PHEROMONE_INIT = 10.0
PHEROMONE_MIN = 0.0
PHEROMONE_ATTENUATING_RATIO = 0.97

BASE_POS = ((2, EDGE - 1), ((MAP_SIZE - 1) - 2, EDGE - 1))
GENERATION_CYCLE = (4, 2, 1)
ANT_MAX_HP = (10, 25, 50)
ANT_REWARD = (3, 5, 7)

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


class BuildingType(IntEnum):
    EMPTY = 0
    TOWER = 1
    BASE = 2


class PointType(IntEnum):
    VOID = -1
    PATH = 0
    BARRIER = 1
    PLAYER0_HIGHLAND = 2
    PLAYER1_HIGHLAND = 3


class AntState(IntEnum):
    ALIVE = 0
    SUCCESS = 1
    FAIL = 2
    TOO_OLD = 3
    FROZEN = 4


class TowerType(IntEnum):
    BASIC = 0
    HEAVY = 1
    HEAVY_PLUS = 11
    ICE = 12
    CANNON = 13
    QUICK = 2
    QUICK_PLUS = 21
    DOUBLE = 22
    SNIPER = 23
    MORTAR = 3
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


def distance(x0: int, y0: int, x1: int, y1: int) -> int:
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
    return 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and MAP_PROPERTY[x][y] != PointType.VOID


def is_path(x: int, y: int) -> bool:
    return 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and MAP_PROPERTY[x][y] == PointType.PATH


def is_highland(player: int, x: int, y: int) -> bool:
    if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
        return False
    target = PointType.PLAYER0_HIGHLAND if player == 0 else PointType.PLAYER1_HIGHLAND
    return MAP_PROPERTY[x][y] == target


def get_direction(x0: int, y0: int, x1: int, y1: int) -> int:
    dx = x1 - x0
    dy = y1 - y0
    for idx, (off_x, off_y) in enumerate(OFFSET[y0 % 2]):
        if off_x == dx and off_y == dy:
            return idx
    return -1


class LcgRandom:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def get(self) -> int:
        self.seed = (25214903917 * self.seed) & ((1 << 48) - 1)
        return self.seed


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
    path: List[int] = field(default_factory=list)

    AGE_LIMIT = 32

    def move(self, direction: int) -> None:
        self.path.append(direction)
        off_x, off_y = OFFSET[self.y % 2][direction]
        self.x += off_x
        self.y += off_y

    def max_hp(self) -> int:
        return ANT_MAX_HP[self.level]

    def reward(self) -> int:
        return ANT_REWARD[self.level]

    def is_alive(self) -> bool:
        return self.state in (AntState.ALIVE, AntState.FROZEN)

    def in_range(self, x: int, y: int, radius: int) -> bool:
        return distance(self.x, self.y, x, y) <= radius

    def is_attackable_from(self, player: int, x: int, y: int, radius: int) -> bool:
        return self.player != player and self.is_alive() and self.in_range(x, y, radius)

    def clone(self) -> Ant:
        return Ant(
            self.id,
            self.player,
            self.x,
            self.y,
            self.hp,
            self.level,
            self.age,
            AntState(self.state),
            self.evasion,
            self.deflector,
            list(self.path),
        )


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
    range: int = 0
    speed: float = 0.0

    def __post_init__(self) -> None:
        seeded_cd = self.cd
        self.upgrade(self.type)
        if seeded_cd != -2:
            self.cd = seeded_cd

    def clone(self) -> Tower:
        copied = Tower(self.id, self.player, self.x, self.y, self.type, self.cd)
        copied.emp = self.emp
        copied.damage = self.damage
        copied.range = self.range
        copied.speed = self.speed
        return copied

    def get_attackable_ants(self, ants: Sequence[Ant], x: int, y: int, radius: int) -> List[int]:
        return [idx for idx, ant in enumerate(ants) if ant.is_attackable_from(self.player, x, y, radius)]

    def find_targets(self, ants: Sequence[Ant], target_num: int) -> List[int]:
        idxs = self.get_attackable_ants(ants, self.x, self.y, self.range)
        idxs.sort(key=lambda idx: (distance(ants[idx].x, ants[idx].y, self.x, self.y), idx))
        return idxs[:target_num]

    def find_attackable(self, ants: Sequence[Ant], target_idxs: Sequence[int]) -> List[int]:
        attackable: List[int] = []
        for idx in target_idxs:
            if self.type in (TowerType.MORTAR, TowerType.MORTAR_PLUS):
                extra = self.get_attackable_ants(ants, ants[idx].x, ants[idx].y, 1)
            elif self.type == TowerType.PULSE:
                extra = self.get_attackable_ants(ants, self.x, self.y, self.range)
            elif self.type == TowerType.MISSILE:
                extra = self.get_attackable_ants(ants, ants[idx].x, ants[idx].y, 2)
            else:
                extra = [idx]
            attackable.extend(extra)
        return attackable

    def action(self, ant: Ant) -> None:
        if ant.evasion > 0:
            ant.evasion -= 1
            return
        if ant.deflector and self.damage < ant.max_hp() // 2:
            return
        ant.hp -= self.damage
        if self.type == TowerType.ICE:
            ant.state = AntState.FROZEN
        if ant.hp <= 0:
            ant.state = AntState.FAIL

    def attack(self, ants: List[Ant]) -> List[int]:
        attacked: List[int] = []
        if self.cd > 0:
            self.cd -= 1
        if self.cd <= 0:
            loops = 1 if self.speed >= 1 else int(1 / self.speed)
            target_num = 2 if self.type == TowerType.DOUBLE else 1
            while loops > 0:
                loops -= 1
                target_idxs = self.find_targets(ants, target_num)
                hits = self.find_attackable(ants, target_idxs)
                for idx in hits:
                    self.action(ants[idx])
                attacked.extend(hits)
            if attacked:
                attacked = sorted(set(attacked))
                self.reset_cd()
        return attacked

    def reset_cd(self) -> None:
        self.cd = int(self.speed) if self.speed > 1 else 1

    def upgrade(self, new_type: TowerType) -> None:
        self.type = TowerType(new_type)
        attack, speed, radius = TOWER_INFO[self.type]
        self.damage = attack
        self.speed = speed
        self.range = radius
        self.reset_cd()

    def is_upgrade_type_valid(self, target_type: int) -> bool:
        try:
            target = TowerType(target_type)
        except ValueError:
            return False
        if self.type == TowerType.BASIC:
            return target in (TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR)
        if self.type == TowerType.HEAVY:
            return target in (TowerType.HEAVY_PLUS, TowerType.ICE, TowerType.CANNON)
        if self.type == TowerType.QUICK:
            return target in (TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER)
        if self.type == TowerType.MORTAR:
            return target in (TowerType.MORTAR_PLUS, TowerType.PULSE, TowerType.MISSILE)
        return False

    def downgrade(self) -> None:
        self.upgrade(TowerType(self.type // 10))

    def is_downgrade_valid(self) -> bool:
        return self.type != TowerType.BASIC


@dataclass(slots=True)
class Base:
    player: int
    x: int
    y: int
    hp: int = 50
    gen_speed_level: int = 0
    ant_level: int = 0

    @classmethod
    def create(cls, player: int) -> Base:
        x, y = BASE_POS[player]
        return cls(player, x, y)

    def clone(self) -> Base:
        return Base(self.player, self.x, self.y, self.hp, self.gen_speed_level, self.ant_level)

    def generate_ant(self, ant_id: int, round_id: int) -> Optional[Ant]:
        if round_id % GENERATION_CYCLE[self.gen_speed_level] != 0:
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
    left_time: int = 0
    range: int = 0

    def __post_init__(self) -> None:
        duration, radius, _, _ = SUPER_WEAPON_INFO[self.type]
        self.left_time = duration + 1
        self.range = radius

    def in_range(self, x: int, y: int) -> bool:
        return distance(x, y, self.x, self.y) <= self.range

    def clone(self) -> SuperWeapon:
        copied = SuperWeapon(self.type, self.player, self.x, self.y)
        copied.left_time = self.left_time
        copied.range = self.range
        return copied


@dataclass(slots=True, frozen=True)
class Operation:
    type: OperationType
    arg0: int = -1
    arg1: int = -1

    def to_line(self) -> str:
        parts = [str(int(self.type))]
        if self.arg0 != -1:
            parts.append(str(self.arg0))
        if self.arg1 != -1:
            parts.append(str(self.arg1))
        return " ".join(parts)


@dataclass(slots=True)
class RoundPacket:
    round: int
    towers: List[Tower]
    ants: List[Ant]
    coin0: int
    coin1: int
    hp0: int
    hp1: int


class GameInfo:
    def __init__(self, seed: int) -> None:
        self.round = 0
        self.towers: List[Tower] = []
        self.ants: List[Ant] = []
        self.bases = [Base.create(0), Base.create(1)]
        self.coins = [COIN_INIT, COIN_INIT]
        self.pheromone = [[[0.0 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)] for _ in range(2)]
        self.building_tag = [[BuildingType.EMPTY for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
        self.super_weapons: List[SuperWeapon] = []
        self.super_weapon_cd = [[0 for _ in range(int(SuperWeaponType.COUNT))] for _ in range(2)]
        self.old_count = [0, 0]
        self.die_count = [0, 0]
        self.next_ant_id = 0
        self.next_tower_id = 0

        rng = LcgRandom(seed)
        for player in range(2):
            for x in range(MAP_SIZE):
                for y in range(MAP_SIZE):
                    self.pheromone[player][x][y] = rng.get() * pow(2, -46) + 8
        for player in range(2):
            bx, by = BASE_POS[player]
            self.building_tag[bx][by] = BuildingType.BASE

    def clone(self) -> GameInfo:
        copied = object.__new__(GameInfo)
        copied.round = self.round
        copied.towers = [tower.clone() for tower in self.towers]
        copied.ants = [ant.clone() for ant in self.ants]
        copied.bases = [base.clone() for base in self.bases]
        copied.coins = list(self.coins)
        copied.pheromone = [[[self.pheromone[p][x][y] for y in range(MAP_SIZE)] for x in range(MAP_SIZE)] for p in range(2)]
        copied.building_tag = [[self.building_tag[x][y] for y in range(MAP_SIZE)] for x in range(MAP_SIZE)]
        copied.super_weapons = [weapon.clone() for weapon in self.super_weapons]
        copied.super_weapon_cd = [list(row) for row in self.super_weapon_cd]
        copied.old_count = list(self.old_count)
        copied.die_count = list(self.die_count)
        copied.next_ant_id = self.next_ant_id
        copied.next_tower_id = self.next_tower_id
        return copied

    def tower_num_of_player(self, player: int) -> int:
        return sum(1 for tower in self.towers if tower.player == player)

    def tower_of_id(self, tower_id: int) -> Optional[Tower]:
        for tower in self.towers:
            if tower.id == tower_id:
                return tower
        return None

    def ant_of_id(self, ant_id: int) -> Optional[Ant]:
        for ant in self.ants:
            if ant.id == ant_id:
                return ant
        return None

    def build_tower(self, tower_id: int, player: int, x: int, y: int, tower_type: TowerType = TowerType.BASIC) -> None:
        self.towers.append(Tower(tower_id, player, x, y, tower_type))
        self.building_tag[x][y] = BuildingType.TOWER

    def upgrade_tower(self, tower_id: int, tower_type: TowerType) -> None:
        tower = self.tower_of_id(tower_id)
        if tower is not None:
            tower.upgrade(tower_type)

    def downgrade_or_destroy_tower(self, tower_id: int) -> None:
        for idx, tower in enumerate(self.towers):
            if tower.id != tower_id:
                continue
            if tower.is_downgrade_valid():
                tower.downgrade()
            else:
                self.building_tag[tower.x][tower.y] = BuildingType.EMPTY
                self.towers.pop(idx)
            return

    def set_coin(self, player: int, value: int) -> None:
        self.coins[player] = value

    def update_coin(self, player: int, delta: int) -> None:
        self.coins[player] += delta

    def set_base_hp(self, player: int, value: int) -> None:
        self.bases[player].hp = value

    def update_base_hp(self, player: int, delta: int) -> None:
        self.bases[player].hp += delta

    def upgrade_generation_speed(self, player: int) -> None:
        self.bases[player].upgrade_generation_speed()

    def upgrade_generated_ant(self, player: int) -> None:
        self.bases[player].upgrade_generated_ant()

    def clear_dead_and_succeeded_ants(self) -> None:
        survivors: List[Ant] = []
        for ant in self.ants:
            if ant.state == AntState.FAIL:
                self.die_count[ant.player] += 1
            elif ant.state == AntState.TOO_OLD:
                self.old_count[ant.player] += 1
            if ant.state not in (AntState.SUCCESS, AntState.FAIL, AntState.TOO_OLD):
                survivors.append(ant)
        self.ants = survivors

    def update_pheromone(self, ant: Ant) -> None:
        if ant.state in (AntState.ALIVE, AntState.FROZEN):
            return
        trail_gain = (0.0, 10.0, -5.0, -3.0)
        delta = trail_gain[int(ant.state)]
        x, y = BASE_POS[ant.player]
        seen = [[False for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
        for step in ant.path:
            if not seen[x][y]:
                seen[x][y] = True
                self.pheromone[ant.player][x][y] += delta
                if self.pheromone[ant.player][x][y] < PHEROMONE_MIN:
                    self.pheromone[ant.player][x][y] = PHEROMONE_MIN
            off_x, off_y = OFFSET[y % 2][step]
            x += off_x
            y += off_y
        if not seen[x][y]:
            self.pheromone[ant.player][x][y] += delta
            if self.pheromone[ant.player][x][y] < PHEROMONE_MIN:
                self.pheromone[ant.player][x][y] = PHEROMONE_MIN

    def update_pheromone_for_ants(self) -> None:
        for ant in self.ants:
            self.update_pheromone(ant)

    def global_pheromone_attenuation(self) -> None:
        for player in range(2):
            for x in range(MAP_SIZE):
                for y in range(MAP_SIZE):
                    if MAP_PROPERTY[x][y] >= 0:
                        self.pheromone[player][x][y] = (
                            PHEROMONE_ATTENUATING_RATIO * self.pheromone[player][x][y]
                            + (1 - PHEROMONE_ATTENUATING_RATIO) * PHEROMONE_INIT
                        )

    def is_shielded_by_emp(self, player: int, x: int, y: int) -> bool:
        return any(
            weapon.type == SuperWeaponType.EMP_BLASTER and weapon.player != player and weapon.in_range(x, y)
            for weapon in self.super_weapons
        )

    def tower_under_emp(self, tower: Tower) -> bool:
        return self.is_shielded_by_emp(tower.player, tower.x, tower.y)

    def is_shielded_by_deflector(self, ant: Ant) -> bool:
        return any(
            weapon.type == SuperWeaponType.DEFLECTOR and weapon.player == ant.player and weapon.in_range(ant.x, ant.y)
            for weapon in self.super_weapons
        )

    def is_operation_valid(self, player: int, op: Operation) -> bool:
        if op.type == OperationType.BUILD_TOWER:
            return is_valid_pos(op.arg0, op.arg1) and is_highland(player, op.arg0, op.arg1) and not self.is_shielded_by_emp(player, op.arg0, op.arg1)
        if op.type == OperationType.UPGRADE_TOWER:
            tower = self.tower_of_id(op.arg0)
            return tower is not None and tower.player == player and tower.is_upgrade_type_valid(op.arg1) and not self.tower_under_emp(tower)
        if op.type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_of_id(op.arg0)
            return tower is not None and tower.player == player and not self.tower_under_emp(tower)
        if op.type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            return is_valid_pos(op.arg0, op.arg1) and self.super_weapon_cd[player][int(op.type) % 10] <= 0
        if op.type == OperationType.UPGRADE_GENERATION_SPEED:
            return self.bases[player].gen_speed_level < 2
        if op.type == OperationType.UPGRADE_GENERATED_ANT:
            return self.bases[player].ant_level < 2
        return False

    def is_operation_sequence_valid(self, player: int, ops: Sequence[Operation], fresh: Operation) -> bool:
        if fresh.type == OperationType.BUILD_TOWER:
            collide = any(op.type == OperationType.BUILD_TOWER and op.arg0 == fresh.arg0 and op.arg1 == fresh.arg1 for op in ops)
        elif fresh.type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER):
            collide = any(op.type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER) and op.arg0 == fresh.arg0 for op in ops)
        elif fresh.type in (OperationType.UPGRADE_GENERATED_ANT, OperationType.UPGRADE_GENERATION_SPEED):
            collide = any(op.type in (OperationType.UPGRADE_GENERATED_ANT, OperationType.UPGRADE_GENERATION_SPEED) for op in ops)
        elif fresh.type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            collide = any(op.type == fresh.type for op in ops)
        else:
            return False
        if collide or not self.is_operation_valid(player, fresh):
            return False
        return self.check_affordable(player, [*ops, fresh])

    def get_operation_income(self, player: int, op: Operation) -> int:
        if op.type == OperationType.BUILD_TOWER:
            return -self.build_tower_cost(self.tower_num_of_player(player))
        if op.type == OperationType.UPGRADE_TOWER:
            return -self.upgrade_tower_cost(op.arg1)
        if op.type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_of_id(op.arg0)
            if tower is None:
                return 0
            if tower.type == TowerType.BASIC:
                return self.destroy_tower_income(self.tower_num_of_player(player))
            return self.downgrade_tower_income(int(tower.type))
        if op.type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            return -self.use_super_weapon_cost(int(op.type) % 10)
        if op.type == OperationType.UPGRADE_GENERATION_SPEED:
            return -self.upgrade_base_cost(self.bases[player].gen_speed_level)
        if op.type == OperationType.UPGRADE_GENERATED_ANT:
            return -self.upgrade_base_cost(self.bases[player].ant_level)
        return 0

    def check_affordable(self, player: int, ops: Sequence[Operation]) -> bool:
        income = 0
        tower_num = self.tower_num_of_player(player)
        for op in ops:
            if op.type == OperationType.BUILD_TOWER:
                income -= self.build_tower_cost(tower_num)
                tower_num += 1
            elif op.type == OperationType.DOWNGRADE_TOWER:
                tower = self.tower_of_id(op.arg0)
                if tower is None:
                    continue
                if tower.type == TowerType.BASIC:
                    income += self.destroy_tower_income(tower_num)
                    tower_num -= 1
                else:
                    income += self.downgrade_tower_income(int(tower.type))
            else:
                income += self.get_operation_income(player, op)
        return income + self.coins[player] >= 0

    def apply_operation(self, player: int, op: Operation) -> None:
        self.update_coin(player, self.get_operation_income(player, op))
        if op.type == OperationType.BUILD_TOWER:
            self.build_tower(self.next_tower_id, player, op.arg0, op.arg1)
            self.next_tower_id += 1
            return
        if op.type == OperationType.UPGRADE_TOWER:
            self.upgrade_tower(op.arg0, TowerType(op.arg1))
            return
        if op.type == OperationType.DOWNGRADE_TOWER:
            self.downgrade_or_destroy_tower(op.arg0)
            return
        if op.type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            self.use_super_weapon(SuperWeaponType(int(op.type) % 10), player, op.arg0, op.arg1)
            return
        if op.type == OperationType.UPGRADE_GENERATION_SPEED:
            self.upgrade_generation_speed(player)
            return
        if op.type == OperationType.UPGRADE_GENERATED_ANT:
            self.upgrade_generated_ant(player)

    def next_move(self, ant: Ant) -> int:
        target_x, target_y = BASE_POS[1 - ant.player]
        current = distance(ant.x, ant.y, target_x, target_y)
        weighted = [[-1.0, -1.0] for _ in range(6)]
        attraction = (1.25, 1.0, 0.75)
        for idx, (off_x, off_y) in enumerate(OFFSET[ant.y % 2]):
            x = ant.x + off_x
            y = ant.y + off_y
            if ant.path and ant.path[-1] == (idx + 3) % 6:
                continue
            if not is_path(x, y):
                continue
            next_dist = distance(x, y, target_x, target_y)
            gain = attraction[next_dist - current + 1]
            weighted[idx][0] = gain * self.pheromone[ant.player][x][y]
            weighted[idx][1] = self.pheromone[ant.player][x][y]
        return max(range(6), key=lambda idx: (weighted[idx][0], weighted[idx][1], -idx))

    @staticmethod
    def destroy_tower_income(tower_num: int) -> int:
        return int(GameInfo.build_tower_cost(tower_num - 1) * TOWER_DOWNGRADE_REFUND_RATIO)

    @staticmethod
    def downgrade_tower_income(tower_type: int) -> int:
        return int(GameInfo.upgrade_tower_cost(tower_type) * TOWER_DOWNGRADE_REFUND_RATIO)

    @staticmethod
    def build_tower_cost(tower_num: int) -> int:
        return int(TOWER_BUILD_PRICE_BASE * pow(TOWER_BUILD_PRICE_RATIO, tower_num))

    @staticmethod
    def upgrade_tower_cost(tower_type: int) -> int:
        if tower_type in (TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR):
            return LEVEL2_TOWER_UPGRADE_PRICE
        if tower_type in (
            TowerType.HEAVY_PLUS,
            TowerType.ICE,
            TowerType.CANNON,
            TowerType.QUICK_PLUS,
            TowerType.DOUBLE,
            TowerType.SNIPER,
            TowerType.MORTAR_PLUS,
            TowerType.PULSE,
            TowerType.MISSILE,
        ):
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
    def use_super_weapon_cost(weapon_type: int) -> int:
        return SUPER_WEAPON_INFO[SuperWeaponType(weapon_type)][3]

    def use_super_weapon(self, weapon_type: SuperWeaponType, player: int, x: int, y: int) -> None:
        weapon = SuperWeapon(weapon_type, player, x, y)
        if weapon.type == SuperWeaponType.EMERGENCY_EVASION:
            for ant in self.ants:
                if ant.player == weapon.player and weapon.in_range(ant.x, ant.y):
                    ant.evasion = 2
        else:
            self.super_weapons.append(weapon)
        self.super_weapon_cd[player][int(weapon_type)] = SUPER_WEAPON_INFO[weapon_type][2]

    def count_down_super_weapons_left_time(self, player: int) -> None:
        kept: List[SuperWeapon] = []
        for weapon in self.super_weapons:
            if weapon.player != player:
                kept.append(weapon)
                continue
            weapon.left_time -= 1
            if weapon.left_time > 0:
                kept.append(weapon)
        self.super_weapons = kept

    def count_down_super_weapons_cd(self) -> None:
        for player in range(2):
            for weapon_type in range(1, 5):
                self.super_weapon_cd[player][weapon_type] = max(self.super_weapon_cd[player][weapon_type] - 1, 0)


class Simulator:
    def __init__(self, info: Optional[GameInfo] = None) -> None:
        self.info = info.clone() if info is not None else None
        self.operations = [[], []]  # type: ignore[list-item]

    def clone(self) -> Simulator:
        copied = Simulator(self.info)
        copied.operations = [list(self.operations[0]), list(self.operations[1])]
        return copied

    def add_operation_of_player(self, player: int, op: Operation) -> bool:
        if self.info.is_operation_sequence_valid(player, self.operations[player], op):
            self.operations[player].append(op)
            return True
        return False

    def apply_operations_of_player(self, player: int) -> None:
        for op in self.operations[player]:
            self.info.apply_operation(player, op)

    def fast_next_round(self, perspective: int) -> bool:
        if self.info.round >= MAX_ROUND:
            return False

        kept_weapons: List[SuperWeapon] = []
        for weapon in self.info.super_weapons:
            weapon.left_time -= 1
            if weapon.left_time > 0:
                kept_weapons.append(weapon)
        self.info.super_weapons = kept_weapons

        self.info.ants = [ant for ant in self.info.ants if ant.player != perspective]
        self.info.towers = [tower for tower in self.info.towers if tower.player == perspective]

        for ant in self.info.ants:
            ant.deflector = False
        for tower in self.info.towers:
            tower.emp = False

        for weapon in self.info.super_weapons:
            if weapon.type == SuperWeaponType.LIGHTNING_STORM and weapon.player == perspective:
                for ant in self.info.ants:
                    if weapon.in_range(ant.x, ant.y):
                        ant.hp = 0
                        ant.state = AntState.FAIL
                        self.info.coins[weapon.player] += ANT_REWARD[ant.level]
            elif weapon.type == SuperWeaponType.DEFLECTOR and weapon.player == 1 - perspective:
                for ant in self.info.ants:
                    if weapon.in_range(ant.x, ant.y):
                        ant.deflector = True
            elif weapon.type == SuperWeaponType.EMP_BLASTER and weapon.player == 1 - perspective:
                for tower in self.info.towers:
                    if weapon.in_range(tower.x, tower.y):
                        tower.emp = True

        for tower in self.info.towers:
            if tower.emp:
                continue
            targets = tower.attack(self.info.ants)
            for idx in targets:
                if self.info.ants[idx].state == AntState.FAIL:
                    self.info.coins[tower.player] += ANT_REWARD[self.info.ants[idx].level]

        for ant in self.info.ants:
            ant.age += 1
            if ant.state == AntState.FAIL:
                continue
            if ant.age > Ant.AGE_LIMIT:
                ant.state = AntState.TOO_OLD
            if ant.state == AntState.ALIVE:
                ant.move(self.info.next_move(ant))
            if (ant.x, ant.y) == BASE_POS[1 - ant.player]:
                ant.state = AntState.SUCCESS
                self.info.bases[1 - ant.player].hp -= 1
                self.info.coins[ant.player] += 5
                if self.info.bases[1 - ant.player].hp <= 0:
                    return False
            if ant.state == AntState.FROZEN:
                ant.state = AntState.ALIVE

        enemy = 1 - perspective
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                if MAP_PROPERTY[x][y] >= 0:
                    self.info.pheromone[enemy][x][y] = PHEROMONE_ATTENUATING_RATIO * self.info.pheromone[enemy][x][y] + 0.3
        for ant in self.info.ants:
            self.info.update_pheromone(ant)

        survivors: List[Ant] = []
        for ant in self.info.ants:
            if ant.state == AntState.FAIL:
                self.info.die_count[ant.player] += 1
            elif ant.state == AntState.TOO_OLD:
                self.info.old_count[ant.player] += 1
            if ant.state not in (AntState.SUCCESS, AntState.FAIL, AntState.TOO_OLD):
                survivors.append(ant)
        self.info.ants = survivors

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
            for weapon_type in range(1, 5):
                if self.info.super_weapon_cd[player][weapon_type] > 0:
                    self.info.super_weapon_cd[player][weapon_type] -= 1
        self.operations[perspective].clear()
        return True


class TokenScanner:
    def __init__(self) -> None:
        self._queue: Deque[bytes] = deque()
        self._reader = sys.stdin.buffer

    def _fill(self) -> None:
        while not self._queue:
            line = self._reader.readline()
            if not line:
                raise EOFError
            self._queue.extend(line.split())

    def next_int(self) -> int:
        self._fill()
        return int(self._queue.popleft())


def read_init_info(scanner: TokenScanner) -> Tuple[int, int]:
    return scanner.next_int(), scanner.next_int()


def read_operations(scanner: TokenScanner) -> List[Operation]:
    count = scanner.next_int()
    ops: List[Operation] = []
    for _ in range(count):
        op_type = OperationType(scanner.next_int())
        if op_type in (OperationType.UPGRADE_GENERATED_ANT, OperationType.UPGRADE_GENERATION_SPEED):
            ops.append(Operation(op_type))
        elif op_type == OperationType.DOWNGRADE_TOWER:
            ops.append(Operation(op_type, scanner.next_int()))
        else:
            ops.append(Operation(op_type, scanner.next_int(), scanner.next_int()))
    return ops


def read_round_packet(scanner: TokenScanner) -> RoundPacket:
    round_id = scanner.next_int()
    tower_count = scanner.next_int()
    towers: List[Tower] = []
    for _ in range(tower_count):
        tower_id = scanner.next_int()
        player = scanner.next_int()
        x = scanner.next_int()
        y = scanner.next_int()
        tower_type = TowerType(scanner.next_int())
        cd = scanner.next_int()
        towers.append(Tower(tower_id, player, x, y, tower_type, cd))
    ant_count = scanner.next_int()
    ants: List[Ant] = []
    for _ in range(ant_count):
        ant_id = scanner.next_int()
        player = scanner.next_int()
        x = scanner.next_int()
        y = scanner.next_int()
        hp = scanner.next_int()
        level = scanner.next_int()
        age = scanner.next_int()
        state = AntState(scanner.next_int())
        ants.append(Ant(ant_id, player, x, y, hp, level, age, state))
    coin0 = scanner.next_int()
    coin1 = scanner.next_int()
    hp0 = scanner.next_int()
    hp1 = scanner.next_int()
    return RoundPacket(round_id, towers, ants, coin0, coin1, hp0, hp1)


def send_operations(ops: Sequence[Operation]) -> None:
    body = [str(len(ops))]
    body.extend(op.to_line() for op in ops)
    payload = ("\n".join(body) + "\n").encode()
    sys.stdout.buffer.write(struct.pack(">I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


class Controller:
    def __init__(self, scanner: Optional[TokenScanner] = None) -> None:
        self.scanner = scanner or TokenScanner()
        self.self_player_id, seed = read_init_info(self.scanner)
        self.info = GameInfo(seed)
        self.self_operations: List[Operation] = []
        self.opponent_operations: List[Operation] = []

    def get_info(self) -> GameInfo:
        return self.info

    def get_self_operations(self) -> Sequence[Operation]:
        return self.self_operations

    def get_opponent_operations(self) -> Sequence[Operation]:
        return self.opponent_operations

    def _update_towers(self, new_towers: List[Tower]) -> None:
        for tower in self.info.towers:
            self.info.building_tag[tower.x][tower.y] = BuildingType.EMPTY
        self.info.towers = new_towers
        for tower in self.info.towers:
            self.info.building_tag[tower.x][tower.y] = BuildingType.TOWER
        self.info.next_tower_id = 0 if not self.info.towers else self.info.towers[-1].id + 1

    def _update_ant(self, fresh: Ant) -> None:
        for ant in self.info.ants:
            if ant.id != fresh.id:
                continue
            if (ant.x, ant.y) != (fresh.x, fresh.y):
                ant.path.append(get_direction(ant.x, ant.y, fresh.x, fresh.y))
            ant.x = fresh.x
            ant.y = fresh.y
            ant.hp = fresh.hp
            ant.age = fresh.age
            ant.state = fresh.state
            return
        self.info.ants.append(fresh)

    def _update_ants(self, ants: Iterable[Ant]) -> None:
        for ant in ants:
            self._update_ant(ant)
        self.info.next_ant_id = 0 if not self.info.ants else self.info.ants[-1].id + 1

    def read_round_info(self) -> None:
        for weapon in self.info.super_weapons:
            if weapon.type != SuperWeaponType.LIGHTNING_STORM:
                continue
            for ant in self.info.ants:
                if weapon.in_range(ant.x, ant.y) and ant.player != weapon.player:
                    ant.hp = 0
                    ant.state = AntState.FAIL
                    self.info.update_coin(weapon.player, ant.reward())

        for ant in self.info.ants:
            ant.deflector = self.info.is_shielded_by_deflector(ant)
        for tower in self.info.towers:
            if self.info.tower_under_emp(tower):
                continue
            targets = tower.attack(self.info.ants)
            for idx in targets:
                if self.info.ants[idx].state == AntState.FAIL:
                    self.info.update_coin(tower.player, self.info.ants[idx].reward())
        for ant in self.info.ants:
            ant.deflector = False

        packet = read_round_packet(self.scanner)
        self._update_towers(packet.towers)
        self._update_ants(packet.ants)
        self.info.global_pheromone_attenuation()
        self.info.update_pheromone_for_ants()
        self.info.clear_dead_and_succeeded_ants()
        self.info.set_coin(0, packet.coin0)
        self.info.set_coin(1, packet.coin1)
        self.info.set_base_hp(0, packet.hp0)
        self.info.set_base_hp(1, packet.hp1)
        self.info.round = packet.round
        self.info.count_down_super_weapons_cd()
        self.self_operations.clear()
        self.opponent_operations.clear()

    def read_opponent_operations(self) -> None:
        self.opponent_operations = read_operations(self.scanner)

    def apply_opponent_operations(self) -> None:
        self.info.count_down_super_weapons_left_time(1 - self.self_player_id)
        for op in self.opponent_operations:
            self.info.apply_operation(1 - self.self_player_id, op)

    def append_self_operation(self, op: Operation) -> bool:
        if self.info.is_operation_sequence_valid(self.self_player_id, self.self_operations, op):
            self.self_operations.append(op)
            return True
        return False

    def apply_self_operations(self) -> None:
        self.info.count_down_super_weapons_left_time(self.self_player_id)
        for op in self.self_operations:
            self.info.apply_operation(self.self_player_id, op)

    def send_self_operations(self) -> None:
        send_operations(self.self_operations)
