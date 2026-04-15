from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import List, Optional, Sequence

from SDK.utils.constants import (
    ANT_AGE_LIMIT,
    ANT_GENERATION_SCHEDULE,
    ANT_KILL_REWARD,
    COMBAT_ANT_KILL_REWARD,
    ANT_MAX_HP,
    COMBAT_ANT_HP,
    BASIC_INCOME,
    BASIC_INCOME_INTERVAL,
    BASE_UPGRADE_COST,
    INITIAL_COINS,
    MAP_PROPERTY,
    MAP_SIZE,
    MAX_ROUND,
    OFFSET,
    OperationType,
    PHEROMONE_ATTENUATION,
    PHEROMONE_FLOOR,
    PHEROMONE_INIT,
    PHEROMONE_SCALE,
    PLAYER_BASES,
    DEFLECTOR_PATH_ATTRACTION,
    EMERGENCY_EVASION_PATH_ATTRACTION,
    LIGHTNING_STORM_ANT_DAMAGE,
    LIGHTNING_STORM_TOWER_DAMAGE,
    LIGHTNING_STORM_TOWER_INTERVAL,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TOWER_DOWNGRADE_REFUND_RATIO,
    TOWER_STATS,
    TOWER_UPGRADE_TREE,
    AntKind,
    TowerType,
    AntStatus,
    tower_build_cost_for_count,
)
from SDK.utils.geometry import hex_distance, is_highland, is_path, is_valid_pos

AntState = AntStatus
BASE_POS = PLAYER_BASES
COIN_INIT = INITIAL_COINS
PHEROMONE_MIN = PHEROMONE_FLOOR
PHEROMONE_ATTENUATING_RATIO = PHEROMONE_ATTENUATION
LEVEL2_BASE_UPGRADE_PRICE, LEVEL3_BASE_UPGRADE_PRICE = BASE_UPGRADE_COST
NO_MOVE = -1


class BuildingType(IntEnum):
    EMPTY = 0
    TOWER = 1
    BASE = 2


class LcgRandom:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def get(self) -> int:
        self.seed = (25214903917 * self.seed) & ((1 << 48) - 1)
        return self.seed


def _trail_for_pheromone(ant: Ant) -> List[tuple[int, int]]:
    trail = list(ant.trail_cells)
    if not trail or trail[-1] != (ant.x, ant.y):
        trail.append((ant.x, ant.y))
    return trail


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
    trail_cells: List[tuple[int, int]] = field(default_factory=list)
    last_move: int = NO_MOVE
    path_len_total: int = 0
    kind: AntKind = AntKind.WORKER

    AGE_LIMIT = ANT_AGE_LIMIT

    def __post_init__(self) -> None:
        if not self.trail_cells:
            self.trail_cells.append((self.x, self.y))

    def record_move(self, direction: int) -> None:
        self.path_len_total += 1
        if direction == NO_MOVE:
            self.last_move = NO_MOVE
            return
        off_x, off_y = OFFSET[self.y % 2][direction]
        self.x += off_x
        self.y += off_y
        self.last_move = direction
        self.trail_cells.append((self.x, self.y))

    def teleport_to(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.last_move = NO_MOVE
        self.trail_cells.append((self.x, self.y))

    def max_hp(self) -> int:
        if self.kind == AntKind.COMBAT:
            return COMBAT_ANT_HP
        return ANT_MAX_HP[self.level]

    def reward(self) -> int:
        if self.kind == AntKind.COMBAT:
            return COMBAT_ANT_KILL_REWARD
        return ANT_KILL_REWARD[self.level]

    def is_alive(self) -> bool:
        return self.state in (AntState.ALIVE, AntState.FROZEN)

    def in_range(self, x: int, y: int, radius: int) -> bool:
        return hex_distance(self.x, self.y, x, y) <= radius

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
            list(self.trail_cells),
            self.last_move,
            self.path_len_total,
            self.kind,
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
    hp: int = -1

    def __post_init__(self) -> None:
        seeded_cd = self.cd
        self.refresh_stats()
        if self.hp < 0:
            self.hp = self.max_hp
        if seeded_cd != -2:
            self.cd = seeded_cd

    def clone(self) -> Tower:
        copied = Tower(self.id, self.player, self.x, self.y, self.type, self.cd, hp=self.hp)
        copied.emp = self.emp
        copied.damage = self.damage
        copied.range = self.range
        copied.speed = self.speed
        return copied

    @property
    def max_hp(self) -> int:
        return TOWER_STATS[self.type].max_hp

    def get_attackable_ants(self, ants: Sequence[Ant], x: int, y: int, radius: int) -> List[int]:
        return [idx for idx, ant in enumerate(ants) if ant.is_attackable_from(self.player, x, y, radius)]

    def find_targets(self, ants: Sequence[Ant], target_num: int) -> List[int]:
        idxs = self.get_attackable_ants(ants, self.x, self.y, self.range)
        idxs.sort(key=lambda idx: (hex_distance(ants[idx].x, ants[idx].y, self.x, self.y), idx))
        return idxs[:target_num]

    def find_attackable(self, ants: Sequence[Ant], target_idxs: Sequence[int]) -> List[int]:
        attackable: List[int] = []
        for idx in target_idxs:
            if self.type in (TowerType.MORTAR, TowerType.MORTAR_PLUS):
                extra = self.get_attackable_ants(ants, ants[idx].x, ants[idx].y, self.range)
            elif self.type == TowerType.PULSE:
                extra = self.get_attackable_ants(ants, self.x, self.y, self.range)
            elif self.type == TowerType.MISSILE:
                extra = self.get_attackable_ants(ants, ants[idx].x, ants[idx].y, self.range)
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
        if self.cd > 0 or self.speed <= 0 or self.range <= 0 or self.damage <= 0:
            return attacked
        if self.cd <= 0:
            loops = 1 if self.speed >= 1 else int(round(1 / self.speed))
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

    def refresh_stats(self) -> None:
        stats = TOWER_STATS[self.type]
        self.damage = stats.damage
        self.speed = stats.speed
        self.range = stats.attack_range
        self.reset_cd()

    def upgrade(self, new_type: TowerType) -> None:
        self.type = TowerType(new_type)
        self.refresh_stats()
        self.hp = self.max_hp

    def is_upgrade_type_valid(self, target_type: int | TowerType) -> bool:
        try:
            target = TowerType(int(target_type))
        except ValueError:
            return False
        return target in TOWER_UPGRADE_TREE.get(self.type, ())

    def downgrade(self) -> None:
        previous_hp = max(0, self.hp)
        previous_max_hp = self.max_hp
        self.type = TowerType(self.type // 10)
        self.refresh_stats()
        if previous_max_hp > 0:
            self.hp = max(1, math.ceil(self.max_hp * previous_hp / previous_max_hp))
        else:
            self.hp = self.max_hp

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
        numerator, denominator = ANT_GENERATION_SCHEDULE[self.gen_speed_level]
        if round_id != 0 and (round_id * denominator) // numerator <= ((round_id - 1) * denominator) // numerator:
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
        stats = SUPER_WEAPON_STATS[self.type]
        self.left_time = stats.duration + 1
        self.range = stats.attack_range

    def in_range(self, x: int, y: int) -> bool:
        return hex_distance(x, y, self.x, self.y) <= self.range

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


class GameInfo:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.round = 0
        self.towers: List[Tower] = []
        self.ants: List[Ant] = []
        self.bases = [Base.create(0), Base.create(1)]
        self.coins = [COIN_INIT, COIN_INIT]
        self.pheromone = [[[0.0 for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)] for _ in range(2)]
        self.building_tag = [[BuildingType.EMPTY for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
        self.super_weapons: List[SuperWeapon] = []
        self.super_weapon_cd = [[0 for _ in range(5)] for _ in range(2)]
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
        copied.seed = self.seed
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
        trail_gain = {
            AntState.SUCCESS: 10.0,
            AntState.FAIL: -5.0,
            AntState.TOO_OLD: -3.0,
        }
        delta = trail_gain.get(ant.state, 0.0)
        seen = [[False for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
        for x, y in reversed(_trail_for_pheromone(ant)):
            if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
                continue
            if seen[x][y]:
                continue
            seen[x][y] = True
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
            return (
                is_valid_pos(op.arg0, op.arg1)
                and is_highland(player, op.arg0, op.arg1)
                and self.building_tag[op.arg0][op.arg1] == BuildingType.EMPTY
                and not self.is_shielded_by_emp(player, op.arg0, op.arg1)
            )
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
                return self.destroy_tower_income(self.tower_num_of_player(player), tower)
            return self.downgrade_tower_income(int(tower.type), tower)
        if op.type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            return -self.use_super_weapon_cost(int(op.type) % 10)
        if op.type == OperationType.UPGRADE_GENERATION_SPEED:
            level = self.bases[player].gen_speed_level
            return -self.upgrade_base_cost(level) if level < len(BASE_UPGRADE_COST) else 0
        if op.type == OperationType.UPGRADE_GENERATED_ANT:
            level = self.bases[player].ant_level
            return -self.upgrade_base_cost(level) if level < len(BASE_UPGRADE_COST) else 0
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
                    income += self.destroy_tower_income(tower_num, tower)
                    tower_num -= 1
                else:
                    income += self.downgrade_tower_income(int(tower.type), tower)
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
        current = hex_distance(ant.x, ant.y, target_x, target_y)
        weighted = [[-1.0, -1.0] for _ in range(6)]
        attraction = (1.25, 1.0, 0.75)
        for idx, (off_x, off_y) in enumerate(OFFSET[ant.y % 2]):
            x = ant.x + off_x
            y = ant.y + off_y
            if ant.last_move == (idx + 3) % 6:
                continue
            if not is_path(x, y):
                continue
            next_dist = hex_distance(x, y, target_x, target_y)
            gain = attraction[next_dist - current + 1]
            storm_penalty = 0.0
            effect_pull = 0.0
            for weapon in self.super_weapons:
                if not weapon.in_range(x, y):
                    continue
                if weapon.type == SuperWeaponType.LIGHTNING_STORM and weapon.player != ant.player:
                    storm_penalty += LIGHTNING_STORM_ANT_DAMAGE / 25.0
                elif weapon.player == ant.player and weapon.type == SuperWeaponType.DEFLECTOR:
                    effect_pull += DEFLECTOR_PATH_ATTRACTION
                elif weapon.player == ant.player and weapon.type == SuperWeaponType.EMERGENCY_EVASION:
                    effect_pull += EMERGENCY_EVASION_PATH_ATTRACTION
            weighted[idx][0] = gain * self.pheromone[ant.player][x][y] + effect_pull - storm_penalty
            weighted[idx][1] = self.pheromone[ant.player][x][y]
        return max(range(6), key=lambda idx: (weighted[idx][0], weighted[idx][1], -idx))

    @staticmethod
    def destroy_tower_income(tower_num: int, tower: Tower | None = None) -> int:
        refund = GameInfo.build_tower_cost(tower_num - 1) * TOWER_DOWNGRADE_REFUND_RATIO
        if tower is None:
            return int(refund)
        return int(refund * max(tower.hp, 0) / max(tower.max_hp, 1))

    @staticmethod
    def downgrade_tower_income(tower_type: int, tower: Tower | None = None) -> int:
        refund = GameInfo.upgrade_tower_cost(tower_type) * TOWER_DOWNGRADE_REFUND_RATIO
        if tower is None:
            return int(refund)
        return int(refund * max(tower.hp, 0) / max(tower.max_hp, 1))

    @staticmethod
    def build_tower_cost(tower_num: int) -> int:
        return tower_build_cost_for_count(tower_num)

    @staticmethod
    def upgrade_tower_cost(tower_type: int) -> int:
        if tower_type in (TowerType.HEAVY, TowerType.QUICK, TowerType.MORTAR):
            return 60
        if tower_type in (
            TowerType.HEAVY_PLUS,
            TowerType.ICE,
            TowerType.BEWITCH,
            TowerType.QUICK_PLUS,
            TowerType.DOUBLE,
            TowerType.SNIPER,
            TowerType.MORTAR_PLUS,
            TowerType.PULSE,
            TowerType.MISSILE,
        ):
            return 200
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
        return SUPER_WEAPON_STATS[SuperWeaponType(weapon_type)].cost

    def use_super_weapon(self, weapon_type: SuperWeaponType, player: int, x: int, y: int) -> None:
        weapon = SuperWeapon(weapon_type, player, x, y)
        if weapon.type == SuperWeaponType.EMERGENCY_EVASION:
            for ant in self.ants:
                if ant.player == weapon.player and weapon.in_range(ant.x, ant.y):
                    ant.evasion = 2
        self.super_weapons.append(weapon)
        self.super_weapon_cd[player][int(weapon_type)] = SUPER_WEAPON_STATS[weapon_type].cooldown

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

        for weapon in self.info.super_weapons:
            if weapon.type != SuperWeaponType.LIGHTNING_STORM or weapon.player != perspective:
                continue
            active_turn = SUPER_WEAPON_STATS[weapon.type].duration - weapon.left_time + 1
            if active_turn <= 0 or active_turn % LIGHTNING_STORM_TOWER_INTERVAL != 0:
                continue
            surviving_towers: List[Tower] = []
            for tower in self.info.towers:
                if tower.player != weapon.player and weapon.in_range(tower.x, tower.y):
                    tower.hp -= LIGHTNING_STORM_TOWER_DAMAGE
                    if tower.hp <= 0:
                        continue
                surviving_towers.append(tower)
            self.info.towers = surviving_towers

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
                        ant.hp -= LIGHTNING_STORM_ANT_DAMAGE
                        if ant.hp <= 0:
                            ant.state = AntState.FAIL
                            self.info.coins[weapon.player] += ant.reward()
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
                    self.info.coins[tower.player] += self.info.ants[idx].reward()

        for ant in self.info.ants:
            ant.age += 1
            if ant.state != AntState.FAIL and ant.kind != AntKind.COMBAT and ant.age > Ant.AGE_LIMIT:
                ant.state = AntState.TOO_OLD
            direction = NO_MOVE
            if ant.state == AntState.ALIVE:
                direction = self.info.next_move(ant)
            ant.record_move(direction)
            if ant.state != AntState.FAIL and (ant.x, ant.y) == BASE_POS[1 - ant.player]:
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
                    self.info.pheromone[enemy][x][y] = PHEROMONE_ATTENUATING_RATIO * self.info.pheromone[enemy][x][y] + (1 - PHEROMONE_ATTENUATING_RATIO) * PHEROMONE_INIT
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
        spawned = base.generate_ant(self.info.next_ant_id, self.info.round)
        if spawned is not None:
            self.info.ants.append(spawned)
            self.info.next_ant_id += 1

        if (self.info.round + 1) % BASIC_INCOME_INTERVAL == 0:
            self.info.coins[0] += BASIC_INCOME
            self.info.coins[1] += BASIC_INCOME
            if self.info.round % 3 != 0:
                self.info.coins[enemy] += BASIC_INCOME

        self.info.round += 1
        for player in range(2):
            for weapon_type in range(1, 5):
                if self.info.super_weapon_cd[player][weapon_type] > 0:
                    self.info.super_weapon_cd[player][weapon_type] -= 1
        self.operations[perspective].clear()
        return True


def build_forecast_state(state) -> GameInfo:
    info = GameInfo(int(state.seed))
    info.round = int(state.round_index)
    info.coins = [int(value) for value in state.coins]
    info.old_count = [int(value) for value in state.old_count]
    info.die_count = [int(value) for value in state.die_count]
    info.next_ant_id = int(state.next_ant_id)
    info.next_tower_id = int(state.next_tower_id)
    info.super_weapon_cd = [[int(value) for value in row] for row in state.weapon_cooldowns.tolist()]

    info.bases = [
        Base(
            player=base.player,
            x=base.x,
            y=base.y,
            hp=base.hp,
            gen_speed_level=base.generation_level,
            ant_level=base.ant_level,
        )
        for base in state.bases
    ]

    info.building_tag = [[BuildingType.EMPTY for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
    for base in info.bases:
        info.building_tag[base.x][base.y] = BuildingType.BASE

    info.towers = []
    for tower in state.towers:
        info.towers.append(
            Tower(
                id=tower.tower_id,
                player=tower.player,
                x=tower.x,
                y=tower.y,
                type=TowerType(int(tower.tower_type)),
                cd=tower.display_cooldown(),
            )
        )
        info.building_tag[tower.x][tower.y] = BuildingType.TOWER

    info.ants = [
        Ant(
            id=ant.ant_id,
            player=ant.player,
            x=ant.x,
            y=ant.y,
            hp=ant.hp,
            level=ant.level,
            age=ant.age,
            state=AntState(int(ant.status)),
            evasion=2 if ant.evasion else 0,
            deflector=bool(ant.deflector),
            trail_cells=list(ant.trail_cells),
            last_move=int(ant.last_move),
            path_len_total=int(ant.path_len_total),
            kind=AntKind(int(ant.kind)),
        )
        for ant in state.ants
    ]

    info.pheromone = [
        [
            [state.pheromone[player, x, y] / float(PHEROMONE_SCALE) for y in range(MAP_SIZE)]
            for x in range(MAP_SIZE)
        ]
        for player in range(2)
    ]

    info.super_weapons = []
    for effect in state.active_effects:
        weapon = SuperWeapon(
            type=SuperWeaponType(int(effect.weapon_type)),
            player=effect.player,
            x=effect.x,
            y=effect.y,
        )
        weapon.left_time = effect.remaining_turns
        info.super_weapons.append(weapon)
    return info


ForecastState = GameInfo
ForecastOperation = Operation
ForecastSimulator = Simulator


__all__ = [
    "Ant",
    "AntState",
    "BASE_POS",
    "Base",
    "BuildingType",
    "ForecastOperation",
    "ForecastSimulator",
    "ForecastState",
    "GameInfo",
    "MAP_PROPERTY",
    "MAP_SIZE",
    "MAX_ROUND",
    "Operation",
    "OperationType",
    "Simulator",
    "SuperWeapon",
    "SuperWeaponType",
    "Tower",
    "TowerType",
    "build_forecast_state",
    "hex_distance",
    "is_valid_pos",
]
