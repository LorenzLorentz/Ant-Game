from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from heapq import heappop, heappush
from typing import Iterable

import numpy as np

from SDK.utils.constants import (
    AntKind,
    AntBehavior,
    ANT_BREACH_REWARD,
    ANT_TELEPORT_INTERVAL,
    ANT_TELEPORT_RATIO,
    BASE_HP,
    BASE_UPGRADE_COST,
    BEWITCH_MOVE_TEMPERATURE,
    BASIC_INCOME,
    BASIC_INCOME_INTERVAL,
    CENTERLINE_WEIGHTS,
    COMBAT_INITIAL_EVASION,
    CROWDING_PENALTY,
    DEFAULT_MOVE_TEMPERATURE,
    HIGHLAND_CELLS,
    INITIAL_COINS,
    LAMBDA_DENOM,
    LAMBDA_NUM,
    MAP_SIZE,
    MAX_ROUND,
    COMBAT_SELF_DESTRUCT_DAMAGE,
    COMBAT_SELF_DESTRUCT_RANGE,
    COMBAT_RISK_FIELD_DISTANCE_DECAY,
    OFFSET,
    OperationType,
    PATH_CELLS,
    PHEROMONE_FAIL_BONUS_INT,
    PHEROMONE_INIT_INT,
    PHEROMONE_SUCCESS_BONUS_INT,
    PHEROMONE_TOO_OLD_BONUS_INT,
    TAU_BASE_ADD_INT,
    PLAYER_BASES,
    PLAYER_COUNT,
    RANDOM_ANT_DECAY_TURNS,
    RETREAT_MOVE_PENALTY,
    SPAWN_PROFILE_WEIGHTS,
    STALL_MOVE_PENALTY,
    STRATEGIC_BUILD_ORDER,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TARGET_PULL_DISTANCE_SCALE,
    TOWER_DOWNGRADE_REFUND_RATIO,
    TOWER_STATS,
    TOWER_UPGRADE_TREE,
    TowerType,
    WORKER_RISK_FIELD_DISTANCE_DECAY,
    WeaponStats,
    AntStatus,
    LEVEL2_TOWER_UPGRADE_COST,
    LEVEL3_TOWER_UPGRADE_COST,
    DEFLECTOR_PATH_ATTRACTION,
    EMERGENCY_EVASION_PATH_ATTRACTION,
    LIGHTNING_STORM_ANT_DAMAGE,
    LIGHTNING_STORM_TOWER_DAMAGE,
    LIGHTNING_STORM_TOWER_INTERVAL,
    tower_build_cost_for_count,
)
from SDK.backend.model import NO_MOVE, Ant, Base, Operation, Tower, WeaponEffect, default_behavior_expiry
from SDK.utils.geometry import hex_distance, is_highland, is_path, is_valid_pos, neighbors

RNG_MASK = (1 << 48) - 1
RNG_MULTIPLIER = 25214903917
RNG_INCREMENT = 11
RANDOM_FLOAT_BITS = 24
DAMAGE_FIELD_HP_REFERENCE = 25.0
WALKABLE_CELLS = PATH_CELLS + PLAYER_BASES
COMBAT_SELF_DESTRUCT_PULL_BONUS = 3.0
COMBAT_TOWER_TARGET_BONUS = 8.0
COMBAT_TOWER_APPROACH_PULL_BASE = 8.0
WORKER_TOWER_TARGET_BONUS = 2.75
MOVEMENT_POLICY_LEGACY = "legacy"
MOVEMENT_POLICY_ENHANCED = "enhanced"
DEFAULT_MOVEMENT_POLICY = MOVEMENT_POLICY_ENHANCED
WORKER_PATH_DAMAGE_WEIGHT = 0.20
WORKER_PATH_CONTROL_WEIGHT = 1.80
WORKER_PATH_TRAFFIC_WEIGHT = 0.75
WORKER_PATH_EFFECT_WEIGHT = 0.35
WORKER_RESERVATION_WEIGHT = 1.40
WORKER_TOWER_CLAIM_WEIGHT = 1.00
WORKER_BLOCKED_ATTACK_BONUS = 6.00
WORKER_ROUTE_IMPROVEMENT_EPS = 0.50
COMBAT_PATH_DAMAGE_WEIGHT = 0.08
COMBAT_PATH_CONTROL_WEIGHT = 0.45
COMBAT_PATH_TRAFFIC_WEIGHT = 0.25
COMBAT_PATH_EFFECT_WEIGHT = 0.20
COMBAT_RESERVATION_WEIGHT = 0.45
COMBAT_TOWER_CLAIM_WEIGHT = 0.85
COMBAT_TRAVEL_COST_WEIGHT = 0.90
ATTACK_FINISH_BONUS = 3.00
SURPLUS_HP_VALUE_WEIGHT = 0.15
ENHANCED_COMBAT_ATTACK_EXECUTION_BONUS = 1.50
WORKER_REROUTE_ATTACK_PENALTY_WEIGHT = 1.0
MIN_PATH_STEP_COST = 0.15


def _half_plane_delta(player: int, x: int, y: int) -> int:
    own_base = PLAYER_BASES[player]
    enemy_base = PLAYER_BASES[1 - player]
    return hex_distance(x, y, *own_base) - hex_distance(x, y, *enemy_base)


@lru_cache(maxsize=None)
def _bewitch_cells(player: int, anchor_delta: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (x, y)
        for x, y in WALKABLE_CELLS
        if _half_plane_delta(player, x, y) <= anchor_delta
    )


def _softmax_choice(weights: list[float], temperature: float) -> list[float]:
    if not weights:
        return []
    scale = max(temperature, 1e-6)
    max_weight = max(weights)
    exps = [float(np.exp((weight - max_weight) / scale)) for weight in weights]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [value / total for value in exps]


def _trail_for_pheromone(ant: Ant) -> list[tuple[int, int]]:
    trail = list(ant.trail_cells)
    if not trail or trail[-1] != (ant.x, ant.y):
        trail.append((ant.x, ant.y))
    return trail


def _is_ant_walkable_cell(x: int, y: int) -> bool:
    return (x, y) in PLAYER_BASES or is_path(x, y)


@dataclass(slots=True)
class PublicRoundState:
    round_index: int
    towers: list[tuple[int, ...]]
    ants: list[tuple[int, ...]]
    coins: tuple[int, int]
    camps_hp: tuple[int, int]
    speed_lv: tuple[int, int] | None = None
    anthp_lv: tuple[int, int] | None = None
    weapon_cooldowns: tuple[tuple[int, ...], ...] | None = None
    active_effects: list[tuple[int, ...]] | None = None


@dataclass(slots=True)
class TurnResolution:
    operations: tuple[list[Operation], list[Operation]]
    illegal: tuple[list[Operation], list[Operation]]
    terminal: bool
    winner: int | None


@dataclass(slots=True)
class EnhancedTowerPlan:
    total_cost: np.ndarray
    damage_cost: np.ndarray


@dataclass(slots=True)
class EnhancedMoveAnnotation:
    next_cell: tuple[int, int] | None = None
    tower_id: int | None = None


@dataclass(slots=True)
class GameState:
    seed: int = 0
    movement_policy: str = DEFAULT_MOVEMENT_POLICY
    cold_handle_rule_illegal: bool = False
    round_index: int = 0
    towers: list[Tower] = field(default_factory=list)
    ants: list[Ant] = field(default_factory=list)
    bases: list[Base] = field(default_factory=list)
    coins: list[int] = field(default_factory=lambda: [INITIAL_COINS, INITIAL_COINS])
    pheromone: np.ndarray = field(default_factory=lambda: np.zeros((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.int32))
    damage_risk_field: np.ndarray = field(default_factory=lambda: np.zeros((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.float32))
    control_risk_field: np.ndarray = field(default_factory=lambda: np.zeros((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.float32))
    effect_pull_field: np.ndarray = field(default_factory=lambda: np.zeros((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.float32))
    weapon_cooldowns: np.ndarray = field(default_factory=lambda: np.zeros((PLAYER_COUNT, 5), dtype=np.int16))
    active_effects: list[WeaponEffect] = field(default_factory=list)
    old_count: list[int] = field(default_factory=lambda: [0, 0])
    die_count: list[int] = field(default_factory=lambda: [0, 0])
    super_weapon_usage: list[int] = field(default_factory=lambda: [0, 0])
    ai_time: list[int] = field(default_factory=lambda: [0, 0])
    next_ant_id: int = 0
    next_tower_id: int = 0
    terminal: bool = False
    winner: int | None = None
    rng_state: int = 0
    risk_fields_dirty: bool = True
    enhanced_move_phase_active: bool = False
    enhanced_move_cache_dirty: bool = True
    enhanced_worker_costs: np.ndarray = field(
        default_factory=lambda: np.full((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), np.inf, dtype=np.float32)
    )
    enhanced_combat_base_costs: np.ndarray = field(
        default_factory=lambda: np.full((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), np.inf, dtype=np.float32)
    )
    enhanced_traffic_field: np.ndarray = field(
        default_factory=lambda: np.zeros((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    )
    enhanced_reservations: np.ndarray = field(
        default_factory=lambda: np.zeros((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    )
    enhanced_tower_plans: list[dict[int, EnhancedTowerPlan]] = field(default_factory=lambda: [dict(), dict()])
    enhanced_tower_claims: list[dict[int, int]] = field(default_factory=lambda: [dict(), dict()])
    enhanced_move_annotations: dict[int, EnhancedMoveAnnotation] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        seed: int = 0,
        movement_policy: str = DEFAULT_MOVEMENT_POLICY,
        cold_handle_rule_illegal: bool = False,
    ) -> GameState:
        state = cls(
            seed=seed,
            movement_policy=movement_policy,
            cold_handle_rule_illegal=cold_handle_rule_illegal,
        )
        state.bases = [Base(0, *PLAYER_BASES[0], hp=BASE_HP), Base(1, *PLAYER_BASES[1], hp=BASE_HP)]
        state._init_pheromone(seed)
        state.rng_state = (seed ^ RNG_MULTIPLIER) & RNG_MASK
        return state

    def clone(self) -> GameState:
        return GameState(
            seed=self.seed,
            movement_policy=self.movement_policy,
            cold_handle_rule_illegal=self.cold_handle_rule_illegal,
            round_index=self.round_index,
            towers=[tower.clone() for tower in self.towers],
            ants=[ant.clone() for ant in self.ants],
            bases=[base.clone() for base in self.bases],
            coins=list(self.coins),
            pheromone=self.pheromone.copy(),
            damage_risk_field=self.damage_risk_field.copy(),
            control_risk_field=self.control_risk_field.copy(),
            effect_pull_field=self.effect_pull_field.copy(),
            weapon_cooldowns=self.weapon_cooldowns.copy(),
            active_effects=[effect.clone() for effect in self.active_effects],
            old_count=list(self.old_count),
            die_count=list(self.die_count),
            super_weapon_usage=list(self.super_weapon_usage),
            ai_time=list(self.ai_time),
            next_ant_id=self.next_ant_id,
            next_tower_id=self.next_tower_id,
            terminal=self.terminal,
            winner=self.winner,
            rng_state=self.rng_state,
            risk_fields_dirty=self.risk_fields_dirty,
            enhanced_move_phase_active=False,
            enhanced_move_cache_dirty=True,
        )

    def _init_pheromone(self, seed: int) -> None:
        value = seed & ((1 << 48) - 1)
        for player in range(PLAYER_COUNT):
            for x in range(MAP_SIZE):
                for y in range(MAP_SIZE):
                    value = (25214903917 * value) & ((1 << 48) - 1)
                    self.pheromone[player, x, y] = PHEROMONE_INIT_INT + (value * 10000 >> 46)

    def _next_random(self) -> int:
        self.rng_state = (RNG_MULTIPLIER * self.rng_state + RNG_INCREMENT) & RNG_MASK
        return self.rng_state

    def _random_float(self) -> float:
        return float((self._next_random() >> (48 - RANDOM_FLOAT_BITS)) & ((1 << RANDOM_FLOAT_BITS) - 1)) / float(1 << RANDOM_FLOAT_BITS)

    def _random_index(self, bound: int) -> int:
        if bound <= 1:
            return 0
        return int(self._next_random() % bound)

    def _sample_index(self, probabilities: list[float]) -> int:
        if not probabilities:
            return 0
        threshold = self._random_float()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if threshold <= cumulative:
                return index
        return len(probabilities) - 1

    def tower_count(self, player: int) -> int:
        return sum(1 for tower in self.towers if tower.player == player)

    def towers_of(self, player: int) -> list[Tower]:
        return [tower for tower in self.towers if tower.player == player]

    def ants_of(self, player: int) -> list[Ant]:
        return [ant for ant in self.ants if ant.player == player and ant.is_alive()]

    def tower_at(self, x: int, y: int) -> Tower | None:
        for tower in self.towers:
            if tower.x == x and tower.y == y:
                return tower
        return None

    def tower_by_id(self, tower_id: int) -> Tower | None:
        for tower in self.towers:
            if tower.tower_id == tower_id:
                return tower
        return None

    def strategic_slots(self, player: int) -> tuple[tuple[int, int], ...]:
        return STRATEGIC_BUILD_ORDER[player]

    def build_tower_cost(self, tower_count: int | None = None) -> int:
        if tower_count is None:
            tower_count = self.tower_count(0)
        return tower_build_cost_for_count(tower_count)

    def upgrade_tower_cost(self, target_type: TowerType) -> int:
        if target_type.value < 10:
            return LEVEL2_TOWER_UPGRADE_COST
        return LEVEL3_TOWER_UPGRADE_COST

    def destroy_tower_income(self, tower_count: int, tower: Tower | None = None) -> int:
        refund = self.build_tower_cost(tower_count - 1) * TOWER_DOWNGRADE_REFUND_RATIO
        if tower is None:
            return int(refund)
        return int(refund * max(tower.hp, 0) / max(tower.max_hp, 1))

    def downgrade_tower_income(self, tower_type: TowerType, tower: Tower | None = None) -> int:
        refund = self.upgrade_tower_cost(tower_type) * TOWER_DOWNGRADE_REFUND_RATIO
        if tower is None:
            return int(refund)
        return int(refund * max(tower.hp, 0) / max(tower.max_hp, 1))

    def upgrade_base_cost(self, level: int) -> int:
        return BASE_UPGRADE_COST[level]

    def weapon_cost(self, weapon_type: SuperWeaponType) -> int:
        return SUPER_WEAPON_STATS[weapon_type].cost

    def nearest_ant_distance(self, player: int) -> int:
        base_x, base_y = PLAYER_BASES[player]
        enemies = [hex_distance(ant.x, ant.y, base_x, base_y) for ant in self.ants if ant.player != player and ant.is_alive()]
        return min(enemies) if enemies else 32

    def frontline_distance(self, player: int) -> int:
        base_x, base_y = PLAYER_BASES[1 - player]
        ants = [hex_distance(ant.x, ant.y, base_x, base_y) for ant in self.ants if ant.player == player and ant.is_alive()]
        return min(ants) if ants else 32

    def safe_coin_threshold(self, player: int) -> int:
        enemy = 1 - player
        emp_stats = SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER]
        emp_cd = int(self.weapon_cooldowns[enemy, SuperWeaponType.EMP_BLASTER])
        enemy_coin = int(self.coins[enemy])
        capped_cost = max(emp_stats.cost - 1, 0)
        if emp_cd >= emp_stats.cooldown - 10:
            return 0
        if emp_cd > 0:
            return max(int(min(enemy_coin, capped_cost) - emp_cd * 1.66), 0)
        return min(enemy_coin, capped_cost)

    def current_and_neighbors_empty(self, x: int, y: int) -> bool:
        if (x, y) in PLAYER_BASES:
            return False
        if self.tower_at(x, y) is not None:
            return False
        for _, nx, ny in neighbors(x, y):
            if is_valid_pos(nx, ny) and ((nx, ny) in PLAYER_BASES or self.tower_at(nx, ny) is not None):
                return False
        return True

    def is_shielded_by_emp(self, player: int, x: int, y: int) -> bool:
        for effect in self.active_effects:
            if effect.weapon_type == SuperWeaponType.EMP_BLASTER and effect.player != player and effect.in_range(x, y):
                return True
        return False

    def is_shielded_by_deflector(self, ant: Ant) -> bool:
        for effect in self.active_effects:
            if effect.weapon_type == SuperWeaponType.DEFLECTOR and effect.player == ant.player and effect.in_range(ant.x, ant.y):
                return True
        return False

    def weapon_effect(self, weapon_type: SuperWeaponType, player: int) -> WeaponEffect | None:
        for effect in self.active_effects:
            if effect.weapon_type == weapon_type and effect.player == player:
                return effect
        return None

    def _enemy_tower_at(self, player: int, x: int, y: int) -> Tower | None:
        tower = self.tower_at(x, y)
        if tower is None or tower.player == player:
            return None
        return tower

    def _move_progress_score(self, ant: Ant, x: int, y: int, target_x: int, target_y: int) -> float:
        current_distance = hex_distance(ant.x, ant.y, target_x, target_y)
        next_distance = hex_distance(x, y, target_x, target_y)
        score = float(current_distance - next_distance)
        if next_distance == current_distance:
            score -= STALL_MOVE_PENALTY
        elif next_distance > current_distance:
            score -= RETREAT_MOVE_PENALTY * float(next_distance - current_distance)
        base_distance = hex_distance(*PLAYER_BASES[0], *PLAYER_BASES[1])
        score += max(0.0, float(base_distance - next_distance)) * TARGET_PULL_DISTANCE_SCALE
        return score

    def _move_pheromone_score(self, ant: Ant, x: int, y: int) -> float:
        return float(self.pheromone[ant.player, x, y]) / 10000.0

    def _mark_risk_fields_dirty(self) -> None:
        self.risk_fields_dirty = True
        self._invalidate_enhanced_move_cache()

    def _refresh_static_risk_fields(self) -> None:
        if not self.risk_fields_dirty:
            return
        self.damage_risk_field.fill(0.0)
        self.control_risk_field.fill(0.0)
        self.effect_pull_field.fill(0.0)
        for tower in self.towers:
            if tower.is_producer:
                continue
            threatened_player = 1 - tower.player
            damage_value = tower.damage / DAMAGE_FIELD_HP_REFERENCE
            control_value = 0.0
            if tower.tower_type == TowerType.ICE:
                control_value = 1.0
            elif tower.tower_type == TowerType.BEWITCH:
                control_value = 1.3
            elif tower.tower_type == TowerType.PULSE:
                control_value = 0.7
            for x, y in WALKABLE_CELLS:
                if hex_distance(x, y, tower.x, tower.y) > tower.attack_range:
                    continue
                self.damage_risk_field[threatened_player, x, y] += damage_value
                if control_value > 0.0:
                    self.control_risk_field[threatened_player, x, y] += control_value
        storm_damage = LIGHTNING_STORM_ANT_DAMAGE / DAMAGE_FIELD_HP_REFERENCE
        for effect in self.active_effects:
            if effect.weapon_type == SuperWeaponType.LIGHTNING_STORM:
                threatened_player = 1 - effect.player
                for x, y in WALKABLE_CELLS:
                    if effect.in_range(x, y):
                        self.damage_risk_field[threatened_player, x, y] += storm_damage
                continue
            if effect.weapon_type == SuperWeaponType.DEFLECTOR:
                attraction = DEFLECTOR_PATH_ATTRACTION
            elif effect.weapon_type == SuperWeaponType.EMERGENCY_EVASION:
                attraction = EMERGENCY_EVASION_PATH_ATTRACTION
            else:
                continue
            for x, y in WALKABLE_CELLS:
                if effect.in_range(x, y):
                    self.effect_pull_field[effect.player, x, y] += attraction
        self.risk_fields_dirty = False

    def _invalidate_enhanced_move_cache(self) -> None:
        self.enhanced_move_cache_dirty = True
        if not self.enhanced_move_phase_active:
            self.enhanced_move_annotations.clear()

    def _begin_move_phase(self) -> None:
        if self.movement_policy != MOVEMENT_POLICY_ENHANCED:
            return
        self.enhanced_move_phase_active = True
        self._prepare_enhanced_move_cache(reset_reservations=True)

    def _end_move_phase(self) -> None:
        if self.movement_policy != MOVEMENT_POLICY_ENHANCED:
            return
        self.enhanced_move_phase_active = False
        self.enhanced_move_annotations.clear()
        self._invalidate_enhanced_move_cache()

    def _cell_damage_hp(self, player: int, x: int, y: int) -> float:
        return float(self.damage_risk_field[player, x, y]) * DAMAGE_FIELD_HP_REFERENCE

    def _compute_enhanced_traffic_field(self) -> np.ndarray:
        field = np.zeros((PLAYER_COUNT, MAP_SIZE, MAP_SIZE), dtype=np.float32)
        for ant in self.ants:
            if not ant.is_alive():
                continue
            field[ant.player, ant.x, ant.y] += 1.0
            for _, nx, ny in neighbors(ant.x, ant.y):
                if _is_ant_walkable_cell(nx, ny):
                    field[ant.player, nx, ny] += 0.35
        return field

    def _reverse_weighted_plan(
        self,
        player: int,
        sources: list[tuple[int, int]],
        *,
        damage_weight: float,
        control_weight: float,
        traffic_weight: float,
        effect_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        total = np.full((MAP_SIZE, MAP_SIZE), np.inf, dtype=np.float32)
        damage = np.full((MAP_SIZE, MAP_SIZE), np.inf, dtype=np.float32)
        heap: list[tuple[float, float, int, int]] = []
        for x, y in sources:
            if not _is_ant_walkable_cell(x, y):
                continue
            if float(total[x, y]) <= 0.0:
                continue
            total[x, y] = 0.0
            damage[x, y] = 0.0
            heappush(heap, (0.0, 0.0, x, y))

        while heap:
            current_total, current_damage, x, y = heappop(heap)
            best_total = float(total[x, y])
            best_damage = float(damage[x, y])
            if current_total > best_total + 1e-6:
                continue
            if abs(current_total - best_total) <= 1e-6 and current_damage > best_damage + 1e-6:
                continue

            step_damage = self._cell_damage_hp(player, x, y)
            step_control = float(self.control_risk_field[player, x, y])
            step_traffic = float(self.enhanced_traffic_field[player, x, y])
            step_effect = float(self.effect_pull_field[player, x, y])
            step_total = max(
                MIN_PATH_STEP_COST,
                1.0
                + damage_weight * step_damage
                + control_weight * step_control
                + traffic_weight * step_traffic
                - effect_weight * step_effect,
            )

            for _, px, py in neighbors(x, y):
                if not _is_ant_walkable_cell(px, py):
                    continue
                next_total = current_total + step_total
                next_damage = current_damage + step_damage
                known_total = float(total[px, py])
                known_damage = float(damage[px, py])
                if next_total + 1e-6 < known_total or (
                    abs(next_total - known_total) <= 1e-6 and next_damage + 1e-6 < known_damage
                ):
                    total[px, py] = next_total
                    damage[px, py] = next_damage
                    heappush(heap, (next_total, next_damage, px, py))
        return total, damage

    def _prepare_enhanced_move_cache(self, *, reset_reservations: bool) -> None:
        self._refresh_static_risk_fields()
        self.enhanced_traffic_field = self._compute_enhanced_traffic_field()

        for player in range(PLAYER_COUNT):
            worker_total, _ = self._reverse_weighted_plan(
                player,
                [PLAYER_BASES[1 - player]],
                damage_weight=WORKER_PATH_DAMAGE_WEIGHT,
                control_weight=WORKER_PATH_CONTROL_WEIGHT,
                traffic_weight=WORKER_PATH_TRAFFIC_WEIGHT,
                effect_weight=WORKER_PATH_EFFECT_WEIGHT,
            )
            combat_total, _ = self._reverse_weighted_plan(
                player,
                [PLAYER_BASES[1 - player]],
                damage_weight=COMBAT_PATH_DAMAGE_WEIGHT,
                control_weight=COMBAT_PATH_CONTROL_WEIGHT,
                traffic_weight=COMBAT_PATH_TRAFFIC_WEIGHT,
                effect_weight=COMBAT_PATH_EFFECT_WEIGHT,
            )
            self.enhanced_worker_costs[player] = worker_total
            self.enhanced_combat_base_costs[player] = combat_total

            plans: dict[int, EnhancedTowerPlan] = {}
            for tower in self.towers:
                if tower.player == player:
                    continue
                sources = [(nx, ny) for _, nx, ny in neighbors(tower.x, tower.y) if _is_ant_walkable_cell(nx, ny)]
                if not sources:
                    continue
                total_cost, damage_cost = self._reverse_weighted_plan(
                    player,
                    sources,
                    damage_weight=COMBAT_PATH_DAMAGE_WEIGHT,
                    control_weight=COMBAT_PATH_CONTROL_WEIGHT,
                    traffic_weight=COMBAT_PATH_TRAFFIC_WEIGHT,
                    effect_weight=COMBAT_PATH_EFFECT_WEIGHT,
                )
                plans[tower.tower_id] = EnhancedTowerPlan(total_cost=total_cost, damage_cost=damage_cost)
            self.enhanced_tower_plans[player] = plans

        if reset_reservations:
            self.enhanced_reservations.fill(0.0)
            self.enhanced_tower_claims = [dict(), dict()]
        self.enhanced_move_annotations.clear()
        self.enhanced_move_cache_dirty = False

    def _ensure_enhanced_move_cache(self) -> None:
        if self.movement_policy != MOVEMENT_POLICY_ENHANCED:
            return
        if self.enhanced_move_phase_active:
            if self.enhanced_move_cache_dirty:
                self._prepare_enhanced_move_cache(reset_reservations=False)
            return
        self._prepare_enhanced_move_cache(reset_reservations=True)

    def _tower_attack_value(self, ant: Ant, tower: Tower, arrival_hp: float) -> float:
        if arrival_hp <= 0:
            return -1e9
        if ant.kind == AntKind.COMBAT and arrival_hp * 2 < ant.max_hp:
            total_damage = 0.0
            destroyed = 0
            for other in self.towers:
                if other.player == ant.player:
                    continue
                if hex_distance(other.x, other.y, tower.x, tower.y) > COMBAT_SELF_DESTRUCT_RANGE:
                    continue
                total_damage += float(min(COMBAT_SELF_DESTRUCT_DAMAGE, other.hp))
                if other.hp <= COMBAT_SELF_DESTRUCT_DAMAGE:
                    destroyed += 1
            return total_damage + destroyed * ATTACK_FINISH_BONUS + SURPLUS_HP_VALUE_WEIGHT * arrival_hp
        direct_damage = float(min(ant.tower_attack_damage, tower.hp))
        destroy_bonus = ATTACK_FINISH_BONUS if tower.hp <= ant.tower_attack_damage else 0.0
        return direct_damage + destroy_bonus + SURPLUS_HP_VALUE_WEIGHT * arrival_hp

    def _record_enhanced_reservation(self, ant: Ant, direction: int) -> None:
        if self.movement_policy != MOVEMENT_POLICY_ENHANCED or not self.enhanced_move_phase_active:
            return
        annotation = self.enhanced_move_annotations.pop(ant.ant_id, None)
        if annotation is None:
            return
        if annotation.next_cell is not None and direction != NO_MOVE:
            x, y = annotation.next_cell
            self.enhanced_reservations[ant.player, x, y] += 1.0
        if annotation.tower_id is not None:
            claims = self.enhanced_tower_claims[ant.player]
            claims[annotation.tower_id] = claims.get(annotation.tower_id, 0) + 1

    def _directional_field_scores(
        self,
        ant: Ant,
        candidates: list[tuple[int, int, int]],
        field: np.ndarray,
    ) -> list[float]:
        self._refresh_static_risk_fields()
        scores = [0.0] * len(candidates)
        owner = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int16)
        distance_map = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int16)
        queue: deque[tuple[int, int]] = deque()
        seeded: set[int] = set()
        current_value = float(field[ant.player, ant.x, ant.y])

        for index, (_, nx, ny) in enumerate(candidates):
            if self._enemy_tower_at(ant.player, nx, ny) is not None or not _is_ant_walkable_cell(nx, ny):
                scores[index] = current_value
                continue
            if owner[nx, ny] != -1:
                continue
            owner[nx, ny] = index
            distance_map[nx, ny] = 0
            queue.append((nx, ny))
            seeded.add(index)

        while queue:
            x, y = queue.popleft()
            owner_index = int(owner[x, y])
            next_distance = int(distance_map[x, y]) + 1
            for _, nx, ny in neighbors(x, y):
                if not _is_ant_walkable_cell(nx, ny):
                    continue
                if owner[nx, ny] != -1:
                    continue
                owner[nx, ny] = owner_index
                distance_map[nx, ny] = next_distance
                queue.append((nx, ny))

        numerators = [0.0] * len(candidates)
        denominators = [0.0] * len(candidates)
        decay = COMBAT_RISK_FIELD_DISTANCE_DECAY if ant.kind == AntKind.COMBAT else WORKER_RISK_FIELD_DISTANCE_DECAY
        for x, y in WALKABLE_CELLS:
            owner_index = int(owner[x, y])
            if owner_index < 0:
                continue
            weight = decay ** int(distance_map[x, y])
            numerators[owner_index] += float(field[ant.player, x, y]) * weight
            denominators[owner_index] += weight

        for index, (_, nx, ny) in enumerate(candidates):
            if index not in seeded:
                continue
            if denominators[index] > 0.0:
                scores[index] = numerators[index] / denominators[index]
            else:
                scores[index] = float(field[ant.player, nx, ny])
        return scores

    def _tower_pull_score(self, ant: Ant, x: int, y: int, tower_target: Tower | None = None) -> float:
        if tower_target is not None:
            bonus = COMBAT_TOWER_TARGET_BONUS if ant.kind == AntKind.COMBAT else WORKER_TOWER_TARGET_BONUS
            if ant.should_self_destruct_on_tower_attack:
                bonus += COMBAT_SELF_DESTRUCT_PULL_BONUS
            return bonus
        if ant.kind != AntKind.COMBAT:
            return 0.0
        best = 0.0
        self_destruct_bonus = COMBAT_SELF_DESTRUCT_PULL_BONUS if ant.should_self_destruct_on_tower_attack else 0.0
        for tower in self.towers:
            if tower.player == ant.player:
                continue
            distance_score = max(0.0, COMBAT_TOWER_APPROACH_PULL_BASE - float(hex_distance(x, y, tower.x, tower.y)))
            best = max(best, distance_score + self_destruct_bonus)
        return best

    def _move_target_for_ant(self, ant: Ant) -> tuple[int, int]:
        if ant.kind != AntKind.COMBAT:
            return PLAYER_BASES[1 - ant.player]
        enemy_base = PLAYER_BASES[1 - ant.player]
        enemy_towers = [tower for tower in self.towers if tower.player != ant.player]
        if not enemy_towers:
            return enemy_base
        target = min(
            enemy_towers,
            key=lambda tower: (
                hex_distance(ant.x, ant.y, tower.x, tower.y),
                hex_distance(tower.x, tower.y, *enemy_base),
                tower.tower_id,
            ),
        )
        return target.x, target.y

    def _compose_move_score(
        self,
        ant: Ant,
        x: int,
        y: int,
        *,
        target_x: int,
        target_y: int,
        damage_cost: float,
        control_cost: float,
        effect_pull: float,
        tower_target: Tower | None = None,
    ) -> tuple[float, float]:
        weights = ant.move_weights
        progress = self._move_progress_score(ant, x, y, target_x, target_y)
        pheromone = self._move_pheromone_score(ant, x, y)
        crowd = self._crowding_penalty(ant, x, y)
        tower_pull = self._tower_pull_score(ant, x, y, tower_target)
        raw = progress + pheromone + tower_pull + effect_pull
        total = (
            weights.progress * progress
            + weights.pheromone * pheromone
            - weights.crowding * crowd
            - weights.expected_damage * damage_cost
            - weights.control_risk * control_cost
            + weights.tower_pull * tower_pull
            + weights.effect_pull * effect_pull
        )
        return total, raw

    def _spawn_cells_for_tower(self, tower: Tower) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for _, nx, ny in neighbors(tower.x, tower.y):
            if is_path(nx, ny):
                cells.append((nx, ny))
        return cells

    def _initialize_spawned_ant(self, ant: Ant, behavior: AntBehavior) -> None:
        ant.set_behavior(behavior)
        if ant.kind == AntKind.COMBAT:
            ant.grant_evasion(COMBAT_INITIAL_EVASION, grant_control_free_on_deplete=True)
        self._invalidate_enhanced_move_cache()

    def _spawn_ant_from_tower(self, tower: Tower, kind: AntKind, behavior: AntBehavior) -> None:
        cells = self._spawn_cells_for_tower(tower)
        if not cells:
            return
        enemy_base = PLAYER_BASES[1 - tower.player]
        best_x, best_y = max(
            cells,
            key=lambda cell: (
                -hex_distance(cell[0], cell[1], *enemy_base),
                -self._crowding_penalty(
                    Ant(
                        ant_id=-1,
                        player=tower.player,
                        x=cell[0],
                        y=cell[1],
                        hp=1,
                        level=self.bases[tower.player].ant_level,
                        kind=kind,
                    ),
                    cell[0],
                    cell[1],
                ),
            ),
        )
        ant = self.bases[tower.player].spawn_ant(self.next_ant_id, kind=kind)
        ant.x = best_x
        ant.y = best_y
        ant.trail_cells = [(best_x, best_y)]
        self._initialize_spawned_ant(ant, behavior)
        self.ants.append(ant)
        self.next_ant_id += 1

    def _support_frontline_ant(self, tower: Tower) -> None:
        if tower.stats.support_interval <= 0:
            return
        enemy_base = PLAYER_BASES[1 - tower.player]
        candidates = [
            ant
            for ant in self.ants
            if ant.player == tower.player
            and ant.is_alive()
        ]
        if not candidates:
            return
        frontline_distance = min(hex_distance(ant.x, ant.y, *enemy_base) for ant in candidates)
        candidates = [
            ant for ant in candidates if hex_distance(ant.x, ant.y, *enemy_base) <= frontline_distance + 1
        ]
        target = min(
            candidates,
            key=lambda ant: (
                ant.kind != AntKind.COMBAT,
                ant.hp,
                hex_distance(ant.x, ant.y, *enemy_base),
                ant.ant_id,
            ),
        )
        target.hp = target.max_hp
        target.add_evasion(1, grant_control_free_on_deplete=True)
        target.refresh_status()

    def _remove_tower(self, tower_id: int) -> None:
        self.towers = [tower for tower in self.towers if tower.tower_id != tower_id]
        self._mark_risk_fields_dirty()

    def _ant_in_own_half(self, ant: Ant) -> bool:
        return _half_plane_delta(ant.player, ant.x, ant.y) <= 0

    def _random_bewitch_target(self, ant: Ant) -> tuple[int, int]:
        anchor_delta = _half_plane_delta(ant.player, ant.x, ant.y)
        cells = [
            cell
            for cell in _bewitch_cells(ant.player, anchor_delta)
            if cell != (ant.x, ant.y)
        ]
        if not cells:
            return PLAYER_BASES[ant.player]
        return cells[self._random_index(len(cells))]

    def _control_ant(self, ant: Ant, behavior: AntBehavior, *, target: tuple[int, int] | None = None) -> None:
        if ant.control_immune or not ant.is_alive():
            return
        ant.set_behavior(behavior, target=target)

    def _maybe_control_free(self, ant: Ant, *, was_active: bool, is_active: bool) -> None:
        if was_active and not is_active and ant.behavior != AntBehavior.CONTROL_FREE:
            ant.set_behavior(AntBehavior.CONTROL_FREE)

    def _operation_income(self, player: int, operation: Operation, tower_count_hint: int | None = None) -> int:
        if operation.op_type == OperationType.BUILD_TOWER:
            return -self.build_tower_cost(self.tower_count(player) if tower_count_hint is None else tower_count_hint)
        if operation.op_type == OperationType.UPGRADE_TOWER:
            return -self.upgrade_tower_cost(TowerType(operation.arg1))
        if operation.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            if tower is None:
                return 0
            if tower.tower_type == TowerType.BASIC:
                count = self.tower_count(player) if tower_count_hint is None else tower_count_hint
                return self.destroy_tower_income(count, tower)
            return self.downgrade_tower_income(tower.tower_type, tower)
        if operation.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            return -self.weapon_cost(SuperWeaponType(operation.op_type % 10))
        if operation.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            level = self.bases[player].generation_level
            return -self.upgrade_base_cost(level) if level < len(BASE_UPGRADE_COST) else 0
        if operation.op_type == OperationType.UPGRADE_GENERATED_ANT:
            level = self.bases[player].ant_level
            return -self.upgrade_base_cost(level) if level < len(BASE_UPGRADE_COST) else 0
        return 0

    def operation_income(self, player: int, operation: Operation, tower_count_hint: int | None = None) -> int:
        return self._operation_income(player, operation, tower_count_hint)

    def can_apply_operation(self, player: int, operation: Operation, pending: Iterable[Operation] = ()) -> bool:
        pending_list = list(pending)
        if operation.op_type == OperationType.BUILD_TOWER:
            if not is_highland(player, operation.arg0, operation.arg1):
                return False
            if (operation.arg0, operation.arg1) in PLAYER_BASES or self.tower_at(operation.arg0, operation.arg1) is not None:
                return False
            if self.is_shielded_by_emp(player, operation.arg0, operation.arg1):
                return False
            if any(op.op_type == OperationType.BUILD_TOWER and op.arg0 == operation.arg0 and op.arg1 == operation.arg1 for op in pending_list):
                return False
        elif operation.op_type == OperationType.UPGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            if tower is None or tower.player != player:
                return False
            if self.is_shielded_by_emp(player, tower.x, tower.y):
                return False
            if not tower.is_upgrade_type_valid(TowerType(operation.arg1)):
                return False
            if any(op.op_type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER) and op.arg0 == operation.arg0 for op in pending_list):
                return False
        elif operation.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            if tower is None or tower.player != player:
                return False
            if self.is_shielded_by_emp(player, tower.x, tower.y):
                return False
            if any(op.op_type in (OperationType.UPGRADE_TOWER, OperationType.DOWNGRADE_TOWER) and op.arg0 == operation.arg0 for op in pending_list):
                return False
        elif operation.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            if not is_valid_pos(operation.arg0, operation.arg1):
                return False
            weapon_type = SuperWeaponType(operation.op_type % 10)
            if self.weapon_cooldowns[player, weapon_type] > 0:
                return False
            if any(op.op_type == operation.op_type for op in pending_list):
                return False
        elif operation.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            if self.bases[player].generation_level >= 2:
                return False
            if any(op.op_type in (OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT) for op in pending_list):
                return False
        elif operation.op_type == OperationType.UPGRADE_GENERATED_ANT:
            if self.bases[player].ant_level >= 2:
                return False
            if any(op.op_type in (OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT) for op in pending_list):
                return False
        else:
            return False

        income = 0
        simulated_tower_count = self.tower_count(player)
        for op in (*pending_list, operation):
            if op.op_type == OperationType.BUILD_TOWER:
                income -= self.build_tower_cost(simulated_tower_count)
                simulated_tower_count += 1
            elif op.op_type == OperationType.DOWNGRADE_TOWER:
                tower = self.tower_by_id(op.arg0)
                if tower is None:
                    continue
                if tower.tower_type == TowerType.BASIC:
                    income += self.destroy_tower_income(simulated_tower_count, tower)
                    simulated_tower_count -= 1
                else:
                    income += self.downgrade_tower_income(tower.tower_type, tower)
            else:
                income += self._operation_income(player, op)
        return self.coins[player] + income >= 0

    def apply_operation(self, player: int, operation: Operation) -> None:
        self.coins[player] += self._operation_income(player, operation)
        if operation.op_type == OperationType.BUILD_TOWER:
            self.towers.append(
                Tower(
                    self.next_tower_id,
                    player,
                    operation.arg0,
                    operation.arg1,
                    TowerType.BASIC,
                    TOWER_STATS[TowerType.BASIC].speed,
                )
            )
            self.next_tower_id += 1
            self._mark_risk_fields_dirty()
            return
        if operation.op_type == OperationType.UPGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            assert tower is not None
            tower.upgrade(TowerType(operation.arg1))
            self._mark_risk_fields_dirty()
            return
        if operation.op_type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            assert tower is not None
            destroy = tower.downgrade_or_destroy()
            if destroy:
                self.towers = [item for item in self.towers if item.tower_id != tower.tower_id]
            self._mark_risk_fields_dirty()
            return
        if operation.op_type in (
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            weapon_type = SuperWeaponType(operation.op_type % 10)
            stats = SUPER_WEAPON_STATS[weapon_type]
            self.weapon_cooldowns[player, weapon_type] = stats.cooldown
            self.super_weapon_usage[player] += 1
            effect = WeaponEffect(weapon_type, player, operation.arg0, operation.arg1, stats.duration)
            self.active_effects.append(effect)
            if weapon_type == SuperWeaponType.EMERGENCY_EVASION:
                for ant in self.ants:
                    if ant.player == player and hex_distance(operation.arg0, operation.arg1, ant.x, ant.y) <= stats.attack_range:
                        ant.grant_evasion(2, grant_control_free_on_deplete=True)
            elif weapon_type == SuperWeaponType.LIGHTNING_STORM:
                self._apply_lightning_effect(effect)
            self._mark_risk_fields_dirty()
            return
        if operation.op_type == OperationType.UPGRADE_GENERATION_SPEED:
            self.bases[player].generation_level += 1
            return
        if operation.op_type == OperationType.UPGRADE_GENERATED_ANT:
            self.bases[player].ant_level += 1
            return

    def apply_operation_list(self, player: int, operations: Iterable[Operation]) -> list[Operation]:
        illegal: list[Operation] = []
        accepted: list[Operation] = []
        for operation in operations:
            if self.can_apply_operation(player, operation, accepted):
                self.apply_operation(player, operation)
                accepted.append(operation)
            else:
                illegal.append(operation)
                if not self.cold_handle_rule_illegal:
                    self.terminal = True
                    self.winner = 1 - player
                    break
        return illegal

    def _prepare_ants_for_attack(self) -> None:
        for ant in self.ants:
            if ant.frozen:
                ant.frozen = False
                if ant.pending_behavior is not None:
                    self._control_ant(ant, ant.pending_behavior)
                    ant.pending_behavior = None
            current_deflector = self.is_shielded_by_deflector(ant)
            current_evasion = any(
                effect.weapon_type == SuperWeaponType.EMERGENCY_EVASION
                and effect.player == ant.player
                and effect.in_range(ant.x, ant.y)
                for effect in self.active_effects
            )
            self._maybe_control_free(ant, was_active=ant.deflector, is_active=current_deflector)
            ant.deflector = current_deflector
            if current_evasion:
                ant.grant_evasion(2, grant_control_free_on_deplete=True)
            ant.evasion = ant.shield > 0
            ant.refresh_status()

    def _apply_lightning_effect(self, effect: WeaponEffect) -> None:
        if effect.weapon_type != SuperWeaponType.LIGHTNING_STORM:
            return
        if effect.last_trigger_round == self.round_index:
            return
        effect.last_trigger_round = self.round_index
        duration = SUPER_WEAPON_STATS[effect.weapon_type].duration
        active_turn = duration - effect.remaining_turns + 1
        for ant in self.ants:
            if ant.player != effect.player and ant.is_alive() and effect.in_range(ant.x, ant.y):
                ant.take_damage(LIGHTNING_STORM_ANT_DAMAGE)
        if active_turn <= 0 or active_turn % LIGHTNING_STORM_TOWER_INTERVAL != 0:
            return
        destroyed_ids: set[int] = set()
        for tower in self.towers:
            if tower.player == effect.player or not effect.in_range(tower.x, tower.y):
                continue
            if tower.take_damage(LIGHTNING_STORM_TOWER_DAMAGE):
                destroyed_ids.add(tower.tower_id)
        if destroyed_ids:
            self.towers = [tower for tower in self.towers if tower.tower_id not in destroyed_ids]
            self._mark_risk_fields_dirty()

    def _apply_lightning_storm(self) -> None:
        for effect in self.active_effects:
            self._apply_lightning_effect(effect)

    def _attack_ants(self) -> None:
        self._prepare_ants_for_attack()
        self._apply_lightning_storm()
        for tower in self.towers:
            if tower.is_producer:
                continue
            if self.is_shielded_by_emp(tower.player, tower.x, tower.y):
                continue
            tower.tick()
            if not tower.ready_to_fire():
                continue
            attacked = self._tower_attack(tower)
            if attacked:
                tower.reset_cooldown()

    def _apply_tower_control(self, tower: Tower, ant: Ant) -> None:
        if not ant.is_alive():
            return
        if tower.tower_type == TowerType.ICE:
            if ant.control_immune:
                return
            ant.frozen = True
            ant.pending_behavior = AntBehavior.RANDOM
            ant.refresh_status()
            return
        if tower.tower_type == TowerType.BEWITCH:
            if self._ant_in_own_half(ant):
                target = PLAYER_BASES[ant.player]
            else:
                target = self._random_bewitch_target(ant)
            self._control_ant(ant, AntBehavior.BEWITCHED, target=target)
            return
        if tower.tower_type == TowerType.PULSE:
            self._control_ant(ant, AntBehavior.RANDOM)

    def _damage_ant_from_tower(self, tower: Tower, ant: Ant) -> None:
        ant.take_damage(tower.damage)
        self._apply_tower_control(tower, ant)

    def _tower_attack(self, tower: Tower) -> bool:
        targets = self._find_targets(tower)
        if not targets:
            return False
        attacked_any = False
        repetitions = int(round(1 / tower.speed)) if tower.speed < 1 else 1
        for _ in range(repetitions):
            local_targets = self._find_targets(tower)
            if not local_targets:
                break
            for ant in self._expand_attack_targets(tower, local_targets):
                self._damage_ant_from_tower(tower, ant)
                attacked_any = True
        return attacked_any

    def _find_targets(self, tower: Tower) -> list[Ant]:
        candidates = [
            ant for ant in self.ants
            if ant.player != tower.player and ant.is_alive() and hex_distance(ant.x, ant.y, tower.x, tower.y) <= tower.attack_range
        ]
        candidates.sort(key=lambda ant: (hex_distance(ant.x, ant.y, tower.x, tower.y), ant.ant_id))
        if tower.tower_type == TowerType.DOUBLE:
            return candidates[:2]
        return candidates[:1]

    def _expand_attack_targets(self, tower: Tower, targets: list[Ant]) -> list[Ant]:
        expanded: list[Ant] = []
        for target in targets:
            if tower.tower_type in (TowerType.MORTAR, TowerType.MORTAR_PLUS):
                expanded.extend(self._ants_in_range(tower.player, target.x, target.y, tower.attack_range))
            elif tower.tower_type == TowerType.PULSE:
                expanded.extend(self._ants_in_range(tower.player, tower.x, tower.y, tower.attack_range))
            elif tower.tower_type == TowerType.MISSILE:
                expanded.extend(self._ants_in_range(tower.player, target.x, target.y, tower.attack_range))
            else:
                expanded.append(target)
        unique: dict[int, Ant] = {}
        for ant in expanded:
            unique[ant.ant_id] = ant
        return list(unique.values())

    def _ants_in_range(self, player: int, x: int, y: int, attack_range: int) -> list[Ant]:
        return [ant for ant in self.ants if ant.player != player and ant.is_alive() and hex_distance(ant.x, ant.y, x, y) <= attack_range]

    def _crowding_penalty(self, ant: Ant, x: int, y: int) -> float:
        penalty = 0.0
        for other in self.ants:
            if other.ant_id == ant.ant_id or other.player != ant.player:
                continue
            if other.status in (AntStatus.FAIL, AntStatus.TOO_OLD):
                continue
            distance = hex_distance(x, y, other.x, other.y)
            if distance == 0:
                penalty += 1.0
            elif distance == 1:
                penalty += 0.35
        return penalty

    def _move_candidates(self, ant: Ant, *, allow_backtrack: bool) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        enemy_base = PLAYER_BASES[1 - ant.player]
        own_base = PLAYER_BASES[ant.player]
        for direction, nx, ny in neighbors(ant.x, ant.y):
            if not allow_backtrack and ant.last_move == (direction + 3) % 6:
                continue
            tower = self._enemy_tower_at(ant.player, nx, ny)
            if tower is None and (nx, ny) not in (enemy_base, own_base) and not is_path(nx, ny):
                continue
            if not is_valid_pos(nx, ny):
                continue
            out.append((direction, nx, ny))
        return out

    def _legal_move_candidates(self, ant: Ant) -> list[tuple[int, int, int]]:
        allow_backtrack = ant.behavior in {AntBehavior.RANDOM, AntBehavior.BEWITCHED}
        candidates = self._move_candidates(ant, allow_backtrack=allow_backtrack)
        if not candidates and not allow_backtrack:
            candidates = self._move_candidates(ant, allow_backtrack=True)
        return candidates

    def _choose_random_legal_move(self, ant: Ant) -> int:
        candidates = self._legal_move_candidates(ant)
        if not candidates:
            return NO_MOVE
        return candidates[self._random_index(len(candidates))][0]

    def _sample_move_from_scores(
        self,
        candidates: list[tuple[int, int, int]],
        scores: list[float],
        temperature: float,
    ) -> int:
        if not candidates:
            return -1
        probabilities = _softmax_choice(scores, temperature)
        return candidates[self._sample_index(probabilities)][0]

    def _choose_ant_move_legacy(self, ant: Ant) -> int:
        target_x, target_y = self._move_target_for_ant(ant)
        candidates = self._legal_move_candidates(ant)
        if not candidates:
            return -1

        if ant.behavior == AntBehavior.RANDOM:
            return candidates[self._random_index(len(candidates))][0]

        damage_scores = self._directional_field_scores(ant, candidates, self.damage_risk_field)
        control_scores = self._directional_field_scores(ant, candidates, self.control_risk_field)
        effect_scores = self._directional_field_scores(ant, candidates, self.effect_pull_field)
        if ant.control_immune:
            control_scores = [0.0] * len(control_scores)

        if ant.behavior == AntBehavior.BEWITCHED and ant.bewitch_target_x >= 0 and ant.bewitch_target_y >= 0:
            scores = []
            for index, (_, nx, ny) in enumerate(candidates):
                tower_target = self._enemy_tower_at(ant.player, nx, ny)
                eval_x, eval_y = (ant.x, ant.y) if tower_target is not None else (nx, ny)
                score, _ = self._compose_move_score(
                    ant,
                    eval_x,
                    eval_y,
                    target_x=ant.bewitch_target_x,
                    target_y=ant.bewitch_target_y,
                    damage_cost=damage_scores[index],
                    control_cost=control_scores[index],
                    effect_pull=effect_scores[index],
                    tower_target=tower_target,
                )
                scores.append(score + (4.0 if tower_target is not None else 0.0))
            return self._sample_move_from_scores(candidates, scores, BEWITCH_MOVE_TEMPERATURE)

        weighted_scores: list[float] = []
        raw_scores: list[float] = []
        for index, (_, nx, ny) in enumerate(candidates):
            tower_target = self._enemy_tower_at(ant.player, nx, ny)
            eval_x, eval_y = (ant.x, ant.y) if tower_target is not None else (nx, ny)
            score, raw = self._compose_move_score(
                ant,
                eval_x,
                eval_y,
                target_x=target_x,
                target_y=target_y,
                damage_cost=damage_scores[index],
                control_cost=control_scores[index],
                effect_pull=effect_scores[index],
                tower_target=tower_target,
            )
            weighted_scores.append(score)
            raw_scores.append(raw)

        if ant.behavior in (AntBehavior.CONSERVATIVE, AntBehavior.CONTROL_FREE):
            best_index = max(range(len(candidates)), key=lambda index: (weighted_scores[index], raw_scores[index], -index))
            return candidates[best_index][0]
        return self._sample_move_from_scores(candidates, weighted_scores, DEFAULT_MOVE_TEMPERATURE)

    def _choose_worker_move_enhanced(self, ant: Ant, candidates: list[tuple[int, int, int]]) -> int:
        self._ensure_enhanced_move_cache()
        current_cost = float(self.enhanced_worker_costs[ant.player, ant.x, ant.y])
        walk_remaining: list[float] = [
            float(self.enhanced_worker_costs[ant.player, nx, ny])
            for _, nx, ny in candidates
            if self._enemy_tower_at(ant.player, nx, ny) is None
        ]
        best_walk_remaining = min(walk_remaining) if walk_remaining else np.inf
        reroute_gain = 0.0
        if np.isfinite(current_cost) and np.isfinite(best_walk_remaining):
            reroute_gain = max(0.0, current_cost - best_walk_remaining)
        blocked = not np.isfinite(best_walk_remaining) or not np.isfinite(current_cost) or (
            current_cost - best_walk_remaining <= WORKER_ROUTE_IMPROVEMENT_EPS
        )

        weighted_scores: list[float] = []
        raw_scores: list[float] = []
        annotations: list[EnhancedMoveAnnotation] = []
        for direction, nx, ny in candidates:
            tower_target = self._enemy_tower_at(ant.player, nx, ny)
            if tower_target is not None:
                score = 0.0 if not np.isfinite(current_cost) else -current_cost
                score += 1.2 * float(min(ant.tower_attack_damage, tower_target.hp))
                if tower_target.hp <= ant.tower_attack_damage:
                    score += ATTACK_FINISH_BONUS
                if blocked:
                    score += WORKER_BLOCKED_ATTACK_BONUS
                else:
                    # If there is still a materially better route toward the base, keep workers moving.
                    score -= WORKER_REROUTE_ATTACK_PENALTY_WEIGHT * reroute_gain
                score -= WORKER_TOWER_CLAIM_WEIGHT * self.enhanced_tower_claims[ant.player].get(tower_target.tower_id, 0)
                score += ant.move_weights.pheromone * self._move_pheromone_score(ant, ant.x, ant.y)
                weighted_scores.append(score)
                raw_scores.append(score)
                annotations.append(EnhancedMoveAnnotation(tower_id=tower_target.tower_id))
                continue

            remaining = float(self.enhanced_worker_costs[ant.player, nx, ny])
            if not np.isfinite(remaining):
                score = -1e9
            else:
                score = -remaining
                score -= WORKER_RESERVATION_WEIGHT * float(self.enhanced_reservations[ant.player, nx, ny])
                score -= 0.25 * self._crowding_penalty(ant, nx, ny)
                score += ant.move_weights.pheromone * self._move_pheromone_score(ant, nx, ny)
            weighted_scores.append(score)
            raw_scores.append(score)
            annotations.append(EnhancedMoveAnnotation(next_cell=(nx, ny)))

        if ant.behavior in (AntBehavior.CONSERVATIVE, AntBehavior.CONTROL_FREE):
            best_index = max(range(len(candidates)), key=lambda index: (weighted_scores[index], raw_scores[index], -index))
            self.enhanced_move_annotations[ant.ant_id] = annotations[best_index]
            return candidates[best_index][0]
        chosen = self._sample_move_from_scores(candidates, weighted_scores, DEFAULT_MOVE_TEMPERATURE)
        chosen_index = next(index for index, (direction, _, _) in enumerate(candidates) if direction == chosen)
        self.enhanced_move_annotations[ant.ant_id] = annotations[chosen_index]
        return chosen

    def _choose_combat_move_enhanced(self, ant: Ant, candidates: list[tuple[int, int, int]]) -> int:
        self._ensure_enhanced_move_cache()
        enemy_towers = [tower for tower in self.towers if tower.player != ant.player]
        weighted_scores: list[float] = []
        raw_scores: list[float] = []
        annotations: list[EnhancedMoveAnnotation] = []

        for direction, nx, ny in candidates:
            tower_target = self._enemy_tower_at(ant.player, nx, ny)
            if tower_target is not None:
                score = self._tower_attack_value(ant, tower_target, float(ant.hp))
                # Distinguish attacking now from merely rotating to another adjacent tower cell.
                score += ENHANCED_COMBAT_ATTACK_EXECUTION_BONUS
                score -= COMBAT_TOWER_CLAIM_WEIGHT * self.enhanced_tower_claims[ant.player].get(tower_target.tower_id, 0)
                score += ant.move_weights.pheromone * self._move_pheromone_score(ant, ant.x, ant.y)
                weighted_scores.append(score)
                raw_scores.append(score)
                annotations.append(EnhancedMoveAnnotation(tower_id=tower_target.tower_id))
                continue

            best_score = -1e9
            best_tower_id: int | None = None
            if enemy_towers:
                for tower in enemy_towers:
                    plan = self.enhanced_tower_plans[ant.player].get(tower.tower_id)
                    if plan is None:
                        continue
                    travel_cost = float(plan.total_cost[nx, ny])
                    if not np.isfinite(travel_cost):
                        continue
                    travel_damage = float(plan.damage_cost[nx, ny])
                    arrival_hp = float(ant.hp) - travel_damage
                    utility = self._tower_attack_value(ant, tower, arrival_hp)
                    utility -= COMBAT_TRAVEL_COST_WEIGHT * travel_cost
                    utility -= COMBAT_TOWER_CLAIM_WEIGHT * self.enhanced_tower_claims[ant.player].get(tower.tower_id, 0)
                    if utility > best_score:
                        best_score = utility
                        best_tower_id = tower.tower_id
            else:
                remaining = float(self.enhanced_combat_base_costs[ant.player, nx, ny])
                if np.isfinite(remaining):
                    best_score = -remaining

            if best_tower_id is None and not enemy_towers:
                base_score = best_score
            else:
                base_score = best_score
            if np.isfinite(base_score):
                base_score -= COMBAT_RESERVATION_WEIGHT * float(self.enhanced_reservations[ant.player, nx, ny])
                base_score += ant.move_weights.pheromone * self._move_pheromone_score(ant, nx, ny)
            weighted_scores.append(base_score)
            raw_scores.append(base_score)
            annotations.append(EnhancedMoveAnnotation(next_cell=(nx, ny), tower_id=best_tower_id))

        if ant.behavior in (AntBehavior.CONSERVATIVE, AntBehavior.CONTROL_FREE):
            best_index = max(range(len(candidates)), key=lambda index: (weighted_scores[index], raw_scores[index], -index))
            self.enhanced_move_annotations[ant.ant_id] = annotations[best_index]
            return candidates[best_index][0]
        chosen = self._sample_move_from_scores(candidates, weighted_scores, DEFAULT_MOVE_TEMPERATURE)
        chosen_index = next(index for index, (direction, _, _) in enumerate(candidates) if direction == chosen)
        self.enhanced_move_annotations[ant.ant_id] = annotations[chosen_index]
        return chosen

    def _choose_ant_move_enhanced(self, ant: Ant) -> int:
        candidates = self._legal_move_candidates(ant)
        if not candidates:
            return NO_MOVE
        if ant.behavior == AntBehavior.RANDOM:
            return candidates[self._random_index(len(candidates))][0]
        if ant.behavior == AntBehavior.BEWITCHED:
            return self._choose_ant_move_legacy(ant)
        if ant.kind == AntKind.COMBAT:
            return self._choose_combat_move_enhanced(ant, candidates)
        return self._choose_worker_move_enhanced(ant, candidates)

    def _choose_ant_move(self, ant: Ant) -> int:
        if self.movement_policy == MOVEMENT_POLICY_LEGACY:
            return self._choose_ant_move_legacy(ant)
        return self._choose_ant_move_enhanced(ant)

    def _attack_tower_from_ant(self, ant: Ant, tower: Tower) -> None:
        if ant.should_self_destruct_on_tower_attack:
            destroyed_ids: set[int] = set()
            for target in self.towers:
                if target.player == ant.player:
                    continue
                if hex_distance(target.x, target.y, tower.x, tower.y) > COMBAT_SELF_DESTRUCT_RANGE:
                    continue
                if target.take_damage(COMBAT_SELF_DESTRUCT_DAMAGE):
                    destroyed_ids.add(target.tower_id)
            if destroyed_ids:
                self.towers = [candidate for candidate in self.towers if candidate.tower_id not in destroyed_ids]
                self._mark_risk_fields_dirty()
            ant.hp = 0
            ant.refresh_status()
            return
        destroyed = tower.take_damage(ant.tower_attack_damage)
        if destroyed:
            self._remove_tower(tower.tower_id)

    def _resolve_ant_step(self, ant: Ant, direction: int) -> None:
        if direction == NO_MOVE:
            ant.record_move(direction)
            ant.refresh_status()
            return
        dx, dy = OFFSET[ant.y % 2][direction]
        tower = self._enemy_tower_at(ant.player, ant.x + dx, ant.y + dy)
        if tower is not None:
            self._attack_tower_from_ant(ant, tower)
            ant.last_move = NO_MOVE
            ant.evasion = ant.shield > 0
            ant.refresh_status()
            return
        ant.record_move(direction)
        ant.refresh_status()

    def _resolve_random_move_steps(self, ant: Ant, *, steps: int = 3) -> None:
        for _ in range(steps):
            ant.refresh_status()
            if ant.status in (AntStatus.FAIL, AntStatus.TOO_OLD):
                break
            if self._random_index(3) < 2:
                direction = self._choose_random_legal_move(ant)
            else:
                direction = self._choose_ant_move(ant)
            self._resolve_ant_step(ant, direction)
            self._invalidate_enhanced_move_cache()

    def _teleport_ants(self) -> None:
        if ANT_TELEPORT_INTERVAL <= 0 or (self.round_index + 1) % ANT_TELEPORT_INTERVAL != 0:
            return
        eligible = [
            ant
            for ant in self.ants
            if ant.status not in (AntStatus.FAIL, AntStatus.TOO_OLD)
            and ant.behavior != AntBehavior.CONTROL_FREE
        ]
        if not eligible:
            return
        teleport_count = max(1, int(round(len(eligible) * ANT_TELEPORT_RATIO)))
        chosen: list[Ant] = []
        pool = list(eligible)
        while pool and len(chosen) < teleport_count:
            chosen.append(pool.pop(self._random_index(len(pool))))
        for ant in chosen:
            self._resolve_random_move_steps(ant)

    def _move_ants(self) -> None:
        self._begin_move_phase()
        for ant in self.ants:
            ant.refresh_status()
            direction = NO_MOVE
            if ant.status == AntStatus.ALIVE:
                direction = self._choose_ant_move(ant)
                self._record_enhanced_reservation(ant, direction)
            self._resolve_ant_step(ant, direction)
        self._end_move_phase()
        self._teleport_ants()

    def _update_pheromone(self) -> None:
        # Global attenuation: p_new = 0.97*p + 0.03*10 (integer arithmetic)
        self.pheromone = np.maximum(
            0,
            (LAMBDA_NUM * self.pheromone + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM,
        )
        for ant in self.ants:
            if ant.status in (AntStatus.ALIVE, AntStatus.FROZEN):
                continue
            if ant.status == AntStatus.SUCCESS:
                delta = PHEROMONE_SUCCESS_BONUS_INT
            elif ant.status == AntStatus.FAIL:
                delta = PHEROMONE_FAIL_BONUS_INT
            elif ant.status == AntStatus.TOO_OLD:
                delta = PHEROMONE_TOO_OLD_BONUS_INT
            else:
                continue
            visited: set[tuple[int, int]] = set()
            for x, y in reversed(_trail_for_pheromone(ant)):
                if not is_valid_pos(x, y):
                    continue
                if (x, y) in visited:
                    continue
                self.pheromone[ant.player, x, y] = max(0, self.pheromone[ant.player, x, y] + delta)
                visited.add((x, y))

    def _judge_base_camps(self) -> bool:
        if self.bases[0].hp <= 0 and self.bases[1].hp <= 0:
            self.terminal = True
            self.winner = 0
            return True
        if self.bases[1].hp <= 0:
            self.terminal = True
            self.winner = 0
            return True
        if self.bases[0].hp <= 0:
            self.terminal = True
            self.winner = 1
            return True
        return False

    def _resolve_ant_lifecycle(self) -> None:
        remaining: list[Ant] = []
        base_destroyed = False
        for index, ant in enumerate(self.ants):
            ant.refresh_status()
            if ant.status == AntStatus.SUCCESS:
                self.bases[1 - ant.player].hp -= 1
                self.coins[ant.player] += ANT_BREACH_REWARD
                if self._judge_base_camps():
                    remaining.extend(self.ants[index + 1 :])
                    base_destroyed = True
                    break
            elif ant.status == AntStatus.FAIL:
                self.coins[1 - ant.player] += ant.kill_reward
                self.die_count[ant.player] += 1
            elif ant.status == AntStatus.TOO_OLD:
                self.old_count[ant.player] += 1
            else:
                remaining.append(ant)
        if not base_destroyed:
            survivors: list[Ant] = []
            for ant in remaining:
                ant.refresh_status()
                if ant.status == AntStatus.TOO_OLD:
                    self.old_count[ant.player] += 1
                    continue
                survivors.append(ant)
            remaining = survivors
        self.ants = remaining

    def _draw_spawn_profile(self) -> tuple[AntKind, AntBehavior]:
        roll = self._random_float()
        cumulative = 0.0
        for kind, behavior, probability in SPAWN_PROFILE_WEIGHTS:
            cumulative += probability
            if roll <= cumulative:
                return kind, behavior
        fallback_kind, fallback_behavior, _ = SPAWN_PROFILE_WEIGHTS[-1]
        return fallback_kind, fallback_behavior

    def _spawn_ants(self) -> None:
        for base in self.bases:
            if base.should_spawn(self.round_index):
                kind, behavior = self._draw_spawn_profile()
                ant = base.spawn_ant(self.next_ant_id, kind=kind)
                self._initialize_spawned_ant(ant, behavior)
                self.ants.append(ant)
                self.next_ant_id += 1
        for tower in self.towers:
            if not tower.is_producer:
                continue
            if self.is_shielded_by_emp(tower.player, tower.x, tower.y):
                continue
            tower.tick()
            if (
                tower.tower_type == TowerType.PRODUCER_MEDIC
                and tower.stats.support_interval > 0
                and int(round(max(tower.cooldown_clock, 0.0))) % tower.stats.support_interval == 0
            ):
                self._support_frontline_ant(tower)
            if not tower.ready_to_fire():
                continue
            kind, behavior = self._draw_spawn_profile()
            self._spawn_ant_from_tower(tower, kind, behavior)
            if tower.tower_type == TowerType.PRODUCER_SIEGE:
                if self._random_float() <= tower.stats.siege_spawn_chance:
                    self._spawn_ant_from_tower(tower, AntKind.COMBAT, AntBehavior.DEFAULT)
            tower.reset_cooldown()

    def _increase_ant_age(self) -> None:
        for ant in self.ants:
            ant.age += 1
            ant.behavior_turns += 1
            if ant.behavior == AntBehavior.RANDOM and ant.behavior_turns >= RANDOM_ANT_DECAY_TURNS:
                ant.set_behavior(AntBehavior.DEFAULT, force=True)
            else:
                if (
                    ant.behavior == AntBehavior.BEWITCHED
                    and ant.bewitch_target_x == ant.x
                    and ant.bewitch_target_y == ant.y
                ):
                    ant.set_behavior(AntBehavior.DEFAULT, force=True)
                elif ant.behavior_expiry > 0:
                    ant.behavior_expiry -= 1
                    if ant.behavior not in (AntBehavior.DEFAULT, AntBehavior.RANDOM) and ant.behavior_expiry <= 0:
                        ant.set_behavior(AntBehavior.DEFAULT, force=True)
            ant.refresh_status()

    def _drift_effect(self, effect: WeaponEffect) -> None:
        if effect.weapon_type not in (SuperWeaponType.LIGHTNING_STORM, SuperWeaponType.EMP_BLASTER):
            return
        candidates = [(effect.x, effect.y)]
        for _, nx, ny in neighbors(effect.x, effect.y):
            if is_valid_pos(nx, ny):
                candidates.append((nx, ny))
        effect.x, effect.y = candidates[self._random_index(len(candidates))]

    def _tick_effects(self) -> None:
        for player in range(PLAYER_COUNT):
            for weapon_index in range(1, 5):
                if self.weapon_cooldowns[player, weapon_index] > 0:
                    self.weapon_cooldowns[player, weapon_index] -= 1
        next_effects: list[WeaponEffect] = []
        for effect in self.active_effects:
            self._drift_effect(effect)
            effect.remaining_turns -= 1
            if effect.remaining_turns > 0 and effect.weapon_type != SuperWeaponType.EMERGENCY_EVASION:
                next_effects.append(effect)
        self.active_effects = next_effects
        self._mark_risk_fields_dirty()

    def _judge_timeout_winner(self) -> None:
        if self.bases[0].hp != self.bases[1].hp:
            self.winner = 0 if self.bases[0].hp > self.bases[1].hp else 1
            return
        if self.die_count[0] != self.die_count[1]:
            self.winner = 0 if self.die_count[0] > self.die_count[1] else 1
            return
        if self.super_weapon_usage[0] != self.super_weapon_usage[1]:
            self.winner = 0 if self.super_weapon_usage[0] < self.super_weapon_usage[1] else 1
            return
        if self.ai_time[0] != self.ai_time[1]:
            self.winner = 0 if self.ai_time[0] < self.ai_time[1] else 1
            return
        self.winner = 0

    def advance_round(self) -> None:
        if self.terminal:
            return
        self._attack_ants()
        self._move_ants()
        self._update_pheromone()
        self._resolve_ant_lifecycle()
        if self.terminal:
            self.round_index += 1
            return
        self._spawn_ants()
        self._increase_ant_age()
        if (self.round_index + 1) % BASIC_INCOME_INTERVAL == 0:
            for player in range(PLAYER_COUNT):
                self.coins[player] += BASIC_INCOME
        self._tick_effects()
        self.round_index += 1
        if self.round_index >= MAX_ROUND and not self.terminal:
            self.terminal = True
            self._judge_timeout_winner()
        if not self.terminal:
            self._judge_base_camps()

    def resolve_turn(self, operations0: Iterable[Operation], operations1: Iterable[Operation]) -> TurnResolution:
        operations0 = list(operations0)
        operations1 = list(operations1)
        illegal0 = self.apply_operation_list(0, operations0)
        illegal1: list[Operation] = []
        if not self.terminal:
            illegal1 = self.apply_operation_list(1, operations1)
        if not self.terminal:
            self.advance_round()
        return TurnResolution((list(operations0), list(operations1)), (illegal0, illegal1), self.terminal, self.winner)

    def to_public_round_state(self) -> PublicRoundState:
        towers = [
            (tower.tower_id, tower.player, tower.x, tower.y, int(tower.tower_type), tower.display_cooldown(), tower.hp)
            for tower in sorted(self.towers, key=lambda item: item.tower_id)
        ]
        ants = [
            (
                ant.ant_id,
                ant.player,
                ant.x,
                ant.y,
                ant.hp,
                ant.level,
                ant.age,
                int(ant.status),
                int(ant.behavior),
                int(ant.kind),
            )
            for ant in sorted(self.ants, key=lambda item: item.ant_id)
        ]
        return PublicRoundState(
            round_index=self.round_index,
            towers=towers,
            ants=ants,
            coins=(self.coins[0], self.coins[1]),
            camps_hp=(self.bases[0].hp, self.bases[1].hp),
            speed_lv=(self.bases[0].generation_level, self.bases[1].generation_level),
            anthp_lv=(self.bases[0].ant_level, self.bases[1].ant_level),
            weapon_cooldowns=tuple(
                tuple(int(self.weapon_cooldowns[player, weapon_type]) for weapon_type in SuperWeaponType)
                for player in range(PLAYER_COUNT)
            ),
            active_effects=[
                (int(effect.weapon_type), effect.player, effect.x, effect.y, effect.remaining_turns)
                for effect in sorted(self.active_effects, key=lambda item: (item.player, int(item.weapon_type), item.x, item.y))
            ],
        )

    def sync_public_round_state(self, public_state: PublicRoundState) -> None:
        self.round_index = public_state.round_index
        self.coins[0], self.coins[1] = public_state.coins
        self.bases[0].hp, self.bases[1].hp = public_state.camps_hp
        if public_state.speed_lv is not None:
            self.bases[0].generation_level, self.bases[1].generation_level = public_state.speed_lv
        if public_state.anthp_lv is not None:
            self.bases[0].ant_level, self.bases[1].ant_level = public_state.anthp_lv
        tower_map = {tower.tower_id: tower for tower in self.towers}
        synced_towers: list[Tower] = []
        for tower_row in public_state.towers:
            tower_id, player, x, y, tower_type, cooldown = tower_row[:6]
            public_hp = tower_row[6] if len(tower_row) >= 7 else -1
            tower = tower_map.get(tower_id, Tower(tower_id, player, x, y, TowerType(tower_type), float(cooldown), hp=int(public_hp)))
            tower.player = player
            tower.x = x
            tower.y = y
            tower.tower_type = TowerType(tower_type)
            tower.cooldown_clock = float(cooldown)
            if public_hp >= 0:
                tower.hp = int(public_hp)
            synced_towers.append(tower)
        self.towers = synced_towers
        self._mark_risk_fields_dirty()
        ant_map = {ant.ant_id: ant for ant in self.ants}
        synced_ants: list[Ant] = []
        for ant_row in public_state.ants:
            ant_id, player, x, y, hp, level, public_age, status = ant_row[:8]
            public_behavior = AntBehavior(ant_row[8]) if len(ant_row) >= 9 else None
            ant = ant_map.get(ant_id, Ant(ant_id, player, x, y, hp, level, age=0, status=AntStatus(status)))
            ant.player = player
            ant.x = x
            ant.y = y
            ant.hp = hp
            ant.level = level
            ant.age = public_age
            ant.status = AntStatus(status)
            ant.frozen = ant.status == AntStatus.FROZEN
            if public_behavior is not None:
                if ant.behavior != public_behavior:
                    ant.behavior_turns = 0
                    ant.behavior_expiry = default_behavior_expiry(public_behavior)
                ant.behavior = public_behavior
                if ant.behavior != AntBehavior.BEWITCHED:
                    ant.bewitch_target_x = -1
                    ant.bewitch_target_y = -1
            if len(ant_row) >= 10:
                ant.set_kind(AntKind(ant_row[9]))
            synced_ants.append(ant)
        self.ants = synced_ants
        if public_state.weapon_cooldowns is not None:
            self.weapon_cooldowns.fill(0)
            for player, row in enumerate(public_state.weapon_cooldowns[:PLAYER_COUNT]):
                for weapon_type, cooldown in zip(SuperWeaponType, row):
                    self.weapon_cooldowns[player, weapon_type] = int(cooldown)
        if public_state.active_effects is not None:
            self.active_effects = [
                WeaponEffect(
                    SuperWeaponType(effect_row[0]),
                    int(effect_row[1]),
                    int(effect_row[2]),
                    int(effect_row[3]),
                    int(effect_row[4]),
                )
                for effect_row in public_state.active_effects
                if len(effect_row) >= 5
            ]
            self._mark_risk_fields_dirty()
        if self.towers:
            self.next_tower_id = max(tower.tower_id for tower in self.towers) + 1
        else:
            self.next_tower_id = 0
        if self.ants:
            self.next_ant_id = max(ant.ant_id for ant in self.ants) + 1
        else:
            self.next_ant_id = 0
        self.terminal = False
        self.winner = None
        if self.round_index >= MAX_ROUND:
            self.terminal = True
            self._judge_timeout_winner()
        elif self._judge_base_camps():
            return

    def tower_spread_score(self, player: int) -> float:
        towers = self.towers_of(player)
        if len(towers) < 2:
            return 0.0
        penalty = 0.0
        for index, tower in enumerate(towers[:-1]):
            for other in towers[index + 1 :]:
                distance = hex_distance(tower.x, tower.y, other.x, other.y)
                if distance <= 3:
                    penalty += 5.0
                elif distance <= 6:
                    penalty += 2.0
        return -penalty

    def slot_priority(self, player: int, x: int, y: int) -> float:
        try:
            order = self.strategic_slots(player).index((x, y))
        except ValueError:
            order = len(self.strategic_slots(player))
        priority = max(0.0, 24.0 - order * 0.6)
        priority *= CENTERLINE_WEIGHTS.get((x, y), 1.0)
        base_x, base_y = PLAYER_BASES[player]
        priority += hex_distance(x, y, base_x, base_y) * 0.4
        return priority
