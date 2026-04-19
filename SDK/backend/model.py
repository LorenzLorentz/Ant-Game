from __future__ import annotations

from dataclasses import dataclass, field
import math

from SDK.utils.constants import (
    ANT_AGE_LIMIT,
    AntBehavior,
    AntKind,
    COMBAT_ANT_HP,
    COMBAT_ANT_KILL_REWARD,
    ANT_KILL_REWARD,
    ANT_MAX_HP,
    ANT_GENERATION_SCHEDULE,
    AntStatus,
    COMBAT_TOWER_ATTACK_DAMAGE,
    MoveWeights,
    MOVE_PROFILE_WEIGHTS,
    OFFSET,
    OperationType,
    PLAYER_BASES,
    SPECIAL_BEHAVIOR_DECAY_TURNS,
    SuperWeaponType,
    TOWER_STATS,
    TOWER_UPGRADE_TREE,
    TowerType,
    WORKER_TOWER_ATTACK_DAMAGE,
)
from SDK.utils.geometry import hex_distance

NO_MOVE = -1


def default_behavior_expiry(behavior: AntBehavior) -> int:
    if behavior in (AntBehavior.CONSERVATIVE, AntBehavior.BEWITCHED, AntBehavior.CONTROL_FREE):
        return SPECIAL_BEHAVIOR_DECAY_TURNS
    return 0


@dataclass(slots=True)
class Operation:
    op_type: OperationType
    arg0: int = -1
    arg1: int = -1

    def to_protocol_tokens(self) -> list[int]:
        if self.op_type in (
            OperationType.BUILD_TOWER,
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        ):
            return [int(self.op_type), self.arg0, self.arg1]
        if self.op_type in (OperationType.UPGRADE_TOWER,):
            return [int(self.op_type), self.arg0, self.arg1]
        if self.op_type in (OperationType.DOWNGRADE_TOWER,):
            return [int(self.op_type), self.arg0]
        return [int(self.op_type)]


@dataclass(slots=True)
class Ant:
    ant_id: int
    player: int
    x: int
    y: int
    hp: int
    level: int
    kind: AntKind = AntKind.WORKER
    age: int = 0
    status: AntStatus = AntStatus.ALIVE
    trail_cells: list[tuple[int, int]] = field(default_factory=list)
    last_move: int = NO_MOVE
    path_len_total: int = 0
    shield: int = 0
    evasion_grants_control_free: bool = False
    deflector: bool = False
    frozen: bool = False
    evasion: bool = False
    behavior: AntBehavior = AntBehavior.DEFAULT
    behavior_turns: int = 0
    behavior_expiry: int = 0
    bewitch_target_x: int = -1
    bewitch_target_y: int = -1
    pending_behavior: AntBehavior | None = None
    move_weights: MoveWeights | None = field(default=None)

    def __post_init__(self) -> None:
        if not self.trail_cells:
            self.trail_cells.append((self.x, self.y))
        if self.move_weights is None:
            self.move_weights = MOVE_PROFILE_WEIGHTS[self.kind]

    def clone(self) -> Ant:
        return Ant(
            ant_id=self.ant_id,
            player=self.player,
            x=self.x,
            y=self.y,
            hp=self.hp,
            level=self.level,
            kind=self.kind,
            age=self.age,
            status=self.status,
            trail_cells=list(self.trail_cells),
            last_move=self.last_move,
            path_len_total=self.path_len_total,
            shield=self.shield,
            evasion_grants_control_free=self.evasion_grants_control_free,
            deflector=self.deflector,
            frozen=self.frozen,
            evasion=self.evasion,
            behavior=self.behavior,
            behavior_turns=self.behavior_turns,
            behavior_expiry=self.behavior_expiry,
            bewitch_target_x=self.bewitch_target_x,
            bewitch_target_y=self.bewitch_target_y,
            pending_behavior=self.pending_behavior,
            move_weights=self.move_weights,
        )

    def record_move(self, direction: int) -> None:
        self.path_len_total += 1
        if direction == NO_MOVE:
            self.last_move = NO_MOVE
            return
        dx, dy = OFFSET[self.y % 2][direction]
        self.x += dx
        self.y += dy
        self.last_move = direction
        self.trail_cells.append((self.x, self.y))

    def teleport_to(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.last_move = NO_MOVE
        self.trail_cells.append((self.x, self.y))

    @property
    def max_hp(self) -> int:
        if self.kind == AntKind.COMBAT:
            return COMBAT_ANT_HP
        return ANT_MAX_HP[self.level]

    @property
    def kill_reward(self) -> int:
        if self.kind == AntKind.COMBAT:
            return COMBAT_ANT_KILL_REWARD
        return ANT_KILL_REWARD[self.level]

    def is_alive(self) -> bool:
        return self.status in (AntStatus.ALIVE, AntStatus.FROZEN) and self.hp > 0

    @property
    def control_immune(self) -> bool:
        return self.behavior == AntBehavior.CONTROL_FREE

    @property
    def tower_attack_damage(self) -> int:
        if self.kind == AntKind.COMBAT:
            return COMBAT_TOWER_ATTACK_DAMAGE
        return WORKER_TOWER_ATTACK_DAMAGE[self.level]

    @property
    def should_self_destruct_on_tower_attack(self) -> bool:
        return self.kind == AntKind.COMBAT and self.hp * 2 < self.max_hp

    def set_kind(self, kind: AntKind) -> None:
        self.kind = kind
        self.move_weights = MOVE_PROFILE_WEIGHTS[kind]

    def grant_evasion(self, stacks: int, *, grant_control_free_on_deplete: bool = True) -> None:
        if stacks <= 0:
            return
        self.shield = max(self.shield, stacks)
        self.evasion = self.shield > 0
        self.evasion_grants_control_free = self.evasion_grants_control_free or grant_control_free_on_deplete

    def add_evasion(self, stacks: int, *, grant_control_free_on_deplete: bool = True) -> None:
        if stacks <= 0:
            return
        self.shield += stacks
        self.evasion = self.shield > 0
        self.evasion_grants_control_free = self.evasion_grants_control_free or grant_control_free_on_deplete

    def set_behavior(
        self,
        behavior: AntBehavior,
        *,
        reset_turns: bool = True,
        force: bool = False,
        target: tuple[int, int] | None = None,
        expiry_turns: int | None = None,
    ) -> None:
        if not force and self.control_immune and behavior != AntBehavior.CONTROL_FREE:
            return
        self.behavior = behavior
        if reset_turns:
            self.behavior_turns = 0
        self.behavior_expiry = default_behavior_expiry(behavior) if expiry_turns is None else max(0, expiry_turns)
        if behavior == AntBehavior.BEWITCHED and target is not None:
            self.bewitch_target_x, self.bewitch_target_y = target
        elif behavior != AntBehavior.BEWITCHED:
            self.bewitch_target_x = -1
            self.bewitch_target_y = -1

    def refresh_status(self) -> None:
        if self.hp <= 0:
            self.status = AntStatus.FAIL
            return
        base_x, base_y = PLAYER_BASES[1 - self.player]
        if self.x == base_x and self.y == base_y:
            self.status = AntStatus.SUCCESS
            return
        if self.kind != AntKind.COMBAT and self.age > ANT_AGE_LIMIT:
            self.status = AntStatus.TOO_OLD
            return
        if self.frozen:
            self.status = AntStatus.FROZEN
            return
        self.status = AntStatus.ALIVE

    def take_damage(self, amount: int, apply_freeze: bool = False) -> None:
        if amount <= 0:
            return
        if self.shield > 0:
            self.shield -= 1
            self.evasion = self.shield > 0
            if self.shield == 0 and self.evasion_grants_control_free and not self.control_immune:
                self.evasion_grants_control_free = False
                self.set_behavior(AntBehavior.CONTROL_FREE)
            return
        if self.deflector and amount * 2 < self.max_hp:
            return
        self.hp -= amount
        if apply_freeze and self.hp > 0:
            self.frozen = True
        self.refresh_status()


@dataclass(slots=True)
class Tower:
    tower_id: int
    player: int
    x: int
    y: int
    tower_type: TowerType = TowerType.BASIC
    cooldown_clock: float = 0.0
    hp: int = -1

    def __post_init__(self) -> None:
        if self.hp < 0:
            self.hp = self.stats.max_hp

    def clone(self) -> Tower:
        return Tower(
            tower_id=self.tower_id,
            player=self.player,
            x=self.x,
            y=self.y,
            tower_type=self.tower_type,
            cooldown_clock=self.cooldown_clock,
            hp=self.hp,
        )

    @property
    def stats(self):
        return TOWER_STATS[self.tower_type]

    @property
    def damage(self) -> int:
        return self.stats.damage

    @property
    def speed(self) -> float:
        return self.stats.speed

    @property
    def attack_range(self) -> int:
        return self.stats.attack_range

    @property
    def max_hp(self) -> int:
        return self.stats.max_hp

    @property
    def is_producer(self) -> bool:
        return self.tower_type in (
            TowerType.PRODUCER,
            TowerType.PRODUCER_FAST,
            TowerType.PRODUCER_SIEGE,
            TowerType.PRODUCER_MEDIC,
        )

    @property
    def level(self) -> int:
        if self.tower_type == TowerType.BASIC:
            return 0
        if self.tower_type.value < 10:
            return 1
        return 2

    def reset_cooldown(self) -> None:
        if self.is_producer:
            self.cooldown_clock = float(self.stats.spawn_interval)
        else:
            self.cooldown_clock = self.speed

    def is_upgrade_type_valid(self, target: TowerType) -> bool:
        return target in TOWER_UPGRADE_TREE.get(self.tower_type, ())

    def upgrade(self, target: TowerType) -> None:
        self.tower_type = target
        self.hp = self.max_hp
        self.reset_cooldown()

    def downgrade_or_destroy(self) -> bool:
        if self.tower_type == TowerType.BASIC:
            return True
        previous_max_hp = self.max_hp
        previous_hp = max(0, self.hp)
        self.tower_type = TowerType(self.tower_type.value // 10)
        if previous_max_hp > 0:
            self.hp = max(1, math.ceil(self.max_hp * previous_hp / previous_max_hp))
        else:
            self.hp = self.max_hp
        self.reset_cooldown()
        return False

    def take_damage(self, amount: int) -> bool:
        if amount <= 0:
            return False
        self.hp -= amount
        return self.hp <= 0

    def ready_to_fire(self) -> bool:
        return self.cooldown_clock <= 0.0

    def tick(self) -> None:
        if self.cooldown_clock > 0.0:
            self.cooldown_clock -= 1.0

    def display_cooldown(self) -> int:
        if self.is_producer:
            return max(int(self.cooldown_clock), 0)
        if self.speed < 1:
            return 0
        return max(int(self.cooldown_clock), 0)


@dataclass(slots=True)
class Base:
    player: int
    x: int
    y: int
    hp: int = 50
    generation_level: int = 0
    ant_level: int = 0

    def clone(self) -> Base:
        return Base(
            player=self.player,
            x=self.x,
            y=self.y,
            hp=self.hp,
            generation_level=self.generation_level,
            ant_level=self.ant_level,
        )

    def should_spawn(self, round_index: int) -> bool:
        numerator, denominator = ANT_GENERATION_SCHEDULE[self.generation_level]
        if round_index == 0:
            return True
        return (round_index * denominator) // numerator > ((round_index - 1) * denominator) // numerator

    def spawn_ant(self, ant_id: int, *, kind: AntKind = AntKind.WORKER) -> Ant:
        return Ant(
            ant_id=ant_id,
            player=self.player,
            x=self.x,
            y=self.y,
            hp=COMBAT_ANT_HP if kind == AntKind.COMBAT else ANT_MAX_HP[self.ant_level],
            level=self.ant_level,
            kind=kind,
        )


@dataclass(slots=True)
class WeaponEffect:
    weapon_type: SuperWeaponType
    player: int
    x: int
    y: int
    remaining_turns: int
    last_trigger_round: int = -1

    def clone(self) -> WeaponEffect:
        return WeaponEffect(
            weapon_type=self.weapon_type,
            player=self.player,
            x=self.x,
            y=self.y,
            remaining_turns=self.remaining_turns,
            last_trigger_round=self.last_trigger_round,
        )

    def in_range(self, x: int, y: int) -> bool:
        from SDK.utils.constants import SUPER_WEAPON_STATS

        return hex_distance(self.x, self.y, x, y) <= SUPER_WEAPON_STATS[self.weapon_type].attack_range
