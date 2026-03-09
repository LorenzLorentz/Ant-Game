from __future__ import annotations

from dataclasses import dataclass, field

from SDK.constants import (
    ANT_AGE_LIMIT,
    AntBehavior,
    ANT_KILL_REWARD,
    ANT_MAX_HP,
    ANT_GENERATION_CYCLE,
    AntStatus,
    OFFSET,
    OperationType,
    PLAYER_BASES,
    SuperWeaponType,
    TOWER_STATS,
    TOWER_UPGRADE_TREE,
    TowerType,
)
from SDK.geometry import hex_distance

NO_MOVE = -1


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
    age: int = 0
    status: AntStatus = AntStatus.ALIVE
    trail_cells: list[tuple[int, int]] = field(default_factory=list)
    last_move: int = NO_MOVE
    path_len_total: int = 0
    shield: int = 0
    deflector: bool = False
    frozen: bool = False
    evasion: bool = False
    behavior: AntBehavior = AntBehavior.DEFAULT
    behavior_turns: int = 0
    bewitch_target_x: int = -1
    bewitch_target_y: int = -1
    pending_behavior: AntBehavior | None = None

    def __post_init__(self) -> None:
        if not self.trail_cells:
            self.trail_cells.append((self.x, self.y))

    def clone(self) -> Ant:
        return Ant(
            ant_id=self.ant_id,
            player=self.player,
            x=self.x,
            y=self.y,
            hp=self.hp,
            level=self.level,
            age=self.age,
            status=self.status,
            trail_cells=list(self.trail_cells),
            last_move=self.last_move,
            path_len_total=self.path_len_total,
            shield=self.shield,
            deflector=self.deflector,
            frozen=self.frozen,
            evasion=self.evasion,
            behavior=self.behavior,
            behavior_turns=self.behavior_turns,
            bewitch_target_x=self.bewitch_target_x,
            bewitch_target_y=self.bewitch_target_y,
            pending_behavior=self.pending_behavior,
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
        return ANT_MAX_HP[self.level]

    @property
    def kill_reward(self) -> int:
        return ANT_KILL_REWARD[self.level]

    def is_alive(self) -> bool:
        return self.status in (AntStatus.ALIVE, AntStatus.FROZEN) and self.hp > 0

    @property
    def control_immune(self) -> bool:
        return self.behavior == AntBehavior.CONTROL_FREE

    def set_behavior(
        self,
        behavior: AntBehavior,
        *,
        reset_turns: bool = True,
        target: tuple[int, int] | None = None,
    ) -> None:
        if self.control_immune and behavior != AntBehavior.CONTROL_FREE:
            return
        self.behavior = behavior
        if reset_turns:
            self.behavior_turns = 0
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
        if self.age > ANT_AGE_LIMIT:
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

    def clone(self) -> Tower:
        return Tower(
            tower_id=self.tower_id,
            player=self.player,
            x=self.x,
            y=self.y,
            tower_type=self.tower_type,
            cooldown_clock=self.cooldown_clock,
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
    def level(self) -> int:
        if self.tower_type == TowerType.BASIC:
            return 0
        if self.tower_type.value < 10:
            return 1
        return 2

    def reset_cooldown(self) -> None:
        self.cooldown_clock = self.speed

    def is_upgrade_type_valid(self, target: TowerType) -> bool:
        return target in TOWER_UPGRADE_TREE.get(self.tower_type, ())

    def upgrade(self, target: TowerType) -> None:
        self.tower_type = target
        self.reset_cooldown()

    def downgrade_or_destroy(self) -> bool:
        if self.tower_type == TowerType.BASIC:
            return True
        self.tower_type = TowerType(self.tower_type.value // 10)
        self.reset_cooldown()
        return False

    def ready_to_fire(self) -> bool:
        return self.cooldown_clock <= 0.0

    def tick(self) -> None:
        if self.cooldown_clock > 0.0:
            self.cooldown_clock -= 1.0

    def display_cooldown(self) -> int:
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
        return round_index % ANT_GENERATION_CYCLE[self.generation_level] == 0

    def spawn_ant(self, ant_id: int) -> Ant:
        return Ant(
            ant_id=ant_id,
            player=self.player,
            x=self.x,
            y=self.y,
            hp=ANT_MAX_HP[self.ant_level],
            level=self.ant_level,
        )


@dataclass(slots=True)
class WeaponEffect:
    weapon_type: SuperWeaponType
    player: int
    x: int
    y: int
    remaining_turns: int

    def clone(self) -> WeaponEffect:
        return WeaponEffect(
            weapon_type=self.weapon_type,
            player=self.player,
            x=self.x,
            y=self.y,
            remaining_turns=self.remaining_turns,
        )

    def in_range(self, x: int, y: int) -> bool:
        from SDK.constants import SUPER_WEAPON_STATS

        return hex_distance(self.x, self.y, x, y) <= SUPER_WEAPON_STATS[self.weapon_type].attack_range
