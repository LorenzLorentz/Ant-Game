from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from SDK import native_antwar
from SDK.constants import AntStatus, OperationType, SuperWeaponType, TowerType
from SDK.engine import GameState, PublicRoundState, TurnResolution
from SDK.model import Ant, Base, Operation, Tower, WeaponEffect


def _to_native_operation(operation: Operation) -> native_antwar.Operation:
    return native_antwar.Operation(int(operation.op_type), int(operation.arg0), int(operation.arg1))


def _to_python_operation(operation: native_antwar.Operation) -> Operation:
    return Operation(OperationType(int(operation.type)), int(operation.arg0), int(operation.arg1))


def _build_shadow_state(native: native_antwar.NativeState) -> GameState:
    state = GameState.initial(seed=int(native.seed))
    _sync_shadow_state(state, native)
    return state


def _sync_shadow_state(state: GameState, native: native_antwar.NativeState) -> None:
    state.round_index = int(native.round_index())
    state.coins = list(native.coins())
    native_old_count = list(native.old_count())
    if any(native_old_count) or not any(state.old_count):
        state.old_count = native_old_count
    state.die_count = list(native.die_count())
    state.super_weapon_usage = list(native.super_weapon_usage())
    state.ai_time = list(native.ai_time())
    state.weapon_cooldowns = np.asarray(native.weapon_cooldowns(), dtype=np.int16)

    state.towers = [
        Tower(
            tower_id=int(tower_id),
            player=int(player),
            x=int(x),
            y=int(y),
            tower_type=TowerType(int(tower_type)),
            cooldown_clock=float(cooldown),
        )
        for tower_id, player, x, y, tower_type, cooldown in native.tower_rows()
    ]

    ant_map = {ant.ant_id: ant for ant in state.ants}
    synced_ants: list[Ant] = []
    for ant_id, player, x, y, hp, level, age, status in native.ant_rows():
        ant = ant_map.get(int(ant_id))
        if ant is None:
            ant = Ant(
                ant_id=int(ant_id),
                player=int(player),
                x=int(x),
                y=int(y),
                hp=int(hp),
                level=int(level),
                age=int(age),
                status=AntStatus(int(status)),
            )
        ant.player = int(player)
        ant.x = int(x)
        ant.y = int(y)
        ant.hp = int(hp)
        ant.level = int(level)
        ant.age = int(age)
        ant.status = AntStatus(int(status))
        synced_ants.append(ant)
    state.ants = synced_ants

    bases = [
        Base(
            player=int(player),
            x=int(x),
            y=int(y),
            hp=int(hp),
            generation_level=int(generation_level),
            ant_level=int(ant_level),
        )
        for player, x, y, hp, generation_level, ant_level in native.base_rows()
    ]
    bases.sort(key=lambda item: item.player)
    state.bases = bases

    state.active_effects = [
        WeaponEffect(
            weapon_type=SuperWeaponType(int(weapon_type)),
            player=int(player),
            x=int(x),
            y=int(y),
            remaining_turns=int(remaining_turns),
        )
        for weapon_type, player, x, y, remaining_turns in native.effect_rows()
    ]

    state.next_ant_id = int(native.next_ant_id())
    state.next_tower_id = int(native.next_tower_id())
    state.terminal = bool(native.terminal)
    winner = int(native.winner)
    state.winner = None if winner < 0 else winner


@dataclass(slots=True)
class NativeGameStateAdapter:
    native: native_antwar.NativeState
    _shadow: GameState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._refresh_cache()

    @classmethod
    def initial(cls, seed: int = 0) -> NativeGameStateAdapter:
        return cls(native_antwar.NativeState(seed))

    def __getattr__(self, name: str):
        return getattr(self._shadow, name)

    def _refresh_cache(self) -> None:
        try:
            shadow = object.__getattribute__(self, "_shadow")
        except AttributeError:
            shadow = None
        if shadow is not None:
            _sync_shadow_state(self._shadow, self.native)
        else:
            self._shadow = _build_shadow_state(self.native)

    def clone(self) -> NativeGameStateAdapter:
        clone = NativeGameStateAdapter(self.native.clone())
        clone._shadow = self._shadow.clone()
        clone._refresh_cache()
        return clone

    def apply_operation_list(self, player: int, operations) -> list[Operation]:
        operation_list = list(operations)
        illegal = self.native.apply_operation_list(player, [_to_native_operation(operation) for operation in operation_list])
        self._shadow.apply_operation_list(player, operation_list)
        self._refresh_cache()
        return [_to_python_operation(operation) for operation in illegal]

    def apply_operation(self, player: int, operation: Operation) -> None:
        self.native.apply_operation_list(player, [_to_native_operation(operation)])
        self._shadow.apply_operation(player, operation)
        self._refresh_cache()

    def advance_round(self) -> None:
        self.native.advance_round()
        self._shadow.advance_round()
        self._refresh_cache()

    def resolve_turn(self, operations0, operations1) -> TurnResolution:
        operations0 = list(operations0)
        operations1 = list(operations1)
        result = self.native.resolve_turn(
            [_to_native_operation(operation) for operation in operations0],
            [_to_native_operation(operation) for operation in operations1],
        )
        self._shadow.resolve_turn(operations0, operations1)
        self._refresh_cache()
        winner = int(result["winner"])
        return TurnResolution(
            (operations0, operations1),
            (
                [_to_python_operation(operation) for operation in result["illegal0"]],
                [_to_python_operation(operation) for operation in result["illegal1"]],
            ),
            bool(result["terminal"]),
            None if winner < 0 else winner,
        )

    def to_public_round_state(self) -> PublicRoundState:
        return self._shadow.to_public_round_state()

    def sync_public_round_state(self, public_state: PublicRoundState) -> None:
        self.native.sync_public_round_state(
            int(public_state.round_index),
            [list(row) for row in public_state.towers],
            [list(row) for row in public_state.ants],
            list(public_state.coins),
            list(public_state.camps_hp),
        )
        self._shadow.sync_public_round_state(public_state)
        self._refresh_cache()
