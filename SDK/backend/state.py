from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from SDK.backend.engine import DEFAULT_MOVEMENT_POLICY, GameState, PublicRoundState, TurnResolution
from SDK.backend.model import Ant, Base, Operation, Tower, WeaponEffect


@runtime_checkable
class BackendState(Protocol):
    seed: int
    movement_policy: str
    cold_handle_rule_illegal: bool
    round_index: int
    terminal: bool
    winner: int | None
    towers: list[Tower]
    ants: list[Ant]
    bases: list[Base]
    coins: list[int]
    pheromone: np.ndarray
    weapon_cooldowns: np.ndarray
    active_effects: list[WeaponEffect]
    old_count: list[int]
    die_count: list[int]
    super_weapon_usage: list[int]
    next_ant_id: int
    next_tower_id: int

    def clone(self) -> BackendState: ...
    def tower_count(self, player: int) -> int: ...
    def towers_of(self, player: int) -> list[Tower]: ...
    def ants_of(self, player: int) -> list[Ant]: ...
    def tower_at(self, x: int, y: int) -> Tower | None: ...
    def tower_by_id(self, tower_id: int) -> Tower | None: ...
    def strategic_slots(self, player: int) -> tuple[tuple[int, int], ...]: ...
    def build_tower_cost(self, tower_count: int | None = None) -> int: ...
    def upgrade_tower_cost(self, target_type) -> int: ...
    def destroy_tower_income(self, tower_count: int, tower: Tower | None = None) -> int: ...
    def downgrade_tower_income(self, tower_type) -> int: ...
    def upgrade_base_cost(self, level: int) -> int: ...
    def weapon_cost(self, weapon_type) -> int: ...
    def nearest_ant_distance(self, player: int) -> int: ...
    def frontline_distance(self, player: int) -> int: ...
    def safe_coin_threshold(self, player: int) -> int: ...
    def current_and_neighbors_empty(self, x: int, y: int) -> bool: ...
    def is_shielded_by_emp(self, player: int, x: int, y: int) -> bool: ...
    def is_shielded_by_deflector(self, ant: Ant) -> bool: ...
    def weapon_effect(self, weapon_type, player: int) -> WeaponEffect | None: ...
    def can_apply_operation(self, player: int, operation: Operation, pending: Iterable[Operation] = ()) -> bool: ...
    def operation_income(
        self,
        player: int,
        operation: Operation,
        tower_count_hint: int | None = None,
    ) -> int: ...
    def apply_operation(self, player: int, operation: Operation) -> None: ...
    def apply_operation_list(self, player: int, operations: Iterable[Operation]) -> list[Operation]: ...
    def advance_round(self) -> None: ...
    def resolve_turn(self, operations0: Iterable[Operation], operations1: Iterable[Operation]) -> TurnResolution: ...
    def to_public_round_state(self) -> PublicRoundState: ...
    def sync_public_round_state(self, public_state: PublicRoundState) -> None: ...
    def tower_spread_score(self, player: int) -> float: ...
    def slot_priority(self, player: int, x: int, y: int) -> float: ...


class PythonBackendState:
    """Thin adapter that exposes the stable backend surface."""

    __slots__ = ("_state",)

    def __init__(self, state: GameState) -> None:
        self._state = state

    @classmethod
    def initial(
        cls,
        seed: int = 0,
        movement_policy: str = DEFAULT_MOVEMENT_POLICY,
        cold_handle_rule_illegal: bool = False,
    ) -> PythonBackendState:
        return cls(
            GameState.initial(
                seed=seed,
                movement_policy=movement_policy,
                cold_handle_rule_illegal=cold_handle_rule_illegal,
            )
        )

    @property
    def seed(self) -> int:
        return self._state.seed

    @property
    def round_index(self) -> int:
        return self._state.round_index

    @round_index.setter
    def round_index(self, value: int) -> None:
        self._state.round_index = value

    @property
    def terminal(self) -> bool:
        return self._state.terminal

    @terminal.setter
    def terminal(self, value: bool) -> None:
        self._state.terminal = value

    @property
    def winner(self) -> int | None:
        return self._state.winner

    @winner.setter
    def winner(self, value: int | None) -> None:
        self._state.winner = value

    @property
    def towers(self) -> list[Tower]:
        return self._state.towers

    @property
    def ants(self) -> list[Ant]:
        return self._state.ants

    @property
    def bases(self) -> list[Base]:
        return self._state.bases

    @property
    def coins(self) -> list[int]:
        return self._state.coins

    @property
    def movement_policy(self) -> str:
        return self._state.movement_policy

    @property
    def cold_handle_rule_illegal(self) -> bool:
        return self._state.cold_handle_rule_illegal

    @property
    def pheromone(self) -> np.ndarray:
        return self._state.pheromone

    @property
    def weapon_cooldowns(self) -> np.ndarray:
        return self._state.weapon_cooldowns

    @property
    def active_effects(self) -> list[WeaponEffect]:
        return self._state.active_effects

    @property
    def old_count(self) -> list[int]:
        return self._state.old_count

    @property
    def die_count(self) -> list[int]:
        return self._state.die_count

    @property
    def super_weapon_usage(self) -> list[int]:
        return self._state.super_weapon_usage

    @property
    def next_ant_id(self) -> int:
        return self._state.next_ant_id

    @next_ant_id.setter
    def next_ant_id(self, value: int) -> None:
        self._state.next_ant_id = value

    @property
    def next_tower_id(self) -> int:
        return self._state.next_tower_id

    @next_tower_id.setter
    def next_tower_id(self, value: int) -> None:
        self._state.next_tower_id = value

    def clone(self) -> PythonBackendState:
        return PythonBackendState(self._state.clone())

    def tower_count(self, player: int) -> int:
        return self._state.tower_count(player)

    def towers_of(self, player: int) -> list[Tower]:
        return self._state.towers_of(player)

    def ants_of(self, player: int) -> list[Ant]:
        return self._state.ants_of(player)

    def tower_at(self, x: int, y: int) -> Tower | None:
        return self._state.tower_at(x, y)

    def tower_by_id(self, tower_id: int) -> Tower | None:
        return self._state.tower_by_id(tower_id)

    def strategic_slots(self, player: int) -> tuple[tuple[int, int], ...]:
        return self._state.strategic_slots(player)

    def build_tower_cost(self, tower_count: int | None = None) -> int:
        return self._state.build_tower_cost(tower_count)

    def upgrade_tower_cost(self, target_type) -> int:
        return self._state.upgrade_tower_cost(target_type)

    def destroy_tower_income(self, tower_count: int, tower: Tower | None = None) -> int:
        return self._state.destroy_tower_income(tower_count, tower)

    def downgrade_tower_income(self, tower_type) -> int:
        return self._state.downgrade_tower_income(tower_type)

    def upgrade_base_cost(self, level: int) -> int:
        return self._state.upgrade_base_cost(level)

    def weapon_cost(self, weapon_type) -> int:
        return self._state.weapon_cost(weapon_type)

    def nearest_ant_distance(self, player: int) -> int:
        return self._state.nearest_ant_distance(player)

    def frontline_distance(self, player: int) -> int:
        return self._state.frontline_distance(player)

    def safe_coin_threshold(self, player: int) -> int:
        return self._state.safe_coin_threshold(player)

    def current_and_neighbors_empty(self, x: int, y: int) -> bool:
        return self._state.current_and_neighbors_empty(x, y)

    def is_shielded_by_emp(self, player: int, x: int, y: int) -> bool:
        return self._state.is_shielded_by_emp(player, x, y)

    def is_shielded_by_deflector(self, ant: Ant) -> bool:
        return self._state.is_shielded_by_deflector(ant)

    def weapon_effect(self, weapon_type, player: int) -> WeaponEffect | None:
        return self._state.weapon_effect(weapon_type, player)

    def can_apply_operation(self, player: int, operation: Operation, pending: Iterable[Operation] = ()) -> bool:
        return self._state.can_apply_operation(player, operation, pending)

    def operation_income(
        self,
        player: int,
        operation: Operation,
        tower_count_hint: int | None = None,
    ) -> int:
        return self._state.operation_income(player, operation, tower_count_hint)

    def apply_operation(self, player: int, operation: Operation) -> None:
        self._state.apply_operation(player, operation)

    def apply_operation_list(self, player: int, operations: Iterable[Operation]) -> list[Operation]:
        return self._state.apply_operation_list(player, operations)

    def advance_round(self) -> None:
        self._state.advance_round()

    def resolve_turn(self, operations0: Iterable[Operation], operations1: Iterable[Operation]) -> TurnResolution:
        return self._state.resolve_turn(operations0, operations1)

    def to_public_round_state(self) -> PublicRoundState:
        return self._state.to_public_round_state()

    def sync_public_round_state(self, public_state: PublicRoundState) -> None:
        self._state.sync_public_round_state(public_state)

    def tower_spread_score(self, player: int) -> float:
        return self._state.tower_spread_score(player)

    def slot_priority(self, player: int, x: int, y: int) -> float:
        return self._state.slot_priority(player, x, y)


def create_python_backend_state(seed: int = 0, cold_handle_rule_illegal: bool = False) -> BackendState:
    return PythonBackendState.initial(seed=seed, cold_handle_rule_illegal=cold_handle_rule_illegal)


__all__ = [
    "BackendState",
    "PythonBackendState",
    "create_python_backend_state",
]
