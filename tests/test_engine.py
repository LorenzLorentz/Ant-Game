from __future__ import annotations

from SDK.constants import LAMBDA_DENOM, LAMBDA_NUM, PHEROMONE_FAIL_BONUS_INT, TAU_BASE_ADD_INT
from SDK.constants import ANT_TELEPORT_INTERVAL, AntBehavior, AntStatus, OperationType, SuperWeaponType, TowerType
from SDK.engine import GameState, PublicRoundState
from SDK.model import Ant, Operation, Tower, WeaponEffect


def test_initial_round_spawns_ants_and_advances_time() -> None:
    state = GameState.initial(seed=7)
    state.resolve_turn([], [])
    assert state.round_index == 1
    assert len(state.ants) == 2
    assert state.coins == [51, 51]


def test_build_and_upgrade_tower_updates_coin_and_state() -> None:
    state = GameState.initial(seed=3)
    build = Operation(OperationType.BUILD_TOWER, 6, 9)
    assert state.can_apply_operation(0, build)
    assert state.apply_operation_list(0, [build]) == []
    assert state.coins[0] == 35
    tower = state.tower_at(6, 9)
    assert tower is not None
    upgrade = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(TowerType.HEAVY))
    state.coins[0] = 100
    assert state.can_apply_operation(0, upgrade)
    assert state.apply_operation_list(0, [upgrade]) == []
    assert tower.tower_type == TowerType.HEAVY
    assert state.coins[0] == 40


def test_quick_tower_attacks_enemy_ant() -> None:
    state = GameState.initial(seed=1)
    state.towers.append(Tower(0, 0, 6, 9, TowerType.QUICK, cooldown_clock=1.0))
    state.ants.append(Ant(0, 1, 8, 9, hp=10, level=0))
    state.advance_round()
    assert state.die_count[1] == 1 or any(ant.hp < 10 for ant in state.ants)


def test_emp_prevents_building_inside_field() -> None:
    state = GameState.initial(seed=1)
    state.active_effects.append(__import__('SDK.model', fromlist=['WeaponEffect']).WeaponEffect(__import__('SDK.constants', fromlist=['SuperWeaponType']).SuperWeaponType.EMP_BLASTER, 1, 6, 9, 3))
    blocked = Operation(OperationType.BUILD_TOWER, 6, 9)
    assert not state.can_apply_operation(0, blocked)


def test_random_ant_degrades_to_default_after_five_rounds() -> None:
    ant = Ant(0, 0, 2, 9, hp=10, level=0, behavior=AntBehavior.RANDOM)
    state = GameState.initial(seed=3)
    state.ants.append(ant)
    for _ in range(5):
        state._increase_ant_age()
    assert ant.behavior == AntBehavior.DEFAULT


def test_ice_freeze_promotes_ant_to_random_after_thaw() -> None:
    state = GameState.initial(seed=2)
    ant = Ant(0, 1, 8, 9, hp=25, level=1, behavior=AntBehavior.CONSERVATIVE)
    tower = Tower(0, 0, 6, 9, TowerType.ICE, cooldown_clock=0.0)
    state.ants.append(ant)
    state._damage_ant_from_tower(tower, ant)
    assert ant.frozen
    state._prepare_ants_for_attack()
    assert ant.behavior == AntBehavior.RANDOM


def test_control_free_ant_ignores_control_and_teleport() -> None:
    state = GameState.initial(seed=9)
    immune = Ant(0, 1, 8, 9, hp=10, level=0, behavior=AntBehavior.CONTROL_FREE)
    target = Ant(1, 1, 9, 9, hp=10, level=0, behavior=AntBehavior.DEFAULT)
    state.ants.extend([immune, target])
    original = (immune.x, immune.y)
    state._control_ant(immune, AntBehavior.RANDOM)
    assert immune.behavior == AntBehavior.CONTROL_FREE
    state.round_index = ANT_TELEPORT_INTERVAL - 1
    state._teleport_ants()
    assert (immune.x, immune.y) == original


def test_lightning_and_emp_effects_drift_each_tick() -> None:
    state = GameState.initial(seed=11)
    state.active_effects = [
        WeaponEffect(SuperWeaponType.LIGHTNING_STORM, 0, 9, 9, 3),
        WeaponEffect(SuperWeaponType.EMP_BLASTER, 1, 10, 9, 3),
    ]
    before = [(effect.x, effect.y) for effect in state.active_effects]
    state._tick_effects()
    after = [(effect.x, effect.y) for effect in state.active_effects]
    assert len(after) == 2
    assert all(state.active_effects[index].remaining_turns == 2 for index in range(2))
    assert all(0 <= x < 19 and 0 <= y < 19 for x, y in after)
    assert before != after


def test_public_round_state_serializes_path_length_not_hidden_age() -> None:
    state = GameState.initial(seed=5)
    state.ants.append(Ant(7, 0, 4, 9, hp=10, level=0, age=12, path=[4, 4], status=AntStatus.ALIVE))
    public_state = state.to_public_round_state()
    assert public_state.ants[0][6] == 2


def test_sync_public_round_state_preserves_hidden_age_and_behavior() -> None:
    state = GameState.initial(seed=3)
    ant = Ant(8, 0, 4, 9, hp=10, level=0, age=9, path=[4, 4, 1], behavior=AntBehavior.RANDOM)
    state.ants.append(ant)
    public_state = PublicRoundState(
        round_index=0,
        towers=[],
        ants=[(8, 0, 5, 9, 8, 0, 5, 0)],
        coins=(50, 50),
        camps_hp=(50, 50),
    )
    state.sync_public_round_state(public_state)
    synced = state.ants[0]
    assert synced.age == 9
    assert synced.behavior == AntBehavior.RANDOM
    assert len(synced.path) == 5
    assert synced.path[-2:] == [-1, -1]


def test_update_pheromone_walks_backwards_from_current_position() -> None:
    state = GameState.initial(seed=1)
    ant = Ant(3, 0, 6, 9, hp=0, level=0, age=4, path=[4], status=AntStatus.FAIL)
    state.ants.append(ant)
    before_current = int(state.pheromone[0, 6, 9])
    before_backtrack = int(state.pheromone[0, 5, 9])
    before_base = int(state.pheromone[0, 2, 9])
    state._update_pheromone()
    attenuated_current = max(0, (LAMBDA_NUM * before_current + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM)
    attenuated_backtrack = max(0, (LAMBDA_NUM * before_backtrack + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM)
    attenuated_base = max(0, (LAMBDA_NUM * before_base + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM)
    assert int(state.pheromone[0, 6, 9]) == attenuated_current + PHEROMONE_FAIL_BONUS_INT
    assert int(state.pheromone[0, 5, 9]) == attenuated_backtrack + PHEROMONE_FAIL_BONUS_INT
    assert int(state.pheromone[0, 2, 9]) == attenuated_base


def test_too_old_ants_remain_visible_until_next_lifecycle_cleanup() -> None:
    state = GameState.initial(seed=4)
    ant = Ant(11, 0, 2, 9, hp=10, level=0, age=32)
    state.ants.append(ant)
    state.advance_round()
    tracked = next(item for item in state.ants if item.ant_id == 11)
    assert tracked.status.name == "TOO_OLD"
    assert state.old_count == [0, 0]
    state.advance_round()
    assert all(item.ant_id != 11 for item in state.ants)
    assert state.old_count == [1, 0]
