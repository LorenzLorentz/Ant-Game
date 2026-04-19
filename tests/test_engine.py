from __future__ import annotations

from SDK.utils.constants import LAMBDA_DENOM, LAMBDA_NUM, PHEROMONE_FAIL_BONUS_INT, SUPER_WEAPON_STATS, TAU_BASE_ADD_INT
from SDK.utils.constants import ANT_AGE_LIMIT, ANT_TELEPORT_INTERVAL, ANT_TELEPORT_RATIO, BASIC_INCOME, COMBAT_ANT_KILL_REWARD, INITIAL_COINS, TOWER_DOWNGRADE_REFUND_RATIO, AntBehavior, AntKind, AntStatus, OperationType, PATH_CELLS, PLAYER_BASES, SPECIAL_BEHAVIOR_DECAY_TURNS, SPAWN_PROFILE_WEIGHTS, SuperWeaponType, TowerType
from SDK.backend.engine import (
    MOVEMENT_POLICY_ENHANCED,
    MOVEMENT_POLICY_LEGACY,
    GameState,
    PublicRoundState,
)
from SDK.backend.model import Ant, Operation, Tower, WeaponEffect
from SDK.utils.geometry import direction_between, hex_distance, is_path, neighbors


def _half_plane_delta(player: int, x: int, y: int) -> int:
    return hex_distance(x, y, *PLAYER_BASES[player]) - hex_distance(x, y, *PLAYER_BASES[1 - player])


def _sample_adjacent_tower_attack_count(
    *,
    seeds: int,
    ant_x: int,
    ant_y: int,
    kind: AntKind,
    behavior: AntBehavior,
    movement_policy: str = MOVEMENT_POLICY_ENHANCED,
) -> int:
    attacks = 0
    for seed in range(seeds):
        state = GameState.initial(seed=seed, movement_policy=movement_policy)
        tower = Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0, hp=10)
        ant = Ant(
            0,
            0,
            ant_x,
            ant_y,
            hp=30 if kind == AntKind.COMBAT else 20,
            level=0,
            kind=kind,
            behavior=behavior,
        )
        state.towers.append(tower)
        state.ants.append(ant)
        chosen = state._choose_ant_move(ant)
        if chosen == direction_between(ant.x, ant.y, tower.x, tower.y):
            attacks += 1
    return attacks


def _sample_adjacent_tower_attack_rate(
    *,
    seeds: int,
    kind: AntKind,
    behavior: AntBehavior,
    movement_policy: str = MOVEMENT_POLICY_ENHANCED,
) -> float:
    tower_x, tower_y = 12, 9
    attacks = 0
    samples = 0
    for _, ant_x, ant_y in neighbors(tower_x, tower_y):
        if not is_path(ant_x, ant_y):
            continue
        attacks += _sample_adjacent_tower_attack_count(
            seeds=seeds,
            ant_x=ant_x,
            ant_y=ant_y,
            kind=kind,
            behavior=behavior,
            movement_policy=movement_policy,
        )
        samples += seeds
    return attacks / samples


def _sample_adjacent_tower_attack_resolution_rate(
    *,
    seeds: int,
    kind: AntKind,
    behavior: AntBehavior,
    movement_policy: str = MOVEMENT_POLICY_ENHANCED,
) -> float:
    tower_x, tower_y = 12, 9
    attacks = 0
    samples = 0
    for _, ant_x, ant_y in neighbors(tower_x, tower_y):
        if not is_path(ant_x, ant_y):
            continue
        for seed in range(seeds):
            state = GameState.initial(seed=seed, movement_policy=movement_policy)
            tower = Tower(0, 1, tower_x, tower_y, TowerType.BASIC, cooldown_clock=2.0, hp=10)
            ant = Ant(
                0,
                0,
                ant_x,
                ant_y,
                hp=30 if kind == AntKind.COMBAT else 20,
                level=0,
                kind=kind,
                behavior=behavior,
            )
            state.towers.append(tower)
            state.ants.append(ant)
            start_pos = (ant.x, ant.y)
            start_hp = tower.hp
            state.advance_round()
            tracked_ant = next((item for item in state.ants if item.ant_id == ant.ant_id), None)
            tracked_tower = next((item for item in state.towers if item.tower_id == tower.tower_id), None)
            tower_damaged = tracked_tower is None or tracked_tower.hp < start_hp
            attacks += int(tracked_ant is not None and (tracked_ant.x, tracked_ant.y) == start_pos and tower_damaged)
            samples += 1
    return attacks / samples


def test_initial_round_spawns_ants_and_advances_time() -> None:
    state = GameState.initial(seed=7)
    state.resolve_turn([], [])
    assert state.round_index == 1
    assert len(state.ants) == 2
    assert all(ant.hp in {20, 30} for ant in state.ants)
    assert state.coins == [INITIAL_COINS, INITIAL_COINS]


def test_basic_income_pays_three_every_two_rounds() -> None:
    state = GameState.initial(seed=13)
    state.resolve_turn([], [])
    assert state.coins == [INITIAL_COINS, INITIAL_COINS]
    state.resolve_turn([], [])
    assert state.coins == [INITIAL_COINS + BASIC_INCOME, INITIAL_COINS + BASIC_INCOME]


def test_base_upgrade_curves_match_spec() -> None:
    state = GameState.initial(seed=1)
    assert state.bases[0].should_spawn(0) is True
    assert state.bases[0].should_spawn(1) is False
    assert state.bases[0].should_spawn(2) is False
    assert state.bases[0].should_spawn(3) is False
    assert state.bases[0].should_spawn(4) is False
    assert state.bases[0].should_spawn(5) is True
    assert state.bases[0].spawn_ant(1).hp == 20

    state.bases[0].generation_level = 1
    state.bases[0].ant_level = 1
    assert state.bases[0].should_spawn(1) is False
    assert state.bases[0].should_spawn(2) is False
    assert state.bases[0].should_spawn(3) is False
    assert state.bases[0].should_spawn(4) is True
    assert state.bases[0].should_spawn(5) is False
    assert state.bases[0].should_spawn(6) is False
    assert state.bases[0].should_spawn(7) is False
    assert state.bases[0].should_spawn(8) is True
    assert state.bases[0].spawn_ant(2).hp == 25

    state.bases[0].generation_level = 2
    state.bases[0].ant_level = 2
    assert state.bases[0].should_spawn(1) is False
    assert state.bases[0].should_spawn(2) is False
    assert state.bases[0].should_spawn(3) is False
    assert state.bases[0].should_spawn(4) is True
    assert state.bases[0].should_spawn(5) is False
    assert state.bases[0].should_spawn(6) is False
    assert state.bases[0].should_spawn(7) is True
    assert state.bases[0].spawn_ant(3).hp == 25


def test_tower_rebalance_stats_match_spec() -> None:
    assert Tower(0, 0, 6, 9, TowerType.BASIC).attack_range == 1
    assert Tower(0, 0, 6, 9, TowerType.ICE).attack_range == 2
    assert Tower(0, 0, 6, 9, TowerType.QUICK).attack_range == 1
    assert Tower(0, 0, 6, 9, TowerType.DOUBLE).attack_range == 3
    assert Tower(0, 0, 6, 9, TowerType.SNIPER).attack_range == 4
    assert Tower(0, 0, 6, 9, TowerType.HEAVY).damage == 12
    assert Tower(0, 0, 6, 9, TowerType.HEAVY_PLUS).damage == 24
    assert Tower(0, 0, 6, 9, TowerType.HEAVY_PLUS).attack_range == 1
    assert Tower(0, 0, 6, 9, TowerType.ICE).damage == 12
    assert Tower(0, 0, 6, 9, TowerType.BEWITCH).damage == 14
    assert Tower(0, 0, 6, 9, TowerType.BEWITCH).speed == 2.0
    assert Tower(0, 0, 6, 9, TowerType.QUICK).damage == 6
    assert Tower(0, 0, 6, 9, TowerType.QUICK_PLUS).damage == 6
    assert Tower(0, 0, 6, 9, TowerType.QUICK_PLUS).attack_range == 1
    assert Tower(0, 0, 6, 9, TowerType.DOUBLE).damage == 6
    assert Tower(0, 0, 6, 9, TowerType.DOUBLE).speed == 2.0
    assert Tower(0, 0, 6, 9, TowerType.MORTAR).damage == 12
    assert Tower(0, 0, 6, 9, TowerType.MORTAR_PLUS).damage == 18
    assert Tower(0, 0, 6, 9, TowerType.MORTAR_PLUS).attack_range == 2
    assert Tower(0, 0, 6, 9, TowerType.PULSE).damage == 14
    assert Tower(0, 0, 6, 9, TowerType.PULSE).speed == 4.0
    assert Tower(0, 0, 6, 9, TowerType.PULSE).attack_range == 2
    assert Tower(0, 0, 6, 9, TowerType.MISSILE).damage == 18
    assert Tower(0, 0, 6, 9, TowerType.MISSILE).attack_range == 3
    assert Tower(0, 0, 6, 9, TowerType.PRODUCER).stats.spawn_interval == 10
    assert Tower(0, 0, 6, 9, TowerType.PRODUCER_FAST).stats.spawn_interval == 8
    assert Tower(0, 0, 6, 9, TowerType.PRODUCER_SIEGE).stats.spawn_interval == 10
    assert Tower(0, 0, 6, 9, TowerType.PRODUCER_MEDIC).stats.spawn_interval == 10
    assert Tower(0, 0, 6, 9, TowerType.PRODUCER_MEDIC).stats.support_interval == 4


def test_super_weapon_stats_match_updated_spec() -> None:
    assert SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost == 90
    assert SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cooldown == 35
    assert SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].duration == 15
    assert SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost == 135
    assert SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cooldown == 45
    assert SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].duration == 10
    assert SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost == 60
    assert SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cooldown == 25
    assert SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost == 60
    assert SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cooldown == 25


def test_build_and_upgrade_tower_updates_coin_and_state() -> None:
    state = GameState.initial(seed=3)
    build = Operation(OperationType.BUILD_TOWER, 6, 9)
    assert state.can_apply_operation(0, build)
    assert state.apply_operation_list(0, [build]) == []
    assert state.coins[0] == INITIAL_COINS - 15
    tower = state.tower_at(6, 9)
    assert tower is not None
    upgrade = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(TowerType.HEAVY))
    state.coins[0] = 100
    assert state.can_apply_operation(0, upgrade)
    assert state.apply_operation_list(0, [upgrade]) == []
    assert tower.tower_type == TowerType.HEAVY
    assert tower.max_hp == 15
    assert tower.hp == 15
    assert state.coins[0] == 40


def test_strict_rule_illegal_mode_marks_offending_player_as_loser() -> None:
    state = GameState.initial(seed=31, cold_handle_rule_illegal=False)
    invalid = Operation(OperationType.BUILD_TOWER, 12, 9)

    resolution = state.resolve_turn([invalid], [])

    assert resolution.illegal[0] == [invalid]
    assert resolution.illegal[1] == []
    assert state.terminal is True
    assert state.winner == 1
    assert state.round_index == 0
    assert state.towers == []


def test_cold_rule_illegal_mode_can_skip_invalid_operation_without_ending_match() -> None:
    state = GameState.initial(seed=32, cold_handle_rule_illegal=True)
    invalid = Operation(OperationType.BUILD_TOWER, 12, 9)

    resolution = state.resolve_turn([invalid], [])

    assert resolution.illegal[0] == [invalid]
    assert resolution.illegal[1] == []
    assert state.terminal is False
    assert state.winner is None
    assert state.round_index == 1
    assert state.towers == []


def test_max_level_base_upgrade_returns_zero_income_without_crashing() -> None:
    state = GameState.initial(seed=12)
    state.bases[0].generation_level = 2
    state.bases[0].ant_level = 2

    gen_upgrade = Operation(OperationType.UPGRADE_GENERATION_SPEED)
    ant_upgrade = Operation(OperationType.UPGRADE_GENERATED_ANT)

    assert not state.can_apply_operation(0, gen_upgrade)
    assert not state.can_apply_operation(0, ant_upgrade)
    assert state.operation_income(0, gen_upgrade) == 0
    assert state.operation_income(0, ant_upgrade) == 0


def test_tower_refund_ratio_matches_new_spec() -> None:
    state = GameState.initial(seed=14)
    tower = Tower(0, 0, 6, 9, TowerType.BASIC, hp=7)
    assert state.destroy_tower_income(1, tower) == int(15 * TOWER_DOWNGRADE_REFUND_RATIO * 7 / 10)
    assert [state.build_tower_cost(i) for i in range(7)] == [15, 30, 45, 90, 135, 270, 405]
    assert state.destroy_tower_income(3, tower) == int(45 * TOWER_DOWNGRADE_REFUND_RATIO * 7 / 10)
    assert state.downgrade_tower_income(TowerType.HEAVY) == int(60 * TOWER_DOWNGRADE_REFUND_RATIO)


def test_quick_tower_attacks_enemy_ant() -> None:
    state = GameState.initial(seed=1)
    state.towers.append(Tower(0, 0, 6, 9, TowerType.QUICK, cooldown_clock=1.0))
    state.ants.append(Ant(0, 1, 7, 9, hp=10, level=0))
    state.advance_round()
    assert state.die_count[1] == 1 or any(ant.hp < 10 for ant in state.ants)


def test_emp_prevents_building_inside_field() -> None:
    state = GameState.initial(seed=1)
    state.active_effects.append(__import__('SDK.backend.model', fromlist=['WeaponEffect']).WeaponEffect(__import__('SDK.utils.constants', fromlist=['SuperWeaponType']).SuperWeaponType.EMP_BLASTER, 1, 6, 9, 3))
    blocked = Operation(OperationType.BUILD_TOWER, 6, 9)
    assert not state.can_apply_operation(0, blocked)


def test_random_ant_degrades_to_default_after_five_rounds() -> None:
    ant = Ant(0, 0, 2, 9, hp=10, level=0, behavior=AntBehavior.RANDOM)
    state = GameState.initial(seed=3)
    state.ants.append(ant)
    for _ in range(5):
        state._increase_ant_age()
    assert ant.behavior == AntBehavior.DEFAULT


def test_conservative_ant_degrades_to_default_after_five_rounds() -> None:
    ant = Ant(0, 0, 2, 9, hp=10, level=0)
    ant.set_behavior(AntBehavior.CONSERVATIVE)
    state = GameState.initial(seed=4)
    state.ants.append(ant)
    for _ in range(5):
        state._increase_ant_age()
    assert ant.behavior == AntBehavior.DEFAULT


def test_control_free_ant_degrades_to_default_after_five_rounds() -> None:
    ant = Ant(0, 0, 2, 9, hp=10, level=0)
    ant.set_behavior(AntBehavior.CONTROL_FREE)
    state = GameState.initial(seed=5)
    state.ants.append(ant)
    for _ in range(5):
        state._increase_ant_age()
    assert ant.behavior == AntBehavior.DEFAULT


def test_bewitched_ant_degrades_to_default_after_five_rounds() -> None:
    ant = Ant(0, 0, 2, 9, hp=10, level=0)
    state = GameState.initial(seed=6)
    state.ants.append(ant)
    state._control_ant(ant, AntBehavior.BEWITCHED, target=(16, 9))
    for _ in range(5):
        state._increase_ant_age()
    assert ant.behavior == AntBehavior.DEFAULT


def test_bewitch_targets_own_base_when_ant_is_in_own_half() -> None:
    state = GameState.initial(seed=17)
    ant = Ant(0, 0, 7, 9, hp=10, level=0)
    tower = Tower(0, 1, 12, 9, TowerType.BEWITCH)
    state._apply_tower_control(tower, ant)
    assert ant.behavior == AntBehavior.BEWITCHED
    assert (ant.bewitch_target_x, ant.bewitch_target_y) == PLAYER_BASES[0]


def test_bewitch_targets_backward_half_plane_when_ant_is_in_enemy_half() -> None:
    state = GameState.initial(seed=18)
    ant = Ant(0, 0, 11, 9, hp=10, level=0)
    tower = Tower(0, 1, 12, 9, TowerType.BEWITCH)
    state._apply_tower_control(tower, ant)
    target = (ant.bewitch_target_x, ant.bewitch_target_y)
    assert ant.behavior == AntBehavior.BEWITCHED
    assert target != (ant.x, ant.y)
    assert target in PATH_CELLS or target in PLAYER_BASES
    assert _half_plane_delta(ant.player, *target) <= _half_plane_delta(ant.player, ant.x, ant.y)


def test_pulse_hits_only_enemies_within_declared_range() -> None:
    state = GameState.initial(seed=19)
    tower = Tower(0, 0, 6, 9, TowerType.PULSE)
    near = Ant(0, 1, 8, 9, hp=20, level=0)
    far = Ant(1, 1, 9, 9, hp=20, level=0)
    state.ants.extend([near, far])
    assert state._tower_attack(tower)
    assert near.hp == 6
    assert near.behavior == AntBehavior.RANDOM
    assert far.hp == 20
    assert far.behavior == AntBehavior.DEFAULT


def test_mortar_hits_enemies_within_range_two_blast() -> None:
    state = GameState.initial(seed=20)
    tower = Tower(0, 0, 6, 9, TowerType.MORTAR)
    target = Ant(0, 1, 8, 9, hp=20, level=0)
    splash = Ant(1, 1, 10, 9, hp=20, level=0)
    outside = Ant(2, 1, 11, 9, hp=20, level=0)
    state.ants.extend([target, splash, outside])
    assert state._tower_attack(tower)
    assert target.hp == 8
    assert splash.hp == 8
    assert outside.hp == 20


def test_missile_hits_enemies_within_range_three_blast() -> None:
    state = GameState.initial(seed=21)
    tower = Tower(0, 0, 6, 9, TowerType.MISSILE)
    target = Ant(0, 1, 9, 9, hp=30, level=0)
    splash = Ant(1, 1, 12, 9, hp=30, level=0)
    outside = Ant(2, 1, 13, 9, hp=30, level=0)
    state.ants.extend([target, splash, outside])
    assert state._tower_attack(tower)
    assert target.hp == 12
    assert splash.hp == 12
    assert outside.hp == 30


def test_ice_freeze_promotes_ant_to_random_after_thaw() -> None:
    state = GameState.initial(seed=2)
    ant = Ant(0, 1, 7, 9, hp=25, level=1, behavior=AntBehavior.CONSERVATIVE)
    tower = Tower(0, 0, 6, 9, TowerType.ICE, cooldown_clock=0.0)
    state.ants.append(ant)
    state._damage_ant_from_tower(tower, ant)
    assert ant.frozen
    state._prepare_ants_for_attack()
    assert ant.behavior == AntBehavior.RANDOM


def test_control_free_ant_ignores_control_and_random_move_phase() -> None:
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


def test_move_progress_score_penalizes_stalling_relative_to_advancing() -> None:
    state = GameState.initial(seed=13)
    target_x, target_y = PLAYER_BASES[1]
    advance_score = stall_score = None
    for x, y in PATH_CELLS:
        ant = Ant(0, 0, x, y, hp=10, level=0, behavior=AntBehavior.DEFAULT)
        current_distance = hex_distance(ant.x, ant.y, target_x, target_y)
        local_advance = None
        local_stall = None
        for _, nx, ny in neighbors(ant.x, ant.y):
            if (nx, ny) != (target_x, target_y) and not is_path(nx, ny):
                continue
            next_distance = hex_distance(nx, ny, target_x, target_y)
            score = state._move_progress_score(ant, nx, ny, target_x, target_y)
            if next_distance < current_distance and local_advance is None:
                local_advance = score
            if next_distance == current_distance and local_stall is None:
                local_stall = score
        if local_advance is not None and local_stall is not None:
            advance_score = local_advance
            stall_score = local_stall
            break
    assert advance_score is not None
    assert stall_score is not None
    assert stall_score < 0.0
    assert advance_score > stall_score


def test_directional_damage_field_penalizes_lane_toward_future_tower_fire() -> None:
    state = GameState.initial(seed=15)
    ant = Ant(0, 0, 9, 9, hp=10, level=0, behavior=AntBehavior.CONSERVATIVE)
    state.towers.append(Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0))
    candidates = state._move_candidates(ant, allow_backtrack=False)
    east_index = next(index for index, (direction, _, _) in enumerate(candidates) if direction == 4)
    west_index = next(index for index, (direction, _, _) in enumerate(candidates) if direction == 1)
    state._refresh_static_risk_fields()
    assert state.damage_risk_field[0, 10, 9] == 0.0
    damage_scores = state._directional_field_scores(ant, candidates, state.damage_risk_field)
    assert damage_scores[east_index] > damage_scores[west_index]


def test_directional_control_field_penalizes_lane_toward_future_control_zone() -> None:
    state = GameState.initial(seed=16)
    ant = Ant(0, 0, 9, 9, hp=10, level=0, behavior=AntBehavior.CONSERVATIVE)
    state.towers.append(Tower(0, 1, 12, 9, TowerType.ICE, cooldown_clock=2.0))
    candidates = state._move_candidates(ant, allow_backtrack=False)
    east_index = next(index for index, (direction, _, _) in enumerate(candidates) if direction == 4)
    west_index = next(index for index, (direction, _, _) in enumerate(candidates) if direction == 1)
    state._refresh_static_risk_fields()
    assert state.control_risk_field[0, 10, 9] == 1.0
    control_scores = state._directional_field_scores(ant, candidates, state.control_risk_field)
    assert control_scores[east_index] > control_scores[west_index]


def test_legacy_pathfinding_avoids_lightning_storm_lane() -> None:
    state = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_LEGACY)
    ant = Ant(0, 0, 2, 2, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    state.ants.append(ant)
    baseline = state._choose_ant_move(ant)

    threatened = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_LEGACY)
    threatened_ant = Ant(0, 0, 2, 2, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    threatened.ants.append(threatened_ant)
    threatened.active_effects.append(WeaponEffect(SuperWeaponType.LIGHTNING_STORM, 1, 3, 0, 15))

    assert baseline == 4
    assert threatened._choose_ant_move(threatened_ant) == 5


def test_legacy_pathfinding_prefers_deflector_and_evasion_zones() -> None:
    base_state = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_LEGACY)
    ant = Ant(0, 0, 8, 9, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    base_state.ants.append(ant)
    baseline = base_state._choose_ant_move(ant)

    assert baseline == 3
    for weapon_type, duration in (
        (SuperWeaponType.DEFLECTOR, 10),
        (SuperWeaponType.EMERGENCY_EVASION, 1),
    ):
        guided = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_LEGACY)
        guided_ant = Ant(0, 0, 8, 9, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
        guided.ants.append(guided_ant)
        guided.active_effects.append(WeaponEffect(weapon_type, 0, 11, 10, duration))
        assert guided._choose_ant_move(guided_ant) == 5


def test_default_ant_prefers_advancing_move_on_clear_path() -> None:
    advancing = 0
    for seed in range(16):
        state = GameState.initial(seed=seed, movement_policy=MOVEMENT_POLICY_LEGACY)
        ant = Ant(0, 0, 7, 9, hp=10, level=0, behavior=AntBehavior.DEFAULT)
        before = hex_distance(ant.x, ant.y, *PLAYER_BASES[1])
        state._resolve_ant_step(ant, state._choose_ant_move(ant))
        after = hex_distance(ant.x, ant.y, *PLAYER_BASES[1])
        if after < before:
            advancing += 1
    assert advancing >= 12


def test_combat_ant_targets_nearest_enemy_tower_before_base() -> None:
    state = GameState.initial(seed=17, movement_policy=MOVEMENT_POLICY_LEGACY)
    ant = Ant(0, 0, 8, 9, hp=30, level=0, kind=AntKind.COMBAT, behavior=AntBehavior.CONSERVATIVE)
    state.ants.append(ant)
    state.towers.extend(
        [
            Tower(0, 1, 8, 6, TowerType.BASIC, cooldown_clock=2.0),
            Tower(1, 1, 14, 9, TowerType.BASIC, cooldown_clock=2.0),
        ]
    )
    target = state._move_target_for_ant(ant)
    before = hex_distance(ant.x, ant.y, *target)
    state._resolve_ant_step(ant, state._choose_ant_move(ant))
    after = hex_distance(ant.x, ant.y, *target)
    assert target == (8, 6)
    assert after < before


def test_worker_ant_prefers_safe_advancing_lane_over_adjacent_tower_attack() -> None:
    state = GameState.initial(seed=18, movement_policy=MOVEMENT_POLICY_LEGACY)
    ant = Ant(0, 0, 11, 8, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    state.ants.append(ant)
    state.towers.append(Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0))
    chosen = state._choose_ant_move(ant)
    assert chosen == 4


def test_worker_ant_attacks_adjacent_tower_when_pushing_forward_is_riskier() -> None:
    state = GameState.initial(seed=19, movement_policy=MOVEMENT_POLICY_LEGACY)
    ant = Ant(0, 0, 11, 9, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    state.ants.append(ant)
    state.towers.extend(
        [
            Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0),
            Tower(1, 1, 14, 9, TowerType.BASIC, cooldown_clock=2.0),
        ]
    )
    chosen = state._choose_ant_move(ant)
    assert chosen == 4


def test_enhanced_worker_reservations_split_over_equivalent_frontline_lanes() -> None:
    state = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_ENHANCED)
    first = Ant(100, 0, 8, 9, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    second = Ant(101, 0, 8, 9, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    state.ants.extend([first, second])

    state._move_ants()

    assert (first.x, first.y) == (8, 8)
    assert (second.x, second.y) == (8, 10)


def test_enhanced_combat_ant_prefers_flanking_path_over_stack_of_tower_fire() -> None:
    state = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_ENHANCED)
    ant = Ant(0, 0, 8, 9, hp=30, level=0, kind=AntKind.COMBAT, behavior=AntBehavior.CONSERVATIVE)
    state.ants.append(ant)
    state.towers.extend(
        [
            Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0),
            Tower(1, 1, 14, 9, TowerType.HEAVY, cooldown_clock=2.0),
        ]
    )

    chosen = state._choose_ant_move(ant)

    assert chosen == 3


def test_enhanced_pathfinding_avoids_lightning_storm_lane() -> None:
    state = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_ENHANCED)
    ant = Ant(0, 0, 9, 9, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    state.ants.append(ant)
    baseline = state._choose_ant_move(ant)

    threatened = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_ENHANCED)
    threatened_ant = Ant(0, 0, 9, 9, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    threatened.ants.append(threatened_ant)
    threatened.active_effects.append(WeaponEffect(SuperWeaponType.LIGHTNING_STORM, 1, 12, 9, 15))

    assert baseline == 4
    assert threatened._choose_ant_move(threatened_ant) == 1


def test_enhanced_pathfinding_prefers_deflector_and_evasion_zones() -> None:
    base_state = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_ENHANCED)
    ant = Ant(0, 0, 4, 6, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    base_state.ants.append(ant)
    baseline = base_state._choose_ant_move(ant)

    assert baseline == 3
    for weapon_type, center in (
        (SuperWeaponType.DEFLECTOR, (8, 10)),
        (SuperWeaponType.EMERGENCY_EVASION, (6, 10)),
    ):
        guided = GameState.initial(seed=1, movement_policy=MOVEMENT_POLICY_ENHANCED)
        guided_ant = Ant(0, 0, 4, 6, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
        guided.ants.append(guided_ant)
        duration = 10 if weapon_type == SuperWeaponType.DEFLECTOR else 1
        guided.active_effects.append(WeaponEffect(weapon_type, 0, center[0], center[1], duration))
        assert guided._choose_ant_move(guided_ant) == 0


def test_enhanced_default_combat_ant_adjacent_tower_attack_rate_is_about_eighty_percent() -> None:
    attack_rate = _sample_adjacent_tower_attack_rate(
        seeds=200,
        kind=AntKind.COMBAT,
        behavior=AntBehavior.DEFAULT,
    )

    assert 0.76 <= attack_rate <= 0.84


def test_enhanced_default_worker_adjacent_single_tower_attack_rate_stays_low() -> None:
    attack_rate = _sample_adjacent_tower_attack_rate(
        seeds=200,
        kind=AntKind.WORKER,
        behavior=AntBehavior.DEFAULT,
    )

    assert attack_rate <= 0.05


def test_enhanced_default_combat_adjacent_tower_attack_rate_causes_real_tower_damage() -> None:
    attack_rate = _sample_adjacent_tower_attack_resolution_rate(
        seeds=120,
        kind=AntKind.COMBAT,
        behavior=AntBehavior.DEFAULT,
    )

    assert 0.76 <= attack_rate <= 0.84


def test_enhanced_conservative_combat_ant_adjacent_tower_damages_it_without_moving() -> None:
    state = GameState.initial(seed=0, movement_policy=MOVEMENT_POLICY_ENHANCED)
    tower = Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0, hp=10)
    ant = Ant(24, 0, 11, 9, hp=30, level=0, kind=AntKind.COMBAT, behavior=AntBehavior.CONSERVATIVE)
    state.towers.append(tower)
    state.ants.append(ant)

    state.advance_round()

    tracked_ant = next(item for item in state.ants if item.ant_id == ant.ant_id)
    tracked_tower = next(item for item in state.towers if item.tower_id == tower.tower_id)
    assert (tracked_ant.x, tracked_ant.y) == (11, 9)
    assert tracked_tower.hp == 5


def test_enhanced_conservative_worker_prefers_reroute_when_single_tower_is_avoidable() -> None:
    state = GameState.initial(seed=0, movement_policy=MOVEMENT_POLICY_ENHANCED)
    ant = Ant(0, 0, 11, 8, hp=20, level=0, behavior=AntBehavior.CONSERVATIVE)
    state.ants.append(ant)
    state.towers.append(Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0))

    chosen = state._choose_ant_move(ant)

    assert chosen == 4


def test_random_move_phase_resolves_three_steps_for_selected_ant() -> None:
    state = GameState.initial(seed=10)
    ant = Ant(0, 0, 4, 9, hp=10, level=0, behavior=AntBehavior.DEFAULT)
    state.ants.append(ant)

    original_random_index = GameState._random_index
    original_choose_random_legal_move = GameState._choose_random_legal_move
    random_index_values = iter([0, 0, 0, 0])
    random_move_calls = {"count": 0}

    def fake_random_index(self, bound: int) -> int:
        value = next(random_index_values)
        assert 0 <= value < bound
        return value

    def fake_choose_random_legal_move(self, moving_ant: Ant) -> int:
        random_move_calls["count"] += 1
        return 4

    GameState._random_index = fake_random_index
    GameState._choose_random_legal_move = fake_choose_random_legal_move
    try:
        state.round_index = ANT_TELEPORT_INTERVAL - 1
        state._teleport_ants()
    finally:
        GameState._random_index = original_random_index
        GameState._choose_random_legal_move = original_choose_random_legal_move

    assert random_move_calls["count"] == 3
    assert ant.path_len_total == 3
    assert (ant.x, ant.y) == (7, 9)
    assert ant.trail_cells[-3:] == [(5, 9), (6, 9), (7, 9)]


def test_random_move_steps_recompute_algorithmic_path_after_each_step() -> None:
    state = GameState.initial(seed=11)
    ant = Ant(0, 0, 4, 9, hp=10, level=0, behavior=AntBehavior.DEFAULT)
    state.ants.append(ant)

    original_random_index = GameState._random_index
    original_choose_ant_move = GameState._choose_ant_move
    original_invalidate_cache = GameState._invalidate_enhanced_move_cache
    random_index_values = iter([2, 2, 2])
    seen_positions: list[tuple[int, int]] = []
    invalidation_count = {"count": 0}

    def fake_random_index(self, bound: int) -> int:
        value = next(random_index_values)
        assert 0 <= value < bound
        return value

    def fake_choose_ant_move(self, moving_ant: Ant) -> int:
        seen_positions.append((moving_ant.x, moving_ant.y))
        return 4

    def fake_invalidate_enhanced_move_cache(self) -> None:
        invalidation_count["count"] += 1

    GameState._random_index = fake_random_index
    GameState._choose_ant_move = fake_choose_ant_move
    GameState._invalidate_enhanced_move_cache = fake_invalidate_enhanced_move_cache
    try:
        state._resolve_random_move_steps(ant)
    finally:
        GameState._random_index = original_random_index
        GameState._choose_ant_move = original_choose_ant_move
        GameState._invalidate_enhanced_move_cache = original_invalidate_cache

    assert seen_positions == [(4, 9), (5, 9), (6, 9)]
    assert invalidation_count["count"] == 3
    assert ant.path_len_total == 3
    assert (ant.x, ant.y) == (7, 9)


def test_spawn_profile_weights_match_spec() -> None:
    weights = {(kind, behavior): probability for kind, behavior, probability in SPAWN_PROFILE_WEIGHTS}
    assert weights[(AntKind.WORKER, AntBehavior.DEFAULT)] == 0.4
    assert weights[(AntKind.WORKER, AntBehavior.CONSERVATIVE)] == 0.35
    assert weights[(AntKind.WORKER, AntBehavior.RANDOM)] == 0.10
    assert weights[(AntKind.COMBAT, AntBehavior.DEFAULT)] == 0.15
    assert ANT_TELEPORT_RATIO == 0.1


def test_combat_ant_kill_reward_is_fixed() -> None:
    worker = Ant(0, 0, 2, 9, hp=20, level=2)
    combat = Ant(1, 1, 16, 9, hp=30, level=0, kind=AntKind.COMBAT)
    elite_combat = Ant(2, 1, 16, 9, hp=30, level=2, kind=AntKind.COMBAT)

    assert worker.kill_reward == 14
    assert combat.kill_reward == COMBAT_ANT_KILL_REWARD
    assert elite_combat.kill_reward == COMBAT_ANT_KILL_REWARD


def test_natural_spawn_can_create_combat_ant_with_shield() -> None:
    state = GameState.initial(seed=10)
    original = GameState._draw_spawn_profile
    GameState._draw_spawn_profile = lambda self: (AntKind.COMBAT, AntBehavior.DEFAULT)
    try:
        state._spawn_ants()
    finally:
        GameState._draw_spawn_profile = original
    assert len(state.ants) == 2
    assert all(ant.kind == AntKind.COMBAT for ant in state.ants)
    assert all(ant.behavior == AntBehavior.DEFAULT for ant in state.ants)
    assert all(ant.hp == 30 for ant in state.ants)
    assert all(ant.shield == 3 for ant in state.ants)


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


def test_lightning_storm_damages_enemy_worker_and_combat_ants() -> None:
    state = GameState.initial(seed=12)
    state.ants = [
        Ant(1, 1, 9, 9, hp=25, level=1),
        Ant(2, 1, 10, 9, hp=30, level=0, kind=AntKind.COMBAT),
        Ant(3, 0, 9, 9, hp=20, level=0),
    ]
    state.active_effects = [WeaponEffect(SuperWeaponType.LIGHTNING_STORM, 0, 9, 9, 15)]

    state._apply_lightning_storm()

    ants = {ant.ant_id: ant for ant in state.ants}
    assert ants[1].hp == 5
    assert ants[2].hp == 10
    assert ants[3].hp == 20


def test_lightning_storm_triggers_immediately_when_deployed() -> None:
    state = GameState.initial(seed=13)
    state.ants = [
        Ant(1, 1, 9, 9, hp=25, level=1),
        Ant(2, 1, 10, 9, hp=30, level=0, kind=AntKind.COMBAT),
    ]

    state.apply_operation(0, Operation(OperationType.USE_LIGHTNING_STORM, 9, 9))

    ants = {ant.ant_id: ant for ant in state.ants}
    assert ants[1].hp == 5
    assert ants[2].hp == 10
    assert len(state.active_effects) == 1
    assert state.active_effects[0].last_trigger_round == 0


def test_lightning_storm_consumes_evasion_shield_before_hp() -> None:
    state = GameState.initial(seed=16)
    ant = Ant(1, 1, 9, 9, hp=25, level=1)
    ant.grant_evasion(2, grant_control_free_on_deplete=True)
    state.ants = [ant]

    state.apply_operation(0, Operation(OperationType.USE_LIGHTNING_STORM, 9, 9))

    assert state.ants[0].hp == 25
    assert state.ants[0].shield == 1


def test_new_lightning_storm_does_not_double_tick_in_same_round() -> None:
    state = GameState.initial(seed=15)
    state.ants = [Ant(1, 1, 9, 9, hp=25, level=1)]

    state.apply_operation(0, Operation(OperationType.USE_LIGHTNING_STORM, 9, 9))
    state._apply_lightning_storm()

    assert state.ants[0].hp == 5


def test_lightning_storm_only_hits_enemy_towers_on_every_fifth_active_turn() -> None:
    state = GameState.initial(seed=14)
    state.towers = [
        Tower(1, 1, 9, 9, TowerType.PRODUCER, hp=15),
        Tower(2, 1, 13, 9, TowerType.PRODUCER, hp=15),
        Tower(3, 0, 9, 9, TowerType.PRODUCER, hp=15),
    ]
    state.active_effects = [WeaponEffect(SuperWeaponType.LIGHTNING_STORM, 0, 9, 9, 12)]

    state._apply_lightning_storm()
    towers = {tower.tower_id: tower for tower in state.towers}
    assert towers[1].hp == 15
    assert towers[2].hp == 15
    assert towers[3].hp == 15

    state.active_effects[0].remaining_turns = 11
    state.active_effects[0].last_trigger_round = -1
    state._apply_lightning_storm()
    towers = {tower.tower_id: tower for tower in state.towers}
    assert towers[1].hp == 12
    assert towers[2].hp == 15
    assert towers[3].hp == 15


def test_public_round_state_serializes_true_age() -> None:
    state = GameState.initial(seed=5)
    state.ants.append(
        Ant(
            7,
            0,
            4,
            9,
            hp=10,
            level=0,
            age=12,
            trail_cells=[(2, 9), (3, 9), (4, 9)],
            last_move=4,
            path_len_total=2,
            status=AntStatus.ALIVE,
            behavior=AntBehavior.CONTROL_FREE,
        )
    )
    public_state = state.to_public_round_state()
    assert public_state.ants[0][6] == 12
    assert public_state.ants[0][8] == int(AntBehavior.CONTROL_FREE)


def test_sync_public_round_state_updates_visible_age_and_syncs_public_behavior() -> None:
    state = GameState.initial(seed=3)
    ant = Ant(
        8,
        0,
        4,
        9,
        hp=10,
        level=0,
        age=9,
        trail_cells=[(2, 9), (3, 9), (4, 9)],
        last_move=1,
        path_len_total=3,
        behavior=AntBehavior.RANDOM,
    )
    state.ants.append(ant)
    public_state = PublicRoundState(
        round_index=0,
        towers=[],
        ants=[(8, 0, 5, 9, 8, 0, 5, 0, int(AntBehavior.CONSERVATIVE))],
        coins=(50, 50),
        camps_hp=(50, 50),
    )
    state.sync_public_round_state(public_state)
    synced = state.ants[0]
    assert synced.age == 5
    assert synced.behavior == AntBehavior.CONSERVATIVE
    assert synced.behavior_turns == 0
    assert synced.behavior_expiry == SPECIAL_BEHAVIOR_DECAY_TURNS
    assert synced.trail_cells == [(2, 9), (3, 9), (4, 9)]
    assert synced.last_move == 1
    assert synced.path_len_total == 3


def test_sync_public_round_state_maps_frozen_status_to_hidden_flag() -> None:
    state = GameState.initial(seed=6)
    ant = Ant(9, 0, 4, 9, hp=10, level=0, age=2, frozen=False, pending_behavior=AntBehavior.RANDOM)
    state.ants.append(ant)
    public_state = PublicRoundState(
        round_index=0,
        towers=[],
        ants=[(9, 0, 4, 9, 10, 0, 3, int(AntStatus.FROZEN), int(AntBehavior.DEFAULT))],
        coins=(50, 50),
        camps_hp=(50, 50),
    )
    state.sync_public_round_state(public_state)
    synced = state.ants[0]
    assert synced.status == AntStatus.FROZEN
    assert synced.frozen is True
    assert synced.pending_behavior == AntBehavior.RANDOM


def test_sync_public_round_state_applies_public_fields() -> None:
    state = GameState.initial(seed=8)
    public_state = PublicRoundState(
        round_index=3,
        towers=[(5, 1, 12, 9, int(TowerType.BASIC), 2, 7)],
        ants=[(11, 0, 4, 9, 10, 0, 6, int(AntStatus.ALIVE), int(AntBehavior.DEFAULT), int(AntKind.COMBAT))],
        coins=(61, 44),
        camps_hp=(49, 50),
        speed_lv=(2, 1),
        anthp_lv=(1, 2),
        weapon_cooldowns=((9, 8, 7, 6), (1, 2, 3, 4)),
        active_effects=[(int(SuperWeaponType.EMP_BLASTER), 1, 8, 9, 5)],
    )
    state.sync_public_round_state(public_state)
    assert state.round_index == 3
    assert state.towers[0].tower_id == 5
    assert state.towers[0].hp == 7
    assert state.ants[0].ant_id == 11
    assert state.ants[0].kind == AntKind.COMBAT
    assert state.bases[0].generation_level == 2
    assert state.bases[1].generation_level == 1
    assert state.bases[0].ant_level == 1
    assert state.bases[1].ant_level == 2
    assert tuple(int(state.weapon_cooldowns[0, weapon_type]) for weapon_type in SuperWeaponType) == (9, 8, 7, 6)
    assert tuple(int(state.weapon_cooldowns[1, weapon_type]) for weapon_type in SuperWeaponType) == (1, 2, 3, 4)
    assert len(state.active_effects) == 1
    assert state.active_effects[0].weapon_type == SuperWeaponType.EMP_BLASTER
    assert state.active_effects[0].remaining_turns == 5


def test_update_pheromone_walks_backwards_from_current_position() -> None:
    state = GameState.initial(seed=1)
    ant = Ant(
        3,
        0,
        6,
        9,
        hp=0,
        level=0,
        age=4,
        trail_cells=[(5, 9), (6, 9)],
        last_move=4,
        path_len_total=1,
        status=AntStatus.FAIL,
    )
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


def test_teleport_keeps_trail_for_pheromone_and_resets_last_move() -> None:
    state = GameState.initial(seed=1)
    ant = Ant(
        13,
        0,
        4,
        9,
        hp=0,
        level=0,
        age=4,
        trail_cells=[(2, 9), (3, 9), (4, 9)],
        last_move=4,
        path_len_total=2,
        status=AntStatus.FAIL,
    )
    ant.teleport_to(10, 9)
    state.ants.append(ant)
    before_origin = int(state.pheromone[0, 4, 9])
    before_target = int(state.pheromone[0, 10, 9])
    state._update_pheromone()
    attenuated_origin = max(0, (LAMBDA_NUM * before_origin + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM)
    attenuated_target = max(0, (LAMBDA_NUM * before_target + TAU_BASE_ADD_INT + 50) // LAMBDA_DENOM)
    assert ant.last_move == -1
    assert int(state.pheromone[0, 4, 9]) == attenuated_origin + PHEROMONE_FAIL_BONUS_INT
    assert int(state.pheromone[0, 10, 9]) == attenuated_target + PHEROMONE_FAIL_BONUS_INT


def test_path_len_total_counts_no_move_but_not_teleport() -> None:
    ant = Ant(14, 0, 2, 9, hp=10, level=0)
    ant.record_move(-1)
    assert ant.path_len_total == 1
    assert ant.last_move == -1
    assert ant.trail_cells == [(2, 9)]
    ant.teleport_to(9, 9)
    assert ant.path_len_total == 1
    assert ant.last_move == -1
    assert ant.trail_cells[-1] == (9, 9)


def test_too_old_ants_remain_visible_until_next_lifecycle_cleanup() -> None:
    state = GameState.initial(seed=4)
    ant = Ant(11, 0, 2, 9, hp=10, level=0, age=ANT_AGE_LIMIT, kind=AntKind.WORKER)
    state.ants.append(ant)
    state.advance_round()
    tracked = next(item for item in state.ants if item.ant_id == 11)
    assert tracked.status.name == "TOO_OLD"
    assert state.old_count == [0, 0]
    state.advance_round()
    assert all(item.ant_id != 11 for item in state.ants)
    assert state.old_count == [1, 0]


def test_combat_ants_do_not_die_of_old_age() -> None:
    state = GameState.initial(seed=22)
    ant = Ant(15, 0, 2, 9, hp=30, level=0, age=ANT_AGE_LIMIT + 20, kind=AntKind.COMBAT)
    state.ants.append(ant)
    state.advance_round()
    tracked = next(item for item in state.ants if item.ant_id == 15)
    assert tracked.status != AntStatus.TOO_OLD
    assert state.old_count == [0, 0]


def test_terminal_round_stops_before_spawn_and_income() -> None:
    state = GameState.initial(seed=8)
    state.bases[1].hp = 1
    state.ants.append(Ant(12, 0, 16, 9, hp=10, level=0))
    state.advance_round()
    assert state.terminal is True
    assert state.winner == 0
    assert state.round_index == 1
    assert state.coins == [INITIAL_COINS + 10, INITIAL_COINS]
    assert state.ants == []


def test_worker_ant_virtual_attack_damages_tower_without_moving() -> None:
    state = GameState.initial(seed=2)
    tower = Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0, hp=10)
    ant = Ant(21, 0, 11, 9, hp=10, level=0)
    state.towers.append(tower)
    state.ants.append(ant)
    direction = direction_between(ant.x, ant.y, tower.x, tower.y)
    state._resolve_ant_step(ant, direction)
    assert (ant.x, ant.y) == (11, 9)
    assert ant.last_move == -1
    assert tower.hp == 9


def test_combat_ant_at_half_hp_uses_base_attack_without_self_destruct() -> None:
    state = GameState.initial(seed=3)
    tower = Tower(0, 1, 12, 9, TowerType.BASIC, cooldown_clock=2.0, hp=10)
    ant = Ant(22, 0, 11, 9, hp=15, level=0, kind=AntKind.COMBAT)
    ant.grant_evasion(2, grant_control_free_on_deplete=True)
    state.towers.append(tower)
    state.ants.append(ant)
    direction = direction_between(ant.x, ant.y, tower.x, tower.y)
    state._resolve_ant_step(ant, direction)
    assert ant.hp == 15
    assert ant.status == AntStatus.ALIVE
    assert tower.hp == 5
    assert len(state.towers) == 1


def test_combat_ant_self_destruct_damages_target_and_neighboring_towers() -> None:
    state = GameState.initial(seed=4)
    target = Tower(0, 1, 14, 9, TowerType.BASIC, cooldown_clock=2.0, hp=10)
    nearby = Tower(1, 1, 14, 10, TowerType.BASIC, cooldown_clock=2.0, hp=12)
    ant = Ant(23, 0, 13, 10, hp=14, level=0, kind=AntKind.COMBAT)
    state.towers.extend([target, nearby])
    state.ants.append(ant)
    direction = direction_between(ant.x, ant.y, target.x, target.y)
    state._resolve_ant_step(ant, direction)
    assert ant.hp <= 0
    assert state.tower_by_id(target.tower_id) is None
    remaining = state.tower_by_id(nearby.tower_id)
    assert remaining is not None
    assert remaining.hp == 2


def test_enhanced_conservative_low_hp_combat_ant_self_destructs_on_adjacent_tower_cluster() -> None:
    state = GameState.initial(seed=4, movement_policy=MOVEMENT_POLICY_ENHANCED)
    target = Tower(0, 1, 14, 9, TowerType.BASIC, cooldown_clock=2.0, hp=10)
    nearby = Tower(1, 1, 14, 10, TowerType.BASIC, cooldown_clock=2.0, hp=12)
    ant = Ant(23, 0, 13, 10, hp=14, level=0, kind=AntKind.COMBAT, behavior=AntBehavior.CONSERVATIVE)
    state.towers.extend([target, nearby])
    state.ants.append(ant)

    state.advance_round()

    assert state.die_count[0] == 1
    assert all(item.ant_id != ant.ant_id for item in state.ants)
    assert state.tower_by_id(target.tower_id) is None
    remaining = state.tower_by_id(nearby.tower_id)
    assert remaining is not None
    assert remaining.hp == 2


def test_producer_tower_can_spawn_combat_ant_with_profile() -> None:
    state = GameState.initial(seed=5)
    tower = Tower(0, 0, 6, 9, TowerType.PRODUCER_SIEGE, cooldown_clock=0.0)
    state.towers.append(tower)
    state._spawn_ant_from_tower(tower, AntKind.COMBAT, AntBehavior.DEFAULT)
    assert len(state.ants) == 1
    spawned = state.ants[0]
    assert spawned.kind == AntKind.COMBAT
    assert spawned.hp == 30
    assert spawned.shield == 3
    assert spawned.move_weights.tower_pull > 0


def test_medic_support_prefers_combat_ant_in_frontline_two_rows_and_fully_heals() -> None:
    state = GameState.initial(seed=8)
    state.round_index = 1
    tower = Tower(0, 0, 6, 9, TowerType.PRODUCER_MEDIC, cooldown_clock=5.0)
    frontline_worker = Ant(40, 0, 14, 9, hp=1, level=0)
    frontline_combat = Ant(41, 0, 13, 9, hp=7, level=0, kind=AntKind.COMBAT)
    frontline_combat.shield = 1
    backline_combat = Ant(42, 0, 11, 9, hp=2, level=0, kind=AntKind.COMBAT)
    state.towers.append(tower)
    state.ants.extend([frontline_worker, frontline_combat, backline_combat])
    state._spawn_ants()
    assert frontline_combat.hp == frontline_combat.max_hp
    assert frontline_combat.shield == 2
    assert frontline_worker.hp == 1
    assert backline_combat.hp == 2


def test_public_round_state_exposes_visible_runtime_fields() -> None:
    state = GameState.initial(seed=9)
    state.towers.append(Tower(0, 0, 6, 9, TowerType.BASIC, cooldown_clock=1.0, hp=7))
    state.ants.append(Ant(30, 0, 2, 9, hp=10, level=0, kind=AntKind.COMBAT))
    state.bases[0].generation_level = 1
    state.bases[1].ant_level = 2
    state.weapon_cooldowns[0, SuperWeaponType.LIGHTNING_STORM] = 12
    state.weapon_cooldowns[1, SuperWeaponType.EMP_BLASTER] = 5
    state.active_effects = [WeaponEffect(SuperWeaponType.DEFLECTOR, 0, 6, 9, 4)]
    public_state = state.to_public_round_state()
    assert len(public_state.towers[0]) == 7
    assert public_state.towers[0] == (0, 0, 6, 9, int(TowerType.BASIC), 1, 7)
    assert len(public_state.ants[0]) == 10
    assert public_state.ants[0] == (30, 0, 2, 9, 10, 0, 0, int(AntStatus.ALIVE), int(AntBehavior.DEFAULT), int(AntKind.COMBAT))
    assert public_state.speed_lv == (1, 0)
    assert public_state.anthp_lv == (0, 2)
    assert public_state.weapon_cooldowns == ((12, 0, 0, 0), (0, 5, 0, 0))
    assert public_state.active_effects == [(int(SuperWeaponType.DEFLECTOR), 0, 6, 9, 4)]
