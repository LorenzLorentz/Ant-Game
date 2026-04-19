from __future__ import annotations

from pathlib import Path

import AI.ai_greedy as greedy_module
from AI.ai_greedy import AI as GreedyAI, _to_greedy_info, _to_sdk_operation
from AI.ai_mcts import MCTSAgent
from AI.ai_random import RandomAgent
from SDK.utils.actions import ActionCatalog
from SDK.backend import load_backend
from SDK.utils.features import FeatureExtractor
from SDK.utils.constants import COMBAT_ANT_KILL_REWARD, AntBehavior, AntKind, AntStatus, OperationType, SuperWeaponType, TowerType
from SDK.backend.engine import GameState, PublicRoundState
from SDK.backend.forecast import Ant as ForecastAnt, AntState as ForecastAntState, ForecastSimulator, ForecastState, Operation as ForecastOperation
from SDK.backend.model import Ant, Operation, Tower


def test_action_catalog_returns_legal_bundles() -> None:
    state = GameState.initial(seed=11)
    catalog = ActionCatalog(max_actions=32)
    bundles = catalog.build(state, 0)
    assert bundles
    assert bundles[0].name
    for bundle in bundles[:10]:
        accepted = []
        for operation in bundle.operations:
            assert state.can_apply_operation(0, operation, accepted)
            accepted.append(operation)


def test_random_agent_selects_non_empty_legal_bundle() -> None:
    state = GameState.initial(seed=5)
    agent = RandomAgent(seed=5)
    bundles = agent.list_bundles(state, 0)
    bundle = agent.choose_bundle(state, 0, bundles=bundles)
    assert bundle in bundles


def test_action_catalog_tolerates_stale_ant_trails() -> None:
    state = GameState.initial(seed=5)
    state.ants.append(
        Ant(
            99,
            1,
            17,
            9,
            hp=10,
            level=0,
            age=32,
            trail_cells=[(16, 9), (17, 9)],
            last_move=4,
            path_len_total=3,
        )
    )
    catalog = ActionCatalog(max_actions=16)
    bundles = catalog.build(state, 0)
    assert bundles


def test_action_catalog_tolerates_base_upgrade_pairing_without_crashing() -> None:
    state = GameState.initial(seed=21)
    state.coins[0] = 300
    state.bases[0].ant_level = 1

    catalog = ActionCatalog(max_actions=32)
    bundles = catalog.build(state, 0)

    assert bundles


def test_action_catalog_can_offer_storm_against_enemy_towers_without_enemy_ants() -> None:
    state = GameState.initial(seed=24)
    state.coins[0] = 200
    state.towers.append(Tower(5, 1, 10, 9, TowerType.PRODUCER, hp=15))

    catalog = ActionCatalog(max_actions=32)
    bundles = catalog._superweapon_candidates(state, 0)

    assert any(
        bundle.operations
        and bundle.operations[0].op_type == OperationType.USE_LIGHTNING_STORM
        for bundle in bundles
    )


def test_action_catalog_skips_max_level_base_upgrades() -> None:
    state = GameState.initial(seed=6)
    state.coins[0] = 9999
    state.bases[0].generation_level = 2
    state.bases[0].ant_level = 2

    catalog = ActionCatalog(max_actions=64)
    bundles = catalog.build(state, 0)

    assert bundles
    assert all(
        op.op_type not in (OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT)
        for bundle in bundles
        for op in bundle.operations
    )


def test_action_catalog_offers_generation_upgrade_when_next_level_improves_cadence() -> None:
    state = GameState.initial(seed=18)
    state.coins[0] = 9999
    state.bases[0].generation_level = 0

    catalog = ActionCatalog(max_actions=64)
    bundles = catalog._base_upgrade_candidates(state, 0)

    assert any(
        op.op_type == OperationType.UPGRADE_GENERATION_SPEED
        for bundle in bundles
        for op in bundle.operations
    )


def test_feature_extractor_increases_generation_value_when_cycle_improves() -> None:
    extractor = FeatureExtractor()
    state_level1 = GameState.initial(seed=19)
    state_level2 = GameState.initial(seed=20)
    state_level1.bases[0].generation_level = 0
    state_level2.bases[0].generation_level = 1

    summary1 = extractor.summarize(state_level1, 0).named
    summary2 = extractor.summarize(state_level2, 0).named

    assert summary2["generation_level"] > summary1["generation_level"]


def test_action_catalog_skips_ant_upgrade_when_next_level_has_no_real_gain() -> None:
    state = GameState.initial(seed=22)
    state.coins[0] = 9999
    state.bases[0].ant_level = 1

    catalog = ActionCatalog(max_actions=64)
    bundles = catalog.build(state, 0)

    assert all(
        op.op_type != OperationType.UPGRADE_GENERATED_ANT
        for bundle in bundles
        for op in bundle.operations
    )


def test_feature_extractor_clamps_ant_value_when_hp_plateaus() -> None:
    extractor = FeatureExtractor()
    state_level1 = GameState.initial(seed=23)
    state_level2 = GameState.initial(seed=24)
    state_level1.bases[0].ant_level = 1
    state_level2.bases[0].ant_level = 2

    summary1 = extractor.summarize(state_level1, 0).named
    summary2 = extractor.summarize(state_level2, 0).named

    assert summary1["ant_level"] == summary2["ant_level"]


def test_mcts_module_is_self_contained() -> None:
    content = Path("AI/ai_mcts.py").read_text()
    assert "ai_greedy" not in content
    assert "greedy_runtime" not in content


def test_repo_sources_no_longer_reference_legacy_runtime() -> None:
    targets = [
        Path("SDK/native_antwar.cpp"),
        Path("SDK/native_adapter.py"),
        Path("SDK/backend/core.py"),
        Path("tools/setup_native.py"),
    ]
    for path in targets:
        content = path.read_text()
        assert "AI_expert" not in content
        assert "expert_oracle" not in content
        assert "expert_reset" not in content


def test_default_backend_stays_python() -> None:
    assert load_backend().name == "python"


def _native_ant_row(
    ant_id: int,
    player: int,
    x: int,
    y: int,
    *,
    hp: int = 20,
    behavior: AntBehavior = AntBehavior.CONSERVATIVE,
    kind: AntKind = AntKind.WORKER,
) -> tuple[int, ...]:
    return (ant_id, player, x, y, hp, 0, 0, int(AntStatus.ALIVE), int(behavior), int(kind))


def _native_position_after_advance(
    *,
    movement_policy: str,
    ant_row: tuple[int, ...],
    active_effects: list[tuple[int, ...]] | None = None,
) -> tuple[int, int, int] | None:
    state = load_backend(prefer_native=True).initial_state(seed=1, movement_policy=movement_policy)
    public_state = state.to_public_round_state()
    state.sync_public_round_state(
        PublicRoundState(
            round_index=1,
            towers=[],
            ants=[ant_row],
            coins=public_state.coins,
            camps_hp=public_state.camps_hp,
            speed_lv=public_state.speed_lv,
            anthp_lv=public_state.anthp_lv,
            weapon_cooldowns=((0, 0, 0, 0), (0, 0, 0, 0)),
            active_effects=active_effects or [],
        )
    )
    state.advance_round()
    ant = next((item for item in state.ants if item.ant_id == ant_row[0]), None)
    if ant is None:
        return None
    return (ant.x, ant.y, ant.hp)


def _native_state_after_advance(
    *,
    seed: int,
    movement_policy: str,
    tower_rows: list[tuple[int, ...]],
    ant_rows: list[tuple[int, ...]],
    active_effects: list[tuple[int, ...]] | None = None,
):
    state = load_backend(prefer_native=True).initial_state(seed=seed, movement_policy=movement_policy)
    public_state = state.to_public_round_state()
    state.sync_public_round_state(
        PublicRoundState(
            round_index=1,
            towers=tower_rows,
            ants=ant_rows,
            coins=public_state.coins,
            camps_hp=public_state.camps_hp,
            speed_lv=public_state.speed_lv,
            anthp_lv=public_state.anthp_lv,
            weapon_cooldowns=((0, 0, 0, 0), (0, 0, 0, 0)),
            active_effects=active_effects or [],
        )
    )
    state.advance_round()
    return state


def test_native_backend_can_boot_and_advance() -> None:
    state = load_backend(prefer_native=True).initial_state(seed=7)
    state.resolve_turn([], [])
    assert state.round_index == 1
    assert len(state.ants) == 2
    assert state.coins == [50, 50]


def test_native_backend_uses_alternating_tower_build_cost_curve() -> None:
    state = load_backend(prefer_native=True).initial_state(seed=11)
    pending: list[Operation] = []
    first_two_slots: list[tuple[int, int]] = []
    for x, y in state.strategic_slots(0):
        operation = Operation(OperationType.BUILD_TOWER, x, y)
        if state.can_apply_operation(0, operation, pending):
            pending.append(operation)
            first_two_slots.append((x, y))
        if len(first_two_slots) == 2:
            break
    assert len(first_two_slots) == 2
    first_two = [
        Operation(OperationType.BUILD_TOWER, *first_two_slots[0]),
        Operation(OperationType.BUILD_TOWER, *first_two_slots[1]),
    ]
    assert state.apply_operation_list(0, first_two) == []
    assert state.coins[0] == 5

    public_state = state.to_public_round_state()
    state.sync_public_round_state(
        PublicRoundState(
            round_index=public_state.round_index,
            towers=public_state.towers,
            ants=public_state.ants,
            coins=(1000, public_state.coins[1]),
            camps_hp=public_state.camps_hp,
            speed_lv=public_state.speed_lv,
            anthp_lv=public_state.anthp_lv,
            weapon_cooldowns=public_state.weapon_cooldowns,
            active_effects=public_state.active_effects,
        )
    )

    third = None
    for x, y in state.strategic_slots(0):
        if (x, y) in first_two_slots:
            continue
        operation = Operation(OperationType.BUILD_TOWER, x, y)
        if state.can_apply_operation(0, operation):
            third = operation
            break
    assert third is not None
    assert state.apply_operation_list(0, [third]) == []
    assert state.coins[0] == 955


def test_native_backend_lightning_storm_matches_new_damage_profile() -> None:
    state = load_backend(prefer_native=True).initial_state(seed=23)
    state.sync_public_round_state(
        PublicRoundState(
            round_index=1,
            towers=[(7, 1, 11, 9, int(TowerType.PRODUCER), 0, 15)],
            ants=[
                (1, 1, 8, 9, 25, 1, 0, int(AntStatus.ALIVE), int(AntBehavior.DEFAULT), int(AntKind.WORKER)),
                (2, 1, 10, 9, 30, 0, 0, int(AntStatus.ALIVE), int(AntBehavior.DEFAULT), int(AntKind.COMBAT)),
            ],
            coins=(50, 50),
            camps_hp=(50, 50),
            speed_lv=(0, 0),
            anthp_lv=(0, 0),
            weapon_cooldowns=((0, 0, 0, 0), (0, 0, 0, 0)),
            active_effects=[(int(SuperWeaponType.LIGHTNING_STORM), 0, 9, 9, 11)],
        )
    )

    state.advance_round()

    ants = {ant.ant_id: ant for ant in state.ants}
    assert ants[1].hp == 5
    assert ants[2].hp == 10
    tower = next(tower for tower in state.towers if tower.tower_id == 7)
    assert tower.hp == 12


def test_native_backend_lightning_storm_triggers_immediately_when_deployed() -> None:
    state = load_backend(prefer_native=True).initial_state(seed=31)
    state.sync_public_round_state(
        PublicRoundState(
            round_index=1,
            towers=[],
            ants=[
                (1, 1, 8, 9, 25, 1, 0, int(AntStatus.ALIVE), int(AntBehavior.DEFAULT), int(AntKind.WORKER)),
                (2, 1, 10, 9, 30, 0, 0, int(AntStatus.ALIVE), int(AntBehavior.DEFAULT), int(AntKind.COMBAT)),
            ],
            coins=(200, 50),
            camps_hp=(50, 50),
            speed_lv=(0, 0),
            anthp_lv=(0, 0),
            weapon_cooldowns=((0, 0, 0, 0), (0, 0, 0, 0)),
            active_effects=[],
        )
    )

    state.apply_operation(0, Operation(OperationType.USE_LIGHTNING_STORM, 9, 9))

    ants = {ant.ant_id: ant for ant in state.ants}
    assert ants[1].hp == 5
    assert ants[2].hp == 10
    assert state.active_effects[0].remaining_turns == 15


def test_native_backend_lightning_storm_respects_evasion_shield() -> None:
    state = load_backend(prefer_native=True).initial_state(seed=33)
    state.sync_public_round_state(
        PublicRoundState(
            round_index=1,
            towers=[],
            ants=[
                (1, 1, 9, 9, 25, 1, 0, int(AntStatus.ALIVE), int(AntBehavior.DEFAULT), int(AntKind.WORKER)),
            ],
            coins=(200, 200),
            camps_hp=(50, 50),
            speed_lv=(0, 0),
            anthp_lv=(0, 0),
            weapon_cooldowns=((0, 0, 0, 0), (0, 0, 0, 0)),
            active_effects=[],
        )
    )

    state.apply_operation(1, Operation(OperationType.USE_EMERGENCY_EVASION, 9, 9))
    state.advance_round()
    state.apply_operation(0, Operation(OperationType.USE_LIGHTNING_STORM, 9, 9))

    assert state.ants[0].hp == 25


def test_native_backend_basic_tower_only_hits_adjacent_targets() -> None:
    state = load_backend(prefer_native=True).initial_state(seed=41)
    state.sync_public_round_state(
        PublicRoundState(
            round_index=1,
            towers=[(7, 0, 12, 9, int(TowerType.BASIC), 0, 10)],
            ants=[
                (1, 1, 10, 9, 20, 0, 0, int(AntStatus.ALIVE), int(AntBehavior.DEFAULT), int(AntKind.WORKER)),
            ],
            coins=(50, 50),
            camps_hp=(50, 50),
            speed_lv=(0, 0),
            anthp_lv=(0, 0),
            weapon_cooldowns=((0, 0, 0, 0), (0, 0, 0, 0)),
            active_effects=[],
        )
    )

    state.advance_round()

    ant = next(ant for ant in state.ants if ant.ant_id == 1)
    assert ant.hp == 20


def test_native_backend_uses_updated_lightning_storm_cost_cooldown_and_duration() -> None:
    state = load_backend(prefer_native=True).initial_state(seed=29)
    public_state = state.to_public_round_state()
    state.sync_public_round_state(
        PublicRoundState(
            round_index=public_state.round_index,
            towers=public_state.towers,
            ants=public_state.ants,
            coins=(90, public_state.coins[1]),
            camps_hp=public_state.camps_hp,
            speed_lv=public_state.speed_lv,
            anthp_lv=public_state.anthp_lv,
            weapon_cooldowns=public_state.weapon_cooldowns,
            active_effects=public_state.active_effects,
        )
    )

    illegal = state.apply_operation_list(0, [Operation(OperationType.USE_LIGHTNING_STORM, 9, 9)])

    assert illegal == []
    assert state.coins[0] == 0
    assert int(state.weapon_cooldowns[0, SuperWeaponType.LIGHTNING_STORM]) == 35
    effect = state.weapon_effect(SuperWeaponType.LIGHTNING_STORM, 0)
    assert effect is not None
    assert effect.remaining_turns == 15


def test_native_backend_can_make_rule_illegal_fatal_when_cold_handling_is_disabled() -> None:
    state = load_backend(prefer_native=True).initial_state(
        seed=30,
        cold_handle_rule_illegal=False,
    )
    invalid = Operation(OperationType.BUILD_TOWER, 12, 9)

    illegal = state.apply_operation_list(0, [invalid])

    assert illegal == [invalid]
    assert state.terminal is True
    assert state.winner == 1
    assert state.towers == []


def test_native_backend_can_skip_rule_illegal_when_cold_handling_is_enabled() -> None:
    state = load_backend(prefer_native=True).initial_state(
        seed=31,
        cold_handle_rule_illegal=True,
    )
    invalid = Operation(OperationType.BUILD_TOWER, 12, 9)

    illegal = state.apply_operation_list(0, [invalid])

    assert illegal == [invalid]
    assert state.terminal is False
    assert state.winner is None
    assert state.towers == []


def test_native_backend_pathfinding_avoids_lightning_storm_under_both_policies() -> None:
    legacy_ant = _native_ant_row(1, 0, 1, 7)
    assert _native_position_after_advance(movement_policy="legacy", ant_row=legacy_ant) == (0, 8, 20)
    assert _native_position_after_advance(
        movement_policy="legacy",
        ant_row=legacy_ant,
        active_effects=[(int(SuperWeaponType.LIGHTNING_STORM), 1, 3, 10, 15)],
    ) == (1, 6, 20)

    enhanced_ant = _native_ant_row(1, 0, 2, 2)
    assert _native_position_after_advance(movement_policy="enhanced", ant_row=enhanced_ant) == (3, 3, 20)
    assert _native_position_after_advance(
        movement_policy="enhanced",
        ant_row=enhanced_ant,
        active_effects=[(int(SuperWeaponType.LIGHTNING_STORM), 1, 1, 6, 15)],
    ) == (3, 2, 20)


def test_native_backend_pathfinding_prefers_deflector_and_evasion_zones() -> None:
    legacy_ant = _native_ant_row(1, 0, 8, 9)
    assert _native_position_after_advance(movement_policy="legacy", ant_row=legacy_ant) == (8, 8, 20)
    for weapon_type, duration in (
        (SuperWeaponType.DEFLECTOR, 10),
        (SuperWeaponType.EMERGENCY_EVASION, 1),
    ):
        assert _native_position_after_advance(
            movement_policy="legacy",
            ant_row=legacy_ant,
            active_effects=[(int(weapon_type), 0, 11, 10, duration)],
        ) == (8, 10, 20)

    enhanced_ant = _native_ant_row(1, 0, 4, 6)
    assert _native_position_after_advance(movement_policy="enhanced", ant_row=enhanced_ant) == (5, 5, 20)
    for weapon_type, center in (
        (SuperWeaponType.DEFLECTOR, (8, 10)),
        (SuperWeaponType.EMERGENCY_EVASION, (6, 10)),
    ):
        duration = 10 if weapon_type == SuperWeaponType.DEFLECTOR else 1
        assert _native_position_after_advance(
            movement_policy="enhanced",
            ant_row=enhanced_ant,
            active_effects=[(int(weapon_type), 0, center[0], center[1], duration)],
        ) == (4, 7, 20)


def test_native_backend_enhanced_conservative_combat_ant_adjacent_tower_damages_it_without_moving() -> None:
    state = _native_state_after_advance(
        seed=0,
        movement_policy="enhanced",
        tower_rows=[(0, 1, 12, 9, int(TowerType.BASIC), 2, 10)],
        ant_rows=[_native_ant_row(24, 0, 11, 9, hp=30, behavior=AntBehavior.CONSERVATIVE, kind=AntKind.COMBAT)],
    )

    tracked_ant = next(item for item in state.ants if item.ant_id == 24)
    tracked_tower = next(item for item in state.towers if item.tower_id == 0)
    assert (tracked_ant.x, tracked_ant.y) == (11, 9)
    assert tracked_tower.hp == 5


def test_native_backend_enhanced_conservative_worker_reroutes_around_single_avoidable_adjacent_tower() -> None:
    state = _native_state_after_advance(
        seed=0,
        movement_policy="enhanced",
        tower_rows=[(0, 1, 12, 9, int(TowerType.BASIC), 2, 10)],
        ant_rows=[_native_ant_row(25, 0, 11, 8, behavior=AntBehavior.CONSERVATIVE, kind=AntKind.WORKER)],
    )

    tracked_ant = next(item for item in state.ants if item.ant_id == 25)
    tracked_tower = next(item for item in state.towers if item.tower_id == 0)
    assert (tracked_ant.x, tracked_ant.y) == (12, 8)
    assert tracked_tower.hp == 10


def test_native_backend_enhanced_conservative_low_hp_combat_ant_self_destructs_on_adjacent_tower_cluster() -> None:
    state = _native_state_after_advance(
        seed=4,
        movement_policy="enhanced",
        tower_rows=[
            (0, 1, 14, 9, int(TowerType.BASIC), 2, 10),
            (1, 1, 14, 10, int(TowerType.BASIC), 2, 12),
        ],
        ant_rows=[_native_ant_row(23, 0, 13, 10, hp=14, behavior=AntBehavior.CONSERVATIVE, kind=AntKind.COMBAT)],
    )

    assert state.die_count[0] == 1
    assert all(item.ant_id != 23 for item in state.ants)
    assert state.tower_by_id(0) is None
    remaining = state.tower_by_id(1)
    assert remaining is not None
    assert remaining.hp == 2


def test_random_runs_on_python_state_without_native_backend() -> None:
    state = GameState.initial(seed=9)
    agent = RandomAgent(seed=9)
    agent.on_match_start(0, 9)
    operations = agent.choose_operations(state, 0)
    assert isinstance(operations, list)
    assert all(hasattr(operation, "to_protocol_tokens") for operation in operations)


def test_mcts_agent_returns_legal_choice() -> None:
    state = GameState.initial(seed=13)
    state.ants.append(Ant(1, 1, 6, 8, hp=10, level=0))
    agent = MCTSAgent(iterations=6, max_depth=2, seed=2)
    bundles = agent.list_bundles(state, 0)
    bundle = agent.choose_bundle(state, 0, bundles=bundles)
    assert bundle in bundles
    assert all(op.op_type in OperationType for op in bundle.operations)


def test_greedy_ai_smoke_uses_sdk_runtime_view_without_re() -> None:
    state = GameState.initial(seed=17)
    state.resolve_turn([], [])
    agent = GreedyAI()
    operations = agent(0, _to_greedy_info(state))
    accepted = []
    for operation in operations:
        sdk_operation = _to_sdk_operation(operation)
        assert state.can_apply_operation(0, sdk_operation, accepted)
        accepted.append(sdk_operation)


def test_greedy_rollout_pheromone_update_tolerates_teleported_ant_trails() -> None:
    info = ForecastState(19)
    info.ants.append(
        ForecastAnt(
            0,
            0,
            18,
            9,
            0,
            0,
            4,
            ForecastAntState.FAIL,
            trail_cells=[(2, 9), (3, 9), (18, 9)],
            last_move=-1,
            path_len_total=2,
        )
    )
    before_origin = info.pheromone[0][3][9]
    before_target = info.pheromone[0][18][9]
    info.update_pheromone(info.ants[0])
    assert info.pheromone[0][3][9] < before_origin
    assert info.pheromone[0][18][9] < before_target


def test_greedy_tower_investment_uses_current_build_curve() -> None:
    greedy_impl = greedy_module._load_impl("ai")
    info = ForecastState(41)
    info.build_tower(0, 0, 6, 9, TowerType.BASIC)
    info.build_tower(1, 0, 5, 9, TowerType.BASIC)
    info.build_tower(2, 0, 4, 9, TowerType.BASIC)

    node = greedy_impl.ForecastNode(greedy_impl.AI(), ForecastSimulator(info))

    expected = -sum(ForecastState.build_tower_cost(index) for index in range(3)) * 0.2 * 0.75
    assert node._score_tower_investment(info.towers) == expected


def test_forecast_max_level_base_upgrade_returns_zero_income() -> None:
    info = ForecastState(23)
    info.bases[0].gen_speed_level = 2
    info.bases[0].ant_level = 2

    gen_upgrade = ForecastOperation(OperationType.UPGRADE_GENERATION_SPEED)
    ant_upgrade = ForecastOperation(OperationType.UPGRADE_GENERATED_ANT)

    assert not info.is_operation_valid(0, gen_upgrade)
    assert not info.is_operation_valid(0, ant_upgrade)
    assert info.get_operation_income(0, gen_upgrade) == 0
    assert info.get_operation_income(0, ant_upgrade) == 0


def test_forecast_tower_build_cost_sequence_matches_alternating_spec() -> None:
    assert [ForecastState.build_tower_cost(i) for i in range(7)] == [15, 30, 45, 90, 135, 270, 405]


def test_forecast_combat_ant_reward_is_fixed() -> None:
    combat = ForecastAnt(0, 0, 2, 9, 30, 0, 0, ForecastAntState.ALIVE, kind=AntKind.COMBAT)
    elite_combat = ForecastAnt(1, 0, 2, 9, 30, 2, 0, ForecastAntState.ALIVE, kind=AntKind.COMBAT)

    assert combat.reward() == COMBAT_ANT_KILL_REWARD
    assert elite_combat.reward() == COMBAT_ANT_KILL_REWARD


def test_forecast_producer_tower_does_not_crash_attack_loop() -> None:
    info = ForecastState(29)
    info.build_tower(0, 0, 6, 9, TowerType.PRODUCER)
    info.ants.append(
        ForecastAnt(
            0,
            1,
            7,
            9,
            10,
            0,
            0,
            ForecastAntState.ALIVE,
        )
    )

    simulator = ForecastSimulator(info)
    assert simulator.fast_next_round(0)
    enemy_ant = simulator.info.ant_of_id(0)
    assert enemy_ant is not None
    assert enemy_ant.hp == 10


def test_forecast_lightning_storm_damages_enemy_combat_ants_without_instant_kill() -> None:
    info = ForecastState(31)
    info.ants.extend(
        [
            ForecastAnt(0, 1, 9, 9, 25, 1, 0, ForecastAntState.ALIVE),
            ForecastAnt(1, 1, 10, 9, 30, 0, 0, ForecastAntState.ALIVE, kind=AntKind.COMBAT),
        ]
    )
    info.use_super_weapon(SuperWeaponType.LIGHTNING_STORM, 0, 9, 9)

    simulator = ForecastSimulator(info)

    assert simulator.fast_next_round(0)
    worker = simulator.info.ant_of_id(0)
    combat = simulator.info.ant_of_id(1)
    assert worker is not None
    assert combat is not None
    assert worker.hp == 5
    assert combat.hp == 10


def test_forecast_pathfinding_responds_to_storm_and_support_fields() -> None:
    storm_base = ForecastState(1)
    storm_ant = ForecastAnt(0, 0, 2, 10, 20, 0, 0, ForecastAntState.ALIVE)
    storm_base.ants.append(storm_ant)
    assert storm_base.next_move(storm_ant) == 3

    storm_threat = ForecastState(1)
    storm_threat_ant = ForecastAnt(0, 0, 2, 10, 20, 0, 0, ForecastAntState.ALIVE)
    storm_threat.ants.append(storm_threat_ant)
    storm_threat.use_super_weapon(SuperWeaponType.LIGHTNING_STORM, 1, 0, 8)
    assert storm_threat.next_move(storm_threat_ant) == 4

    support_base = ForecastState(1)
    support_ant = ForecastAnt(0, 0, 2, 4, 20, 0, 0, ForecastAntState.ALIVE)
    support_base.ants.append(support_ant)
    assert support_base.next_move(support_ant) == 0

    for weapon_type in (SuperWeaponType.DEFLECTOR, SuperWeaponType.EMERGENCY_EVASION):
        support = ForecastState(1)
        support_ant = ForecastAnt(0, 0, 2, 4, 20, 0, 0, ForecastAntState.ALIVE)
        support.ants.append(support_ant)
        support.use_super_weapon(weapon_type, 0, 2, 1)
        assert support.next_move(support_ant) == 4
