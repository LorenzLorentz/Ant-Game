from __future__ import annotations

from pathlib import Path

from AI.ai_greedy import AI as GreedyAI, _to_greedy_info, _to_sdk_operation
from AI.ai_mcts import MCTSAgent
from AI.ai_random import RandomAgent
from SDK.utils.actions import ActionCatalog
from SDK.backend import load_backend
from SDK.utils.features import FeatureExtractor
from SDK.utils.constants import COMBAT_ANT_KILL_REWARD, AntKind, OperationType, TowerType
from SDK.backend.engine import GameState, PublicRoundState
from SDK.backend.forecast import Ant as ForecastAnt, AntState as ForecastAntState, ForecastSimulator, ForecastState, Operation as ForecastOperation
from SDK.backend.model import Ant, Operation


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


def test_action_catalog_skips_generation_upgrade_when_next_level_has_no_real_gain() -> None:
    state = GameState.initial(seed=18)
    state.coins[0] = 9999
    state.bases[0].generation_level = 0

    catalog = ActionCatalog(max_actions=64)
    bundles = catalog.build(state, 0)

    assert all(
        op.op_type != OperationType.UPGRADE_GENERATION_SPEED
        for bundle in bundles
        for op in bundle.operations
    )


def test_feature_extractor_clamps_generation_value_when_cycle_plateaus() -> None:
    extractor = FeatureExtractor()
    state_level1 = GameState.initial(seed=19)
    state_level2 = GameState.initial(seed=20)
    state_level1.bases[0].generation_level = 0
    state_level2.bases[0].generation_level = 1

    summary1 = extractor.summarize(state_level1, 0).named
    summary2 = extractor.summarize(state_level2, 0).named

    assert summary1["generation_level"] == summary2["generation_level"]


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
