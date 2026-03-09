from __future__ import annotations

from pathlib import Path

from AI.AI_expert.ai import AI as ExpertAI
from AI.AI_expert.runtime import _to_expert_info, _to_sdk_operation
from AI.ai_mcts import MCTSAgent
from AI.ai_random import RandomAgent
from SDK.actions import ActionCatalog
from SDK.backend import load_backend
from SDK.constants import OperationType
from SDK.engine import GameState
from SDK.model import Ant


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


def test_action_catalog_tolerates_stale_ant_paths() -> None:
    state = GameState.initial(seed=5)
    state.ants.append(Ant(99, 1, 17, 9, hp=10, level=0, age=32, path=[4, 4, 4]))
    catalog = ActionCatalog(max_actions=16)
    bundles = catalog.build(state, 0)
    assert bundles


def test_mcts_module_is_self_contained() -> None:
    content = Path("AI/ai_mcts.py").read_text()
    assert "ai_greedy" not in content
    assert "greedy_runtime" not in content


def test_repo_sources_no_longer_reference_ai_expert_runtime() -> None:
    targets = [
        Path("SDK/native_antwar.cpp"),
        Path("SDK/native_adapter.py"),
        Path("SDK/backend.py"),
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
    assert state.coins == [51, 51]


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


def test_expert_ai_smoke_uses_sdk_runtime_view_without_re() -> None:
    state = GameState.initial(seed=17)
    state.resolve_turn([], [])
    agent = ExpertAI()
    operations = agent(0, _to_expert_info(state))
    accepted = []
    for operation in operations:
        sdk_operation = _to_sdk_operation(operation)
        assert state.can_apply_operation(0, sdk_operation, accepted)
        accepted.append(sdk_operation)
