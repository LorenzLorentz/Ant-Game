from __future__ import annotations

import os

from logic.gamestate import GameState

from SDK.mcts_agent import LinearValueModel, MCTSAgent, SearchConfig


MODEL_PATH = os.environ.get("MCTS_MODEL", os.path.join("AI", "selfplay", "mcts_value.npz"))
SIMULATIONS = int(os.environ.get("MCTS_SIMULATIONS", "16"))
SEARCH_DEPTH = int(os.environ.get("MCTS_DEPTH", "3"))
HEURISTIC_WEIGHT = float(os.environ.get("MCTS_HEURISTIC_WEIGHT", "0.7"))
CPUCT = float(os.environ.get("MCTS_CPUCT", "1.35"))

_AGENT: MCTSAgent | None = None


def _build_agent() -> MCTSAgent:
    model = LinearValueModel.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    config = SearchConfig(
        simulations=SIMULATIONS,
        max_depth=SEARCH_DEPTH,
        c_puct=CPUCT,
        heuristic_weight=HEURISTIC_WEIGHT,
    )
    return MCTSAgent(model=model, search_config=config)


def policy(round_idx: int, my_seat: int, state: GameState) -> list[list[int]]:
    del round_idx
    global _AGENT
    if _AGENT is None:
        _AGENT = _build_agent()
    return _AGENT.policy(state.round, my_seat, state)


def ai_func(state: GameState) -> list[list[int]]:
    return policy(getattr(state, "round", 1), 0, state)
