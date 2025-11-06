import importlib
import random
import os
from datetime import datetime
from typing import Callable, Optional

from logic.gamestate import GameState, init_generals, update_round
from logic.ai2logic import execute_single_command
from logic.game_rules import is_game_over, tiebreak_now


AI = Callable[[int, int, GameState], list[list[int]]]


def _write_init_replay(state: GameState) -> None:
    data = state.trans_state_to_init_json(-1)
    data["Round"] = 0
    with open(state.replay_file, "w") as f:
        f.write(str(data).replace("'", '"') + "\n")


def run_match(
    ai0: AI,
    ai1: AI,
    seed: Optional[int] = None,
    max_rounds: int = 200,
    replay_file: str | None = None,
    p0_name: str = "p0",
    p1_name: str = "p1",
    replay_dir: str = "replays",
) -> tuple[int, GameState]:
    """
    Run a local match between two AI callables.
    Returns (winner, final_state); winner is -1 if no winner before tiebreak.
    """
    if seed is not None:
        random.seed(seed)
    state = GameState()
    # Prepare replay path
    if replay_file is None:
        os.makedirs(replay_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_file = os.path.join(
            replay_dir,
            f"{ts}_p0-{p0_name}_p1-{p1_name}_seed-{seed if seed is not None else 'na'}_rounds-{max_rounds}.jsonl",
        )
    state.replay_file = replay_file
    # Initialize map and generals
    init_generals(state)
    _write_init_replay(state)

    round_idx = 1
    winner = -1
    while round_idx <= max_rounds and winner == -1:
        # Player 0 turn
        ops0 = ai0(round_idx, 0, state)
        _apply_ops(0, state, ops0)
        winner = is_game_over(state)
        if winner != -1:
            break
        # Player 1 turn
        ops1 = ai1(round_idx, 1, state)
        _apply_ops(1, state, ops1)
        winner = is_game_over(state)
        # End of full round
        update_round(state)
        round_idx += 1

    if winner == -1 and round_idx > max_rounds:
        winner = tiebreak_now(state)
    return winner, state


def _apply_ops(player: int, state: GameState, ops: list[list[int]]) -> None:
    """Apply a list of operations. Invalid ops are ignored (treated as no-ops).
    Turn ends when an op with type==8 is encountered or ops are exhausted.
    """
    for op in ops:
        if not op:
            continue
        if op[0] == 8:
            break
        execute_single_command(player, state, op[0], op[1:])
    # Ensure not exceeding remaining move steps is handled by underlying logic


def load_callable(spec: str) -> AI:
    """Load a callable from 'module:function' string."""
    mod_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func  # type: ignore
