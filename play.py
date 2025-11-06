#!/usr/bin/env python3
import argparse
import importlib
import json
from copy import deepcopy
from typing import Callable

from logic.gamestate import GameState, init_generals, update_round
from logic.ai2logic import execute_single_command
from logic.runner import run_match
from logic.game_rules import is_game_over


AI = Callable[[int, int, GameState], list[list[int]]]


def load_ai_by_name(name: str) -> AI:
    module_name = f"AI.ai_{name}"
    mod = importlib.import_module(module_name)
    return getattr(mod, "policy")


def show_hints(state: GameState, player: int) -> None:
    # Display basic legal-operation hints
    cells = []
    gens = []
    for i in range(len(state.board)):
        for j in range(len(state.board[0])):
            cell = state.board[i][j]
            if cell.player == player:
                if cell.army > 1:
                    cells.append((i, j, cell.army))
                if cell.generals is not None:
                    gens.append((cell.generals.id, i, j))
    print("Your controlled cells with >1 army:")
    print(", ".join([f"({x},{y}):{a}" for x, y, a in cells]) or "None")
    print("Your generals [id@pos]:")
    print(", ".join([f"{gid}@({x},{y})" for gid, x, y in gens]) or "None")
    print("Commands:")
    print("  1 x y dir num       # Army move (dir=1..4)")
    print("  2 id x y            # General move")
    print("  3 id kind           # Level up (1 prod,2 def,3 mob)")
    print("  4 id skill [x y]    # Skill (1..5; 1/2 need x y)")
    print("  5 type              # Tech update (1..4)")
    print("  6 wt x y [sx sy]    # Superweapon (1..4)")
    print("  7 x y               # Call general")
    print("  8                   # End turn")


def manual_turn(state: GameState, player: int) -> bool:
    print(f"=== Player {player} turn ===")
    show_hints(state, player)
    ops: list[list[int]] = []
    while True:
        line = input("Enter command (or 'hint'/'end'): ").strip()
        if line.lower() in ("end", "8"):
            break
        if line.lower() == "hint":
            show_hints(state, player)
            continue
        try:
            cmd = [int(x) for x in line.split()]
        except Exception:
            print("Invalid format. See 'hint' for usage.")
            continue
        if not cmd:
            continue
        t = cmd[0]
        params = cmd[1:]
        # Validate by simulating on a copy
        sim = deepcopy(state)
        ok = execute_single_command(player, sim, t, params)
        if not ok:
            print("Illegal operation. See 'hint' for valid ranges and your resources.")
            continue
        # Apply on real state
        execute_single_command(player, state, t, params)
        ops.append(cmd)
        # check immediate win
        if is_game_over(state) != -1:
            break
    # End turn
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="Play or simulate a match.")
    p.add_argument("mode", choices=["ai-vs-ai", "manual-vs-ai", "manual-vs-manual"], help="Play mode")
    p.add_argument("--ai0", default="greedy", help="AI name for player 0 (file AI/ai_{name}.py)")
    p.add_argument("--ai1", default="random_safe", help="AI name for player 1 (file AI/ai_{name}.py)")
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    if args.mode == "ai-vs-ai":
        ai0 = load_ai_by_name(args.ai0)
        ai1 = load_ai_by_name(args.ai1)
        winner, state = run_match(
            ai0,
            ai1,
            seed=args.seed,
            max_rounds=args.rounds,
            replay_file=None,
            p0_name=args.ai0,
            p1_name=args.ai1,
        )
        print(json.dumps({"winner": winner, "replay": state.replay_file}))
        return 0

    # Manual modes use a fresh state
    state = GameState()
    init_generals(state)
    # Initialize replay
    from logic.runner import _write_init_replay  # reuse helper
    from datetime import datetime
    import os

    os.makedirs("replays", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p1n = args.ai1 if args.mode == "manual-vs-ai" else "manual"
    state.replay_file = os.path.join("replays", f"{ts}_p0-manual_p1-{p1n}.jsonl")
    _write_init_replay(state)
    player = 0
    ai1 = load_ai_by_name(args.ai1) if args.mode == "manual-vs-ai" else None
    round_idx = 1
    while True:
        if args.mode == "manual-vs-manual" or (args.mode == "manual-vs-ai" and player == 0):
            manual_turn(state, player)
        else:
            ops = ai1(round_idx, 1, state)  # type: ignore
            for op in ops:
                if op and op[0] != 8:
                    execute_single_command(1, state, op[0], op[1:])
        if is_game_over(state) != -1:
            break
        if player == 1:
            update_round(state)
            round_idx += 1
        player = 1 - player
    print("Game over.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
