#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import random
from typing import Callable, Optional

from logic.gamestate import GameState
from logic.runner import run_match

AI = Callable[[int, int, GameState], list[list[int]]]


def load_ai(spec: str) -> AI:
    """
    Load AI by spec.
    - If spec contains ':', interpret as module:function
    - Else interpret as AI.ai_{spec}:policy
    """
    if ":" in spec:
        mod_name, func_name = spec.split(":", 1)
    else:
        mod_name, func_name = f"AI.ai_{spec}", "policy"
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


def load_metrics(spec: Optional[str]):
    if not spec:
        return None
    mod_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


def main(argv=None):
    p = argparse.ArgumentParser(description="Batch evaluate two AIs with seat-swapping and metrics.")
    p.add_argument("--ai0", required=True, help="AI 0 (name or module:function)")
    p.add_argument("--ai1", required=True, help="AI 1 (name or module:function)")
    p.add_argument("--games", type=int, default=20, help="Number of games")
    p.add_argument("--rounds", type=int, default=60, help="Max rounds per game")
    p.add_argument("--seed", type=int, default=None, help="Base random seed")
    p.add_argument("--swap_seats", action="store_true", help="Alternate seats across games")
    p.add_argument("--replay_dir", type=str, default="replays", help="Directory for replays")
    p.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Optional module:function returning dict of custom per-game metrics",
    )
    args = p.parse_args(argv)

    ai0f = load_ai(args.ai0)
    ai1f = load_ai(args.ai1)
    metric_fn = load_metrics(args.metrics)

    rng = random.Random(args.seed)
    os.makedirs(args.replay_dir, exist_ok=True)

    stats = {
        "total_games": 0,
        "p0_wins": 0,
        "p1_wins": 0,
        "draws": 0,
        "ai0_as_p0_wins": 0,
        "ai0_as_p1_wins": 0,
        "ai1_as_p0_wins": 0,
        "ai1_as_p1_wins": 0,
        "avg_rounds": 0.0,
    }
    extra = []

    for g in range(args.games):
        swap = args.swap_seats and (g % 2 == 1)
        if not swap:
            p0_name, p1_name = args.ai0, args.ai1
            p0, p1 = ai0f, ai1f
        else:
            p0_name, p1_name = args.ai1, args.ai0
            p0, p1 = ai1f, ai0f

        seed = rng.randrange(1 << 30)
        winner, state = run_match(
            p0,
            p1,
            seed=seed,
            max_rounds=args.rounds,
            replay_file=None,
            p0_name=p0_name,
            p1_name=p1_name,
            replay_dir=args.replay_dir,
        )
        stats["total_games"] += 1
        stats["avg_rounds"] += state.round
        if winner == 0:
            stats["p0_wins"] += 1
            if not swap:
                stats["ai0_as_p0_wins"] += 1
            else:
                stats["ai1_as_p0_wins"] += 1
        elif winner == 1:
            stats["p1_wins"] += 1
            if not swap:
                stats["ai1_as_p1_wins"] += 1
            else:
                stats["ai0_as_p1_wins"] += 1
        else:
            stats["draws"] += 1

        if metric_fn is not None:
            try:
                extra.append(metric_fn(state))
            except Exception as e:
                extra.append({"metrics_error": str(e)})

    if stats["total_games"] > 0:
        stats["avg_rounds"] = stats["avg_rounds"] / stats["total_games"]
    # Derived metrics
    total_non_draw = stats["p0_wins"] + stats["p1_wins"]
    stats["win_rate_p0"] = (stats["p0_wins"] / total_non_draw) if total_non_draw else 0.0
    stats["win_rate_p1"] = (stats["p1_wins"] / total_non_draw) if total_non_draw else 0.0

    result = {"summary": stats, "extra": extra}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

