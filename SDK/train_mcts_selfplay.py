from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from SDK.mcts_agent import SearchConfig, TrainConfig, train_mcts_selfplay


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the handcrafted-feature MCTS value model with self-play.")
    parser.add_argument("--save", type=str, default="AI/selfplay/mcts_value.npz")
    parser.add_argument("--warm_start", type=str, default=None)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--c_puct", type=float, default=1.35)
    parser.add_argument("--heuristic_weight", type=float, default=0.7)
    parser.add_argument("--train_every", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument(
        "--opponents",
        type=str,
        default="self,handcraft",
        help="Comma-separated pool from {self,handcraft,greedy,random_safe} or module:function specs",
    )
    args = parser.parse_args(argv)

    search_cfg = SearchConfig(
        simulations=args.simulations,
        max_depth=args.depth,
        c_puct=args.c_puct,
        heuristic_weight=args.heuristic_weight,
    )
    train_cfg = TrainConfig(
        games=args.games,
        max_rounds=args.rounds,
        seed=args.seed,
        train_every=args.train_every,
        buffer_size=args.buffer_size,
        fit_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        opponents=tuple(item.strip() for item in args.opponents.split(",") if item.strip()),
    )

    summary = train_mcts_selfplay(
        args.save,
        search_config=search_cfg,
        train_config=train_cfg,
        warm_start_path=args.warm_start,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
