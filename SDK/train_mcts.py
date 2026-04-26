from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SDK.training import (  # noqa: E402
    AlphaZeroSelfPlayTrainer,
    AlphaZeroTrainerConfig,
    AntWarSequentialEnv,
    TrainingLogger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Ant Game MCTS policy/value model with self-play.")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--search-iterations", type=int, default=96)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=192)
    parser.add_argument("--max-actions", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--l2-weight", type=float, default=1e-5)
    parser.add_argument("--c-puct", type=float, default=1.2)
    parser.add_argument("--root-action-limit", type=int, default=16)
    parser.add_argument("--child-action-limit", type=int, default=8)
    parser.add_argument("--prior-mix", type=float, default=0.75)
    parser.add_argument("--value-mix", type=float, default=0.7)
    parser.add_argument("--value-scale", type=float, default=350.0)
    parser.add_argument("--root-temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-round", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--hidden-dim2", type=int, default=96)
    parser.add_argument("--evaluation-episodes", type=int, default=4)
    parser.add_argument("--promotion-episodes", type=int, default=6)
    parser.add_argument("--promotion-win-rate", type=float, default=0.55)
    parser.add_argument("--champion-path", type=str, default="checkpoints/ai_mcts_champion.npz")
    parser.add_argument("--opponent-pool-size", type=int, default=6)
    parser.add_argument("--selfplay-mirror-ratio", type=float, default=0.4)
    parser.add_argument("--selfplay-heuristic-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/ai_mcts_latest.npz")
    parser.add_argument("--run-dir", type=str, default="checkpoints/runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--export-agent-model", type=str, default="AI/ai_mcts_model.npz")
    parser.add_argument("--prefer-native-backend", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AlphaZeroTrainerConfig:
    return AlphaZeroTrainerConfig(
        batches=args.batches,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        value_weight=args.value_weight,
        l2_weight=args.l2_weight,
        search_iterations=args.search_iterations,
        max_depth=args.max_depth,
        c_puct=args.c_puct,
        root_action_limit=args.root_action_limit,
        child_action_limit=args.child_action_limit,
        prior_mix=args.prior_mix,
        value_mix=args.value_mix,
        value_scale=args.value_scale,
        root_temperature=args.root_temperature,
        temperature_drop_round=args.temperature_drop_round,
        seed=args.seed,
        max_rounds=args.max_rounds,
        max_actions=args.max_actions,
        hidden_dim=args.hidden_dim,
        hidden_dim2=args.hidden_dim2,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
        evaluation_episodes=args.evaluation_episodes,
        promotion_episodes=args.promotion_episodes,
        promotion_win_rate=args.promotion_win_rate,
        champion_path=args.champion_path,
        opponent_pool_size=args.opponent_pool_size,
        selfplay_mirror_ratio=args.selfplay_mirror_ratio,
        selfplay_heuristic_ratio=args.selfplay_heuristic_ratio,
    )


def export_agent_model(checkpoint_path: str | Path, export_path: str | Path | None) -> str | None:
    if not export_path:
        return None
    source = Path(checkpoint_path)
    if not source.exists():
        return None
    target = Path(export_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    logger = TrainingLogger(base_dir=args.run_dir, run_name=args.run_name)
    logger.log_config(
        {
            "trainer": "AlphaZeroSelfPlayTrainer",
            "repo_root": str(REPO_ROOT),
            "prefer_native_backend": bool(args.prefer_native_backend),
            "export_agent_model": args.export_agent_model,
            "config": asdict(config),
        }
    )

    def env_factory(seed: int):
        return AntWarSequentialEnv(
            seed=seed,
            max_actions=config.max_actions,
            prefer_native_backend=args.prefer_native_backend,
        )

    trainer = AlphaZeroSelfPlayTrainer(
        env_factory=env_factory,
        config=config,
        logger=logger,
    )

    try:
        history, samples = trainer.train()
        export_source = config.champion_path if Path(config.champion_path).exists() else config.checkpoint_path
        exported_model = export_agent_model(export_source, args.export_agent_model)
        summary = {
            "checkpoint_path": str(Path(config.checkpoint_path)),
            "champion_path": str(Path(config.champion_path)),
            "exported_agent_model": exported_model,
            "history": history,
            "sample_episodes": [asdict(sample) for sample in samples[: min(5, len(samples))]],
        }
        logger.log_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:
        logger.log_error(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
