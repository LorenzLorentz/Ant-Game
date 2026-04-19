from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SDK.training import AlphaZeroSelfPlayTrainer, AlphaZeroTrainerConfig, AntWarSequentialEnv, TrainingLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the model-prior MCTS agent with PettingZoo self-play.")
    parser.add_argument("--batches", type=int, default=1, help="Number of train/update cycles to run.")
    parser.add_argument("--episodes", type=int, default=2, help="Self-play episodes collected per update.")
    parser.add_argument("--iterations", type=int, default=24, help="MCTS iterations per decision.")
    parser.add_argument("--max-depth", type=int, default=3, help="Search depth in whole-turn plies.")
    parser.add_argument("--max-rounds", type=int, default=128, help="Hard cap for each self-play match.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for search and environment resets.")
    parser.add_argument("--max-actions", type=int, default=96, help="Candidate action budget exposed by the env.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Shared SGD step size for the network.")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Weight on the value regression loss.")
    parser.add_argument("--l2-weight", type=float, default=1e-5, help="L2 regularization on policy-value weights.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Width of the first hidden layer.")
    parser.add_argument("--hidden-dim2", type=int, default=64, help="Width of the second hidden layer.")
    parser.add_argument("--c-puct", type=float, default=1.25, help="Exploration constant used by PUCT.")
    parser.add_argument("--prior-mix", type=float, default=0.7, help="Blend ratio of learned priors against heuristics.")
    parser.add_argument("--value-mix", type=float, default=0.7, help="Blend ratio of learned value against heuristics.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ai_mcts_latest.npz", help="Path to save the latest checkpoint.")
    parser.add_argument("--log-dir", type=str, default="logs/train_mcts", help="Base directory for training logs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name used under the log directory.")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint path used to resume training.")
    parser.add_argument("--evaluation-episodes", type=int, default=2, help="How many heuristic matches to run after each update.")
    parser.add_argument(
        "--progress-log-decisions",
        type=int,
        default=8,
        help="Emit an in-episode progress heartbeat every N decisions. Use 1 for very detailed logs.",
    )
    parser.add_argument(
        "--progress-log-seconds",
        type=float,
        default=5.0,
        help="Emit an in-episode progress heartbeat if this many seconds pass without a heartbeat.",
    )
    parser.add_argument(
        "--prefer-native-backend",
        action="store_true",
        help="Prefer the optional native backend for environment resets if it is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlphaZeroTrainerConfig(
        batches=args.batches,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        value_weight=args.value_weight,
        l2_weight=args.l2_weight,
        search_iterations=args.iterations,
        max_depth=args.max_depth,
        c_puct=args.c_puct,
        prior_mix=args.prior_mix,
        value_mix=args.value_mix,
        seed=args.seed,
        max_rounds=args.max_rounds,
        max_actions=args.max_actions,
        hidden_dim=args.hidden_dim,
        hidden_dim2=args.hidden_dim2,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume_from,
        evaluation_episodes=args.evaluation_episodes,
        progress_log_decisions=args.progress_log_decisions,
        progress_log_seconds=args.progress_log_seconds,
    )
    logger = TrainingLogger(base_dir=args.log_dir, run_name=args.run_name)
    logger.log_config(
        {
            "argv": vars(args),
            "trainer_config": asdict(config),
        }
    )
    try:
        trainer = AlphaZeroSelfPlayTrainer(
            env_factory=lambda seed=0: AntWarSequentialEnv(
                seed=seed,
                max_actions=args.max_actions,
                prefer_native_backend=args.prefer_native_backend,
            ),
            config=config,
            logger=logger,
        )
        history, samples = trainer.train()
        latest = history[-1] if history else {}
        training_entrypoint = os.getenv("AGENT_TRADITION_TRAINING_ENTRYPOINT", "SDK/train_mcts.py")
        result = {
            "episodes": args.episodes,
            "batches": args.batches,
            "iterations": args.iterations,
            "max_depth": args.max_depth,
            "max_rounds": args.max_rounds,
            "checkpoint": str(Path(args.checkpoint)),
            "log_dir": str(logger.run_dir),
            "resume_from": args.resume_from,
            "training_entrypoint": training_entrypoint,
            "trainer_logic_hook": "AlphaZeroSelfPlayTrainer.update_from_batch()",
            "agent_logic_file": "AI/ai_mcts.py",
            "policy_backend": "SDK.alphazero.PolicyValueNet",
            "search_backend": "SDK.alphazero.PriorGuidedMCTS",
            "scaffold_compat": "update_from_episodes was replaced by AlphaZeroSelfPlayTrainer.update_from_batch().",
            "latest_metrics": latest,
            "history": history,
            "samples": [asdict(summary) for summary in samples[: min(len(samples), 3)]],
        }
        logger.log_summary(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        logger.log_error(f"training failed: {exc}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
