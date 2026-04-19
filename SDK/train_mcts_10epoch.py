from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_mcts.py"
    argv = [
        sys.executable,
        str(train_script),
        "--batches",
        "10",
        "--progress-log-decisions",
        "1",
        "--progress-log-seconds",
        "3.0",
        "--log-dir",
        "logs/train_mcts_10epoch",
        "--run-name",
        "10epoch",
        "--checkpoint",
        "checkpoints/ai_mcts_10epoch_latest.npz",
        *sys.argv[1:],
    ]
    env = os.environ.copy()
    env["AGENT_TRADITION_TRAINING_ENTRYPOINT"] = "SDK/train_mcts_10epoch.py"
    os.execve(sys.executable, argv, env)


if __name__ == "__main__":
    main()
