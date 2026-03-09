from __future__ import annotations

import subprocess
import sys

from AI.ai_example import AI as ExampleAI
from SDK.backend import GameState


def test_ai_example_can_choose_operations() -> None:
    agent = ExampleAI(seed=5, max_actions=16)
    state = GameState.initial(seed=5)
    operations = agent.choose_operations(state, 0)
    assert isinstance(operations, list)


def test_train_example_script_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/train_example.py", "--seed", "2", "--max-actions", "16"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "AI/ai_example.py" in completed.stdout
    assert "SDK.backend" in completed.stdout


def test_train_mcts_script_runs_short_scaffold() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/train_mcts.py",
            "--episodes",
            "1",
            "--iterations",
            "2",
            "--max-depth",
            "1",
            "--max-rounds",
            "2",
            "--seed",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"episodes": 1' in completed.stdout
    assert "update_from_episodes" in completed.stdout
