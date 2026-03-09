from __future__ import annotations

import numpy as np

from SDK.training import AntWarParallelEnv
from SDK.training.base import BaseSelfPlayTrainer
from SDK.training.selfplay import LinearSelfPlayTrainer, TrainerConfig


class DummyTrainer(BaseSelfPlayTrainer):
    def __init__(self) -> None:
        super().__init__(env_factory=lambda seed=0: AntWarParallelEnv(seed=seed), episodes_per_batch=1, seed=0)

    def select_action(self, observation, explore: bool = True) -> int:
        mask = observation["action_mask"]
        return int(np.argmax(mask))

    def update_from_batch(self, batch):
        return {"count": float(len(batch.actions))}


def test_base_trainer_can_be_subclassed() -> None:
    trainer = DummyTrainer()
    history = trainer.train(1)
    assert history[0]["count"] > 0


def test_environment_observation_exposes_runtime_rule_channels() -> None:
    env = AntWarParallelEnv(seed=2)
    observations, _ = env.reset(seed=2)
    board = observations["player_0"]["board"]
    assert board.shape[0] >= 28
    assert np.any(board[14] > 0)
    env.close()


def test_linear_selfplay_trainer_runs_one_batch() -> None:
    trainer = LinearSelfPlayTrainer(
        lambda seed=0: AntWarParallelEnv(seed=seed),
        TrainerConfig(episodes_per_batch=1, seed=4, learning_rate=5e-3, value_learning_rate=1e-3),
    )
    before = trainer.policy.policy_weights.copy()
    history = trainer.train(1)
    after = trainer.policy.policy_weights
    assert history
    assert not np.allclose(before, after)
    metrics = trainer.evaluate_policy(1)
    assert "eval_return" in metrics
