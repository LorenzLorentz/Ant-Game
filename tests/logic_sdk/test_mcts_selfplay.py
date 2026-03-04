import os

import numpy as np

from SDK.mcts_agent import LinearValueModel, SearchConfig, TrainConfig, train_mcts_selfplay
from SDK.mcts_features import extract_features
from logic.gamestate import GameState


def test_mcts_feature_shape_is_stable():
    features = extract_features(GameState(), 0)
    assert features.ndim == 1
    assert features.shape[0] >= 16
    assert np.all(np.isfinite(features))


def test_train_mcts_zero_game_saves_model(tmp_path):
    save_path = tmp_path / "mcts_value.npz"
    summary = train_mcts_selfplay(
        str(save_path),
        search_config=SearchConfig(simulations=2, max_depth=1),
        train_config=TrainConfig(games=0),
    )

    assert os.path.exists(save_path)
    model = LinearValueModel.load(str(save_path))
    assert model.feature_dim == extract_features(GameState(), 0).shape[0]
    assert summary["games"] == 0.0


def test_train_mcts_with_handcraft_opponent_runs(tmp_path):
    save_path = tmp_path / "mcts_value.npz"
    summary = train_mcts_selfplay(
        str(save_path),
        search_config=SearchConfig(simulations=2, max_depth=2, heuristic_weight=0.8),
        train_config=TrainConfig(
            games=2,
            max_rounds=6,
            seed=7,
            train_every=1,
            fit_epochs=1,
            batch_size=32,
            opponents=("handcraft",),
        ),
    )

    assert os.path.exists(save_path)
    assert summary["games"] == 2.0
    assert summary["model_updates"] > 0.0
