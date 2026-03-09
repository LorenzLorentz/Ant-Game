"""Compatibility wrapper for the training environment."""

from SDK.training.env import AntWarParallelEnv, env

__all__ = [
    "AntWarParallelEnv",
    "env",
]
