from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from SDK.utils.features import FeatureExtractor
from SDK.training.env import AntWarSequentialEnv


@dataclass(slots=True)
class TrajectoryStep:
    observation: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    done: bool


@dataclass(slots=True)
class EpisodeBatch:
    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    returns: np.ndarray


class BaseSelfPlayTrainer(ABC):
    def __init__(
        self,
        env_factory,
        gamma: float = 0.99,
        episodes_per_batch: int = 4,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.gamma = gamma
        self.episodes_per_batch = episodes_per_batch
        self.seed = seed
        self.feature_extractor = FeatureExtractor()

    @abstractmethod
    def select_action(self, observation: dict[str, np.ndarray], explore: bool = True) -> int:
        raise NotImplementedError

    @abstractmethod
    def update_from_batch(self, batch: EpisodeBatch) -> dict[str, float]:
        raise NotImplementedError

    def _discounted_returns(self, rewards: list[float]) -> np.ndarray:
        running = 0.0
        returns = []
        for reward in reversed(rewards):
            running = reward + self.gamma * running
            returns.append(running)
        returns.reverse()
        return np.asarray(returns, dtype=np.float32)

    def collect_episode(self, env: AntWarSequentialEnv, explore: bool = True, seed: int | None = None) -> EpisodeBatch:
        env.reset(seed=self.seed if seed is None else seed)
        traces = {agent: [] for agent in env.possible_agents}
        pending: dict[str, TrajectoryStep | None] = {agent: None for agent in env.possible_agents}
        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            previous = pending[agent]
            if previous is not None:
                previous.reward = float(reward)
                previous.done = bool(termination or truncation)
                pending[agent] = None
            if termination or truncation:
                env.step(None)
                continue

            action = self.select_action(observation, explore=explore)
            current = TrajectoryStep(
                observation=self.feature_extractor.flatten_observation(observation),
                mask=observation["action_mask"].astype(np.float32),
                action=action,
                reward=0.0,
                done=False,
            )
            traces[agent].append(current)
            pending[agent] = current
            env.step(action)

        observation_rows = []
        mask_rows = []
        action_rows = []
        return_rows = []
        for agent in env.possible_agents:
            rewards = [step.reward for step in traces[agent]]
            returns = self._discounted_returns(rewards)
            for step, ret in zip(traces[agent], returns):
                observation_rows.append(step.observation)
                mask_rows.append(step.mask)
                action_rows.append(step.action)
                return_rows.append(ret)
        return EpisodeBatch(
            observations=np.asarray(observation_rows, dtype=np.float32),
            masks=np.asarray(mask_rows, dtype=np.float32),
            actions=np.asarray(action_rows, dtype=np.int64),
            returns=np.asarray(return_rows, dtype=np.float32),
        )

    def train(self, num_batches: int = 1) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for batch_index in range(num_batches):
            batch_list = []
            for episode_offset in range(self.episodes_per_batch):
                env = self.env_factory(seed=self.seed + batch_index * 100 + episode_offset)
                rollout_seed = self.seed + batch_index * 100 + episode_offset
                batch_list.append(self.collect_episode(env, explore=True, seed=rollout_seed))
                env.close()
            merged = EpisodeBatch(
                observations=np.concatenate([batch.observations for batch in batch_list], axis=0),
                masks=np.concatenate([batch.masks for batch in batch_list], axis=0),
                actions=np.concatenate([batch.actions for batch in batch_list], axis=0),
                returns=np.concatenate([batch.returns for batch in batch_list], axis=0),
            )
            metrics = self.update_from_batch(merged)
            metrics["steps"] = float(len(merged.actions))
            history.append(metrics)
        return history

    def evaluate_policy(self, num_episodes: int = 2) -> dict[str, float]:
        returns = []
        for episode_index in range(num_episodes):
            env = self.env_factory(seed=self.seed + 1000 + episode_index)
            rollout_seed = self.seed + 1000 + episode_index
            env.reset(seed=rollout_seed)
            total = 0.0
            for agent in env.agent_iter():
                observation, reward, termination, truncation, _ = env.last()
                if agent == "player_0":
                    total += float(reward)
                if termination or truncation:
                    env.step(None)
                    continue
                env.step(self.select_action(observation, explore=False))
            returns.append(total)
            env.close()
        return {"eval_return": float(np.mean(returns)), "eval_episodes": float(num_episodes)}
