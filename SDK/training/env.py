from __future__ import annotations

from functools import lru_cache

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from SDK.actions import ActionCatalog
from SDK.backend.core import EngineBackend, load_backend
from SDK.backend.state import BackendState, create_python_backend_state
from SDK.features import FeatureExtractor


class AntWarParallelEnv(ParallelEnv):
    metadata = {"name": "antwar_parallel_v0", "render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        max_actions: int = 96,
        prefer_native_backend: bool = False,
        backend: EngineBackend | None = None,
    ) -> None:
        self.base_seed = seed
        self.max_actions = max_actions
        self.backend = backend or load_backend(prefer_native=prefer_native_backend)
        self.feature_extractor = FeatureExtractor(max_actions=max_actions)
        self.action_catalog = ActionCatalog(max_actions=max_actions, feature_extractor=self.feature_extractor)
        self.possible_agents = ["player_0", "player_1"]
        self.agents: list[str] = []
        self._state: BackendState | None = None
        self._bundles: dict[int, list] = {0: [], 1: []}

    @property
    def state(self) -> BackendState:
        assert self._state is not None
        return self._state

    @lru_cache(maxsize=2)
    def observation_space(self, agent: str):
        initial_state = create_python_backend_state()
        board_shape = self.feature_extractor.encode_board(initial_state, 0).shape
        stats_shape = self.feature_extractor.encode_stats(initial_state, 0).shape
        return spaces.Dict(
            {
                "board": spaces.Box(low=-10.0, high=10.0, shape=board_shape, dtype=np.float32),
                "stats": spaces.Box(low=-10.0, high=10.0, shape=stats_shape, dtype=np.float32),
                "action_mask": spaces.MultiBinary(self.max_actions),
            }
        )

    @lru_cache(maxsize=2)
    def action_space(self, agent: str):
        return spaces.Discrete(self.max_actions)

    def _refresh_bundles(self) -> None:
        for player in (0, 1):
            self._bundles[player] = self.action_catalog.build(self.state, player)

    def _observe(self, player: int) -> dict[str, np.ndarray]:
        bundles = self._bundles[player]
        mask = self.action_catalog.action_mask(bundles)
        return self.feature_extractor.encode_observation(self.state, player, mask)

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is None:
            seed = self.base_seed
        self._state = self.backend.initial_state(seed)
        self.agents = list(self.possible_agents)
        self._refresh_bundles()
        observations = {agent: self._observe(index) for index, agent in enumerate(self.possible_agents)}
        infos = {
            agent: {
                "backend": self.backend.name,
                "bundles": self._bundles[index],
            }
            for index, agent in enumerate(self.possible_agents)
        }
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        previous_hp = [base.hp for base in self.state.bases]
        previous_coins = list(self.state.coins)

        chosen = {}
        illegal = {}
        for index, agent in enumerate(self.possible_agents):
            bundles = self._bundles[index]
            selected = int(actions.get(agent, 0)) if agent in actions else 0
            if not (0 <= selected < len(bundles)):
                chosen[index] = bundles[0]
                illegal[agent] = True
            else:
                chosen[index] = bundles[selected]
                illegal[agent] = False

        resolution = self.state.resolve_turn(chosen[0].operations, chosen[1].operations)
        terminations = {agent: self.state.terminal for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        rewards = {}
        for index, agent in enumerate(self.possible_agents):
            enemy = 1 - index
            reward = (previous_hp[enemy] - self.state.bases[enemy].hp) * 10.0
            reward -= (previous_hp[index] - self.state.bases[index].hp) * 10.0
            reward += (self.state.coins[index] - previous_coins[index]) * 0.05
            reward -= (self.state.coins[enemy] - previous_coins[enemy]) * 0.02
            if self.state.terminal:
                if self.state.winner == index:
                    reward += 100.0
                elif self.state.winner == enemy:
                    reward -= 100.0
            if illegal[agent]:
                reward -= 1.0
            rewards[agent] = float(reward)

        if self.state.terminal:
            observations = {agent: self._observe(index) for index, agent in enumerate(self.possible_agents)}
            infos = {
                agent: {
                    "bundle": chosen[index],
                    "illegal": illegal[agent],
                    "invalid_ops": resolution.illegal[index],
                }
                for index, agent in enumerate(self.possible_agents)
            }
            self.agents = []
            return observations, rewards, terminations, truncations, infos

        self._refresh_bundles()
        observations = {agent: self._observe(index) for index, agent in enumerate(self.possible_agents)}
        infos = {
            agent: {
                "bundle": chosen[index],
                "illegal": illegal[agent],
                "invalid_ops": resolution.illegal[index],
                "bundles": self._bundles[index],
            }
            for index, agent in enumerate(self.possible_agents)
        }
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        self.agents = []


def env(**kwargs) -> AntWarParallelEnv:
    return AntWarParallelEnv(**kwargs)
