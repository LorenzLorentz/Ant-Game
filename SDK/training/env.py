from __future__ import annotations

from functools import lru_cache

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from SDK.utils.actions import ActionBundle, ActionCatalog
from SDK.backend.core import EngineBackend, load_backend
from SDK.backend.state import BackendState, create_python_backend_state
from SDK.utils.features import FeatureExtractor
from SDK.utils.turns import DecisionContext


class AntWarSequentialEnv(AECEnv):
    metadata = {"name": "antwar_sequential_v0", "render_modes": []}

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
        self.agent_name_mapping = {agent: index for index, agent in enumerate(self.possible_agents)}
        self.agents: list[str] = []
        self.agent_selection = self.possible_agents[0]
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, object]] = {}
        self._state: BackendState | None = None
        self._bundles: dict[int, list[ActionBundle]] = {0: [], 1: []}
        self._context = DecisionContext.initial()
        self._round_start_hp = (0, 0)
        self._round_start_coins = (0, 0)

    @property
    def state(self) -> BackendState:
        assert self._state is not None
        return self._state

    @property
    def decision_context(self) -> DecisionContext:
        return self._context

    def player_index(self, agent: str) -> int:
        return self.agent_name_mapping[agent]

    @lru_cache(maxsize=2)
    def observation_space(self, agent: str):
        del agent
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
        del agent
        return spaces.Discrete(self.max_actions)

    def _capture_round_start(self) -> None:
        self._round_start_hp = tuple(base.hp for base in self.state.bases)
        self._round_start_coins = tuple(self.state.coins)

    def _refresh_bundles(self) -> None:
        for player in (0, 1):
            self._bundles[player] = []
        if self.state.terminal:
            return
        player = self._context.to_play
        self._bundles[player] = self.action_catalog.build(self.state, player, self._context, rerank=False)

    def _action_mask_for_agent(self, agent: str) -> np.ndarray:
        player = self.player_index(agent)
        if self.state.terminal or player != self._context.to_play:
            return np.zeros(self.max_actions, dtype=np.int8)
        return self.action_catalog.action_mask(self._bundles[player])

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        return self.feature_extractor.encode_observation(
            self.state,
            self.player_index(agent),
            self._action_mask_for_agent(agent),
            context=self._context,
        )

    def _joint_observations(self) -> dict[str, dict[str, np.ndarray]]:
        return {agent: self.observe(agent) for agent in self.possible_agents}

    def _update_infos(self) -> None:
        for index, agent in enumerate(self.possible_agents):
            record = dict(self.infos.get(agent, {}))
            record.update(
                {
                    "backend": self.backend.name,
                    "decision_context": self._context,
                    "phase": self._context.phase.value,
                    "to_play": self._context.to_play,
                    "settles_after_action": self._context.settles_after_action,
                    "bundles": list(self._bundles[index]) if index == self._context.to_play and not self.state.terminal else [],
                }
            )
            self.infos[agent] = record

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is None:
            seed = self.base_seed
        self._state = self.backend.initial_state(seed)
        self._context = DecisionContext.initial()
        self.agents = list(self.possible_agents)
        self.agent_selection = self.possible_agents[0]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._capture_round_start()
        self._refresh_bundles()
        self._update_infos()
        observations = self._joint_observations()
        infos = {agent: dict(info) for agent, info in self.infos.items()}
        return observations, infos

    def _selected_bundle(self, player: int, action: int | None) -> tuple[ActionBundle, bool]:
        bundles = self._bundles[player]
        if not bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",)), True
        if action is None:
            return bundles[0], True
        selected = int(action)
        if 0 <= selected < len(bundles):
            return bundles[selected], False
        return bundles[0], True

    def _round_rewards(self) -> dict[str, float]:
        rewards: dict[str, float] = {}
        previous_hp = self._round_start_hp
        previous_coins = self._round_start_coins
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
            rewards[agent] = float(reward)
        return rewards

    def _step_single(self, action: int | None) -> None:
        if not self.agents:
            return

        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._clear_rewards()
        self._cumulative_rewards[agent] = 0.0
        player = self.player_index(agent)
        bundle, illegal = self._selected_bundle(player, action)
        invalid_ops = self.state.apply_operation_list(player, bundle.operations)
        self.infos[agent] = {
            **self.infos.get(agent, {}),
            "bundle": bundle,
            "illegal": illegal,
            "invalid_ops": invalid_ops,
        }

        rewards = {name: 0.0 for name in self.possible_agents}
        self._context = self._context.next_turn()
        completed_round = False

        if not self.state.terminal and player == 1:
            self.state.advance_round()
            completed_round = True

        if completed_round or self.state.terminal:
            rewards = self._round_rewards()
            if completed_round and not self.state.terminal:
                self._capture_round_start()

        if illegal:
            rewards[agent] -= 1.0

        for name in self.possible_agents:
            self.rewards[name] = float(rewards[name])
            self.terminations[name] = self.state.terminal
            self.truncations[name] = False

        self._refresh_bundles()
        self._update_infos()
        self._accumulate_rewards()

        if self.state.terminal:
            self._deads_step_first()
        else:
            self.agent_selection = self.possible_agents[self._context.to_play]

    def _step_joint(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        if self.agent_selection != self.possible_agents[0]:
            raise RuntimeError("joint stepping is only supported at the start of a round")

        self._step_single(actions.get("player_0", 0))
        if self.agents and not self.state.terminal and self.agent_selection == self.possible_agents[1]:
            self._step_single(actions.get("player_1", 0))

        observations = self._joint_observations()
        rewards = {agent: float(self.rewards.get(agent, 0.0)) for agent in self.possible_agents}
        terminations = {agent: bool(self.terminations.get(agent, False)) for agent in self.possible_agents}
        truncations = {agent: bool(self.truncations.get(agent, False)) for agent in self.possible_agents}
        infos = {agent: dict(self.infos.get(agent, {})) for agent in self.possible_agents}
        return observations, rewards, terminations, truncations, infos

    def step(self, action):
        if isinstance(action, dict):
            return self._step_joint(action)
        self._step_single(action)
        return None

    def render(self):
        return None

    def close(self):
        self.agents = []


AntWarParallelEnv = AntWarSequentialEnv


def env(**kwargs) -> AntWarSequentialEnv:
    return AntWarSequentialEnv(**kwargs)
