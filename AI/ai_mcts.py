from __future__ import annotations

import os
from pathlib import Path
import math
import numpy as np
from dataclasses import dataclass, field
from numpy.lib.stride_tricks import as_strided

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common": raise
    from AI.common import BaseAgent

from SDK.alphazero import PolicyValueNet, PriorGuidedMCTS, SearchConfig, infer_observation_dim
from SDK.utils.actions import ActionBundle
from SDK.utils.constants import MAX_ACTIONS
from SDK.backend.state import BackendState
from SDK.utils.turns import DecisionContext


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        iterations: int = 64,
        max_depth: int = 4,
        seed: int | None = None,
        max_actions: int = MAX_ACTIONS,
        model_path: str | os.PathLike[str] | None = None,
        c_puct: float = 1.25,
        prior_mix: float = 0.7,
        value_mix: float = 0.7,
    ) -> None:
        super().__init__(seed=seed, max_actions=max_actions)
        self.model = self._load_model(model_path)
        self.search = PriorGuidedMCTS(
            model=self.model,
            search_config=SearchConfig(
                iterations=iterations,
                max_depth=max_depth,
                c_puct=c_puct,
                prior_mix=prior_mix,
                value_mix=value_mix,
                seed=seed or 0,
            ),
            feature_extractor=self.feature_extractor,
            action_catalog=self.catalog,
        )

    def _candidate_model_paths(self, override: str | os.PathLike[str] | None) -> list[Path]:
        candidates: list[Path] = []
        if override is not None:
            candidates.append(Path(override))
            return candidates
        env_path = os.getenv("AGENT_TRADITION_MCTS_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        module_root = Path(__file__).resolve().parent
        repo_root = module_root.parent
        candidates.extend(
            [
                module_root / "ai_mcts_model.npz",
                repo_root / "checkpoints" / "ai_mcts_latest.npz",
                repo_root / "SDK" / "checkpoints" / "ai_mcts_latest.npz",
            ]
        )
        return candidates

    def _load_model(self, model_path: str | os.PathLike[str] | None) -> PolicyValueNet | None:
        expected_obs_dim = infer_observation_dim(self.feature_extractor, self.catalog.max_actions)
        for candidate in self._candidate_model_paths(model_path):
            if not candidate.exists():
                continue
            try:
                model = PolicyValueNet.from_checkpoint(candidate)
            except (OSError, ValueError, KeyError):
                continue
            if model.action_dim != self.catalog.max_actions:
                continue
            if model.obs_dim != expected_obs_dim:
                continue
            return model
        return None

    def list_bundles(self, state: BackendState, player: int) -> list[ActionBundle]:
        return self.catalog.build(state, player, context=DecisionContext.for_player(player), rerank=False)

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        if not bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))
        result = self.search.search(
            state=state,
            player=player,
            bundles=bundles,
            context=DecisionContext.for_player(player),
            temperature=1e-6,
            add_root_noise=False,
        )
        return result.bundle
from SDK.utils.constants import PLAYER_BASES

# ==========================================
# 🐛 官方 SDK Bug 修复补丁
# ==========================================
original_can_apply = BackendState.can_apply_operation
def patched_can_apply(self, player: int, operation, pending=None) -> bool:
    if operation.type == 31 and self.bases[player].generation_level >= 3: return False
    if operation.type == 32 and self.bases[player].ant_level >= 3: return False
    try: return original_can_apply(self, player, operation, pending)
    except IndexError: return False
BackendState.can_apply_operation = patched_can_apply

# ==========================================
# 🚀 Numpy 神经网络推理引擎
# ==========================================
def conv2d_numpy(x, w, b, padding=1):
    x_pad = np.pad(x, ((0,0), (padding, padding), (padding, padding)), mode='constant')
    shape = (x_pad.shape[0], x_pad.shape[1] - w.shape[2] + 1, x_pad.shape[2] - w.shape[3] + 1, w.shape[2], w.shape[3])
    strides = (x_pad.strides[0], x_pad.strides[1], x_pad.strides[2], x_pad.strides[1], x_pad.strides[2])
    windows = as_strided(x_pad, shape=shape, strides=strides)
    return np.tensordot(w, windows, axes=([1, 2, 3], [0, 3, 4])) + b.reshape(-1, 1, 1)

def relu(x): return np.maximum(0, x)

class NumpyValueNet:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.w_c1 = data['conv1.weight']; self.b_c1 = data['conv1.bias']
        self.w_c2 = data['conv2.weight']; self.b_c2 = data['conv2.bias']
        self.w_fc1 = data['fc1.weight'];  self.b_fc1 = data['fc1.bias']
        self.w_fc2 = data['fc2.weight'];  self.b_fc2 = data['fc2.bias']

    def forward(self, x):
        x = conv2d_numpy(x, self.w_c1, self.b_c1, padding=1)
        x = relu(x)
        x = conv2d_numpy(x, self.w_c2, self.b_c2, padding=1)
        x = relu(x)
        x = x.flatten()
        x = np.dot(self.w_fc1, x) + self.b_fc1
        return float(np.tanh(np.dot(self.w_fc2, relu(x)) + self.b_fc2)[0])

def state_to_tensor(state: BackendState, player: int) -> np.ndarray:
    tensor = np.zeros((9, 19, 19), dtype=np.float32)
    enemy = 1 - player
    for ant in state.ants_of(player):
        if 0 <= ant.x < 19 and 0 <= ant.y < 19: tensor[0, ant.x, ant.y] += ant.hp / 100.0
    for ant in state.ants_of(enemy):
        if 0 <= ant.x < 19 and 0 <= ant.y < 19: tensor[1, ant.x, ant.y] += ant.hp / 100.0
    for tower in state.towers_of(player):
        if 0 <= tower.x < 19 and 0 <= tower.y < 19: tensor[2, tower.x, tower.y] = tower.level / 3.0
    for tower in state.towers_of(enemy):
        if 0 <= tower.x < 19 and 0 <= tower.y < 19: tensor[3, tower.x, tower.y] = tower.level / 3.0
    bx, by = PLAYER_BASES[player]; ex, ey = PLAYER_BASES[enemy]
    if 0 <= bx < 19 and 0 <= by < 19:
        tensor[4, bx, by] = 1.0; tensor[5, bx, by] = state.bases[player].hp / 50.0; tensor[6, bx, by] = (state.bases[player].generation_level + state.bases[player].ant_level) / 6.0
    if 0 <= ex < 19 and 0 <= ey < 19:
        tensor[4, ex, ey] = -1.0; tensor[5, ex, ey] = state.bases[enemy].hp / 50.0; tensor[7, ex, ey] = (state.bases[enemy].generation_level + state.bases[enemy].ant_level) / 6.0
    tensor[8, :, :] = (state.coins[player] - state.coins[enemy]) / 200.0
    return tensor

@dataclass(slots=True)
class SearchNode:
    state: BackendState
    player: int
    bundle: ActionBundle | None = None
    visits: int = 0; value_sum: float = 0.0; prior: float = 0.0; depth: int = 0
    children: list[SearchNode] = field(default_factory=list)
    unexplored: list[ActionBundle] = field(default_factory=list)
    @property
    def mean_value(self) -> float: return 0.0 if self.visits == 0 else self.value_sum / self.visits

# 极简随机对手预测（抛弃长篇大论的启发式对手）
class RandomFallbackAgent(BaseAgent):
    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        return bundles[0] if bundles else ActionBundle([])

class MCTSAgent(BaseAgent):
    # 实战中迭代 64 次（非常快，完全不超时），依靠大模型的精准估值取胜
    def __init__(self, iterations: int = 64, max_depth: int = 5, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.iterations = iterations
        self.max_depth = max_depth 
        self.opponent_model = RandomFallbackAgent(seed=seed)
        self.nn_model = None
        for path in ["model_weights.npz", os.path.join(os.path.dirname(__file__), "model_weights.npz")]:
            if os.path.exists(path):
                self.nn_model = NumpyValueNet(path)
                break

    def _expand(self, node: SearchNode) -> None:
        if node.unexplored:
            bundle = node.unexplored.pop(0)
            try:
                child_state = node.state.clone()
                enemy_bundle = self.opponent_model.choose_bundle(child_state, 1 - node.player)
                if node.player == 0: child_state.resolve_turn(bundle.operations, enemy_bundle.operations)
                else: child_state.resolve_turn(enemy_bundle.operations, bundle.operations)
                
                raw_unexplored = self.catalog.build(child_state, node.player)[:6] # 只保留前6个最有潜力的动作
                clean_unexplored = [b for b in raw_unexplored if not any(
                    (getattr(op, 'type', -1) == 31 and child_state.bases[node.player].generation_level >= 3) or 
                    (getattr(op, 'type', -1) == 32 and child_state.bases[node.player].ant_level >= 3) for op in b.operations)]

                child = SearchNode(state=child_state, player=node.player, bundle=bundle,
                    prior=max(bundle.score, 0.0) + 1.0, depth=node.depth + 1, unexplored=clean_unexplored)
                node.children.append(child)
            except Exception: pass

    def _rollout(self, state: BackendState, player: int, depth: int) -> float:
        # 【大模型接管时刻】：完全信任神经网络的判断！
        if self.nn_model is not None:
            return self.nn_model.forward(state_to_tensor(state, player))
        return 0.0

    def _simulate(self, root: SearchNode) -> None:
        try:
            path = [root]; node = root
            while node.children and node.depth < self.max_depth and not node.state.terminal:
                node = max(node.children, key=lambda c: c.mean_value + 1.25 * c.prior * math.sqrt(path[-1].visits + 1) / (c.visits + 1))
                path.append(node)
            if node.depth < self.max_depth and not node.state.terminal:
                self._expand(node)
                if node.children:
                    node = node.children[-1]; path.append(node)
            value = self._rollout(node.state, node.player, node.depth)
            for current in reversed(path):
                current.visits += 1; current.value_sum += value
        except Exception: pass

    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        raw_bundles = bundles or self.list_bundles(state, player)
        clean_bundles = [b for b in raw_bundles if not any(
            (getattr(op, 'type', -1) == 31 and state.bases[player].generation_level >= 3) or 
            (getattr(op, 'type', -1) == 32 and state.bases[player].ant_level >= 3) for op in b.operations)]
        
        working_bundles = clean_bundles if clean_bundles else raw_bundles
        if len(working_bundles) <= 1: return working_bundles[0] if working_bundles else ActionBundle([])

        try:
            root = SearchNode(state=state.clone(), player=player, unexplored=working_bundles[: min(12, len(working_bundles))])
            for _ in range(self.iterations): self._simulate(root)
            if not root.children: return root.unexplored[0] if root.unexplored else working_bundles[0]
            return max(root.children, key=lambda c: (c.visits, c.mean_value)).bundle or working_bundles[0]
        except Exception: 
            return working_bundles[0]

class AI(MCTSAgent):
    pass