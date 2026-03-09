from __future__ import annotations

from dataclasses import dataclass, field
import math

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
from AI.common import BaseAgent

from SDK.actions import ActionBundle
from SDK.backend.state import BackendState


@dataclass(slots=True)
class SearchNode:
    state: BackendState
    player: int
    bundle: ActionBundle | None = None
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    depth: int = 0
    children: list[SearchNode] = field(default_factory=list)
    unexplored: list[ActionBundle] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


@dataclass(slots=True)
class PhaseWeights:
    hp: float
    safety: float
    tempo: float
    offense: float
    economy: float


class HeuristicFallbackAgent(BaseAgent):
    def _phase_weights(self, state: BackendState, player: int) -> PhaseWeights:
        hp_delta = state.bases[player].hp - state.bases[1 - player].hp
        nearest_enemy = state.nearest_ant_distance(player)
        safe_coin = state.coins[player] - state.safe_coin_threshold(player)
        if nearest_enemy <= 4 or hp_delta < 0:
            return PhaseWeights(hp=1.4, safety=1.5, tempo=0.5, offense=0.3, economy=0.4)
        if hp_delta > 0 and state.frontline_distance(player) <= 8:
            return PhaseWeights(hp=0.8, safety=0.4, tempo=1.0, offense=1.5, economy=0.8)
        if safe_coin < 0:
            return PhaseWeights(hp=1.0, safety=1.3, tempo=0.5, offense=0.2, economy=1.4)
        return PhaseWeights(hp=1.1, safety=1.0, tempo=1.0, offense=0.9, economy=0.9)

    def _predict_enemy_bundle(self, state: BackendState, player: int) -> ActionBundle:
        enemy_bundles = self.list_bundles(state, 1 - player)
        return enemy_bundles[0] if enemy_bundles else self.list_bundles(state, player)[0]

    def _score_bundle(
        self,
        state: BackendState,
        player: int,
        bundle: ActionBundle,
        enemy_bundle: ActionBundle,
    ) -> float:
        trial = state.clone()
        if player == 0:
            trial.resolve_turn(bundle.operations, enemy_bundle.operations)
        else:
            trial.resolve_turn(enemy_bundle.operations, bundle.operations)
        summary = self.feature_extractor.summarize(trial, player).named
        weights = self._phase_weights(state, player)
        score = bundle.score
        score += summary["hp_delta"] * 12.0 * weights.hp
        score += summary["safe_coin"] * 0.03 * weights.economy
        score += summary["frontline_advantage"] * 1.8 * weights.offense
        score -= summary["enemy_progress"] * 0.7 * weights.safety
        score += summary["my_progress"] * 0.5 * weights.offense
        score += summary["generation_level"] * 4.0 * weights.tempo
        score += summary["ant_level"] * 6.0 * weights.tempo
        score += summary["kill_delta"] * 2.5 * weights.hp
        score += summary["tower_spread"] * 0.8
        if bundle.tags and bundle.tags[0] == "weapon" and state.coins[player] < state.safe_coin_threshold(player):
            score -= 8.0
        return score

    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        enemy_bundle = self._predict_enemy_bundle(state, player)
        shortlist = bundles[: min(18, len(bundles))]
        scored = [(self._score_bundle(state, player, bundle, enemy_bundle), bundle) for bundle in shortlist]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored else bundles[0]


class MCTSAgent(BaseAgent):
    def __init__(self, iterations: int = 64, max_depth: int = 4, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.iterations = iterations
        self.max_depth = max_depth
        self.opponent_model = HeuristicFallbackAgent(seed=seed)

    def _expand(self, node: SearchNode) -> None:
        if node.unexplored:
            bundle = node.unexplored.pop(0)
            child_state = node.state.clone()
            enemy_bundle = self.opponent_model.choose_bundle(child_state, 1 - node.player)
            if node.player == 0:
                child_state.resolve_turn(bundle.operations, enemy_bundle.operations)
            else:
                child_state.resolve_turn(enemy_bundle.operations, bundle.operations)
            child = SearchNode(
                state=child_state,
                player=node.player,
                bundle=bundle,
                prior=max(bundle.score, 0.0) + 1.0,
                depth=node.depth + 1,
                unexplored=self.catalog.build(child_state, node.player)[:10],
            )
            node.children.append(child)

    def _uct(self, parent: SearchNode, child: SearchNode) -> float:
        exploration = 1.25 * child.prior * math.sqrt(parent.visits + 1) / (child.visits + 1)
        return child.mean_value + exploration

    def _rollout(self, state: BackendState, player: int, depth: int) -> float:
        rollout = state.clone()
        for _ in range(depth, self.max_depth):
            if rollout.terminal:
                break
            my_bundle = self.opponent_model.choose_bundle(rollout, player)
            enemy_bundle = self.opponent_model.choose_bundle(rollout, 1 - player)
            if player == 0:
                rollout.resolve_turn(my_bundle.operations, enemy_bundle.operations)
            else:
                rollout.resolve_turn(enemy_bundle.operations, my_bundle.operations)
        return self.feature_extractor.evaluate(rollout, player)

    def _simulate(self, root: SearchNode) -> None:
        path = [root]
        node = root
        while node.children and node.depth < self.max_depth and not node.state.terminal:
            node = max(node.children, key=lambda child: self._uct(path[-1], child))
            path.append(node)
        if node.depth < self.max_depth and not node.state.terminal:
            self._expand(node)
            if node.children:
                node = node.children[-1]
                path.append(node)
        value = self._rollout(node.state, node.player, node.depth)
        for current in reversed(path):
            current.visits += 1
            current.value_sum += value

    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        root = SearchNode(
            state=state.clone(),
            player=player,
            unexplored=bundles[: min(12, len(bundles))],
        )
        if not root.unexplored:
            return bundles[0]
        for _ in range(self.iterations):
            self._simulate(root)
        if not root.children:
            return root.unexplored[0]
        best = max(root.children, key=lambda child: (child.visits, child.mean_value))
        return best.bundle if best.bundle is not None else bundles[0]


class AI(MCTSAgent):
    pass
