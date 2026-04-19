from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import random

import numpy as np

from SDK.backend import create_python_backend_state
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle, ActionCatalog
from SDK.utils.features import FeatureExtractor
from SDK.utils.turns import DecisionContext


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits.astype(np.float32, copy=False)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted).astype(np.float32, copy=False)
    total = float(np.sum(exp))
    if total <= 0.0:
        result = np.zeros_like(logits, dtype=np.float32)
        result[0] = 1.0
        return result
    return (exp / total).astype(np.float32, copy=False)


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = logits.astype(np.float32, copy=True)
    masked[mask <= 0] = -1e9
    if np.all(mask <= 0):
        result = np.zeros_like(masked, dtype=np.float32)
        result[0] = 1.0
        return result
    return _softmax(masked)


def _normalize_policy(policy: np.ndarray, fallback_index: int = 0) -> np.ndarray:
    total = float(np.sum(policy))
    if total <= 0.0:
        fallback = np.zeros_like(policy, dtype=np.float32)
        if fallback.size:
            fallback[min(max(fallback_index, 0), fallback.size - 1)] = 1.0
        return fallback
    return (policy / total).astype(np.float32, copy=False)


def _heuristic_bundle_policy(bundles: list[ActionBundle]) -> np.ndarray:
    if not bundles:
        return np.zeros(0, dtype=np.float32)
    scores = np.asarray([bundle.score for bundle in bundles], dtype=np.float32)
    centered = (scores - np.max(scores)) / 8.0
    return _softmax(centered)


def _terminal_value(state: BackendState, player: int) -> float | None:
    if not state.terminal:
        return None
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner == player else -1.0


@dataclass(slots=True)
class PolicyValueNetConfig:
    hidden_dim: int = 128
    hidden_dim2: int = 64
    seed: int = 0


@dataclass(slots=True)
class PolicyValueInference:
    priors: np.ndarray
    value: float
    observation: np.ndarray
    mask: np.ndarray


@dataclass(slots=True)
class SearchConfig:
    iterations: int = 64
    max_depth: int = 4
    c_puct: float = 1.25
    root_action_limit: int = 16
    child_action_limit: int = 10
    dirichlet_alpha: float = 0.35
    dirichlet_epsilon: float = 0.25
    prior_mix: float = 0.7
    value_mix: float = 0.7
    value_scale: float = 350.0
    seed: int = 0


@dataclass(slots=True)
class SearchResult:
    action_index: int
    bundle: ActionBundle
    policy: np.ndarray
    root_value: float
    visit_count: int
    priors: np.ndarray


@dataclass(slots=True)
class SearchNode:
    state: BackendState
    context: DecisionContext
    prior: float = 0.0
    bundle: ActionBundle | None = None
    action_index: int = 0
    depth: int = 0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    bundles: list[ActionBundle] = field(default_factory=list)
    priors: np.ndarray | None = None
    children: list[SearchNode] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def to_play(self) -> int:
        return self.context.to_play


class PolicyValueNet:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PolicyValueNetConfig | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or PolicyValueNetConfig()
        rng = np.random.default_rng(self.config.seed)
        scale1 = 1.0 / math.sqrt(max(obs_dim, 1))
        scale2 = 1.0 / math.sqrt(max(self.config.hidden_dim, 1))
        scale3 = 1.0 / math.sqrt(max(self.config.hidden_dim2, 1))
        self.w1 = rng.normal(0.0, scale1, size=(obs_dim, self.config.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.config.hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, scale2, size=(self.config.hidden_dim, self.config.hidden_dim2)).astype(np.float32)
        self.b2 = np.zeros(self.config.hidden_dim2, dtype=np.float32)
        self.policy_w = rng.normal(0.0, scale3, size=(self.config.hidden_dim2, action_dim)).astype(np.float32)
        self.policy_b = np.zeros(action_dim, dtype=np.float32)
        self.value_w = rng.normal(0.0, scale3, size=(self.config.hidden_dim2, 1)).astype(np.float32)
        self.value_b = np.zeros(1, dtype=np.float32)
        self.loaded_from: str | None = None

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> PolicyValueNet:
        checkpoint = np.load(Path(path), allow_pickle=False)
        config = PolicyValueNetConfig(
            hidden_dim=int(checkpoint["hidden_dim"]),
            hidden_dim2=int(checkpoint["hidden_dim2"]),
            seed=int(checkpoint["seed"]),
        )
        network = cls(obs_dim=int(checkpoint["obs_dim"]), action_dim=int(checkpoint["action_dim"]), config=config)
        network.w1 = checkpoint["w1"].astype(np.float32, copy=True)
        network.b1 = checkpoint["b1"].astype(np.float32, copy=True)
        network.w2 = checkpoint["w2"].astype(np.float32, copy=True)
        network.b2 = checkpoint["b2"].astype(np.float32, copy=True)
        network.policy_w = checkpoint["policy_w"].astype(np.float32, copy=True)
        network.policy_b = checkpoint["policy_b"].astype(np.float32, copy=True)
        network.value_w = checkpoint["value_w"].astype(np.float32, copy=True)
        network.value_b = checkpoint["value_b"].astype(np.float32, copy=True)
        network.loaded_from = str(Path(path))
        return network

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            obs_dim=np.int64(self.obs_dim),
            action_dim=np.int64(self.action_dim),
            hidden_dim=np.int64(self.config.hidden_dim),
            hidden_dim2=np.int64(self.config.hidden_dim2),
            seed=np.int64(self.config.seed),
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            policy_w=self.policy_w,
            policy_b=self.policy_b,
            value_w=self.value_w,
            value_b=self.value_b,
        )
        self.loaded_from = str(target)

    def _forward(
        self,
        observations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hidden1_pre = observations @ self.w1 + self.b1
        hidden1 = _relu(hidden1_pre)
        hidden2_pre = hidden1 @ self.w2 + self.b2
        hidden2 = _relu(hidden2_pre)
        logits = hidden2 @ self.policy_w + self.policy_b
        raw_values = hidden2 @ self.value_w + self.value_b
        values = np.tanh(raw_values).astype(np.float32, copy=False)
        return hidden1_pre, hidden1, hidden2_pre, hidden2, logits.astype(np.float32, copy=False), values.astype(np.float32, copy=False)

    def predict(self, observation: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        batch = observation.astype(np.float32, copy=False)[None, :]
        _, _, _, _, logits, values = self._forward(batch)
        priors = _masked_softmax(logits[0], mask.astype(np.float32, copy=False))
        return priors, float(values[0, 0])

    def update(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        learning_rate: float = 1e-3,
        value_weight: float = 1.0,
        l2_weight: float = 1e-5,
    ) -> dict[str, float]:
        batch_size = max(len(observations), 1)
        obs = observations.astype(np.float32, copy=False)
        mask = masks.astype(np.float32, copy=False)
        target_policy = policy_targets.astype(np.float32, copy=False)
        target_value = value_targets.astype(np.float32, copy=False).reshape(-1, 1)

        hidden1_pre, hidden1, hidden2_pre, hidden2, logits, values = self._forward(obs)
        masked_logits = logits.astype(np.float32, copy=True)
        masked_logits[mask <= 0] = -1e9
        masked_logits -= np.max(masked_logits, axis=1, keepdims=True)
        exp = np.exp(masked_logits).astype(np.float32, copy=False) * mask
        denom = np.sum(exp, axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        probs = exp / denom

        target_policy = target_policy * mask
        policy_denominator = np.sum(target_policy, axis=1, keepdims=True)
        policy_denominator[policy_denominator <= 0] = 1.0
        target_policy = target_policy / policy_denominator

        safe_probs = np.clip(probs, 1e-8, 1.0)
        policy_loss = -np.sum(target_policy * np.log(safe_probs), axis=1).mean()
        value_error = values - target_value
        value_loss = float(np.mean(value_error ** 2))
        entropy = float(-np.mean(np.sum(np.where(probs > 0, probs * np.log(safe_probs), 0.0), axis=1)))

        grad_logits = (probs - target_policy) / batch_size
        grad_logits *= mask
        grad_policy_w = hidden2.T @ grad_logits + l2_weight * self.policy_w
        grad_policy_b = np.sum(grad_logits, axis=0)

        grad_raw_values = (2.0 * value_weight / batch_size) * value_error * (1.0 - values ** 2)
        grad_value_w = hidden2.T @ grad_raw_values + l2_weight * self.value_w
        grad_value_b = np.sum(grad_raw_values, axis=0)

        grad_hidden2 = grad_logits @ self.policy_w.T + grad_raw_values @ self.value_w.T
        grad_hidden2[hidden2_pre <= 0] = 0.0
        grad_w2 = hidden1.T @ grad_hidden2 + l2_weight * self.w2
        grad_b2 = np.sum(grad_hidden2, axis=0)

        grad_hidden1 = grad_hidden2 @ self.w2.T
        grad_hidden1[hidden1_pre <= 0] = 0.0
        grad_w1 = obs.T @ grad_hidden1 + l2_weight * self.w1
        grad_b1 = np.sum(grad_hidden1, axis=0)

        self.policy_w -= learning_rate * grad_policy_w.astype(np.float32)
        self.policy_b -= learning_rate * grad_policy_b.astype(np.float32)
        self.value_w -= learning_rate * grad_value_w.astype(np.float32)
        self.value_b -= learning_rate * grad_value_b.astype(np.float32)
        self.w2 -= learning_rate * grad_w2.astype(np.float32)
        self.b2 -= learning_rate * grad_b2.astype(np.float32)
        self.w1 -= learning_rate * grad_w1.astype(np.float32)
        self.b1 -= learning_rate * grad_b1.astype(np.float32)

        return {
            "policy_loss": float(policy_loss),
            "value_loss": value_loss,
            "entropy": entropy,
            "mean_value_target": float(np.mean(target_value)),
            "mean_prediction": float(np.mean(values)),
        }


class PriorGuidedMCTS:
    def __init__(
        self,
        model: PolicyValueNet | None = None,
        search_config: SearchConfig | None = None,
        feature_extractor: FeatureExtractor | None = None,
        action_catalog: ActionCatalog | None = None,
    ) -> None:
        self.model = model
        self.search_config = search_config or SearchConfig()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.action_catalog = action_catalog or ActionCatalog(feature_extractor=self.feature_extractor)
        self.rng = random.Random(self.search_config.seed)

    @property
    def action_dim(self) -> int:
        return self.action_catalog.max_actions

    def _heuristic_value(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext | None = None,
    ) -> float:
        terminal = _terminal_value(state, player)
        if terminal is not None:
            return terminal
        raw = self.feature_extractor.evaluate(state, player, context=context)
        return float(np.tanh(raw / self.search_config.value_scale))

    def _blend_policy_value(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext,
        bundles: list[ActionBundle],
    ) -> PolicyValueInference:
        action_mask = self.action_catalog.action_mask(bundles).astype(np.float32)
        observation = self.feature_extractor.encode_observation(state, player, action_mask, context=context)
        flat = self.feature_extractor.flatten_observation(observation)
        heuristic_priors = _heuristic_bundle_policy(bundles)
        heuristic_value = self._heuristic_value(state, player, context=context)
        if self.model is None:
            blended_priors = heuristic_priors
            blended_value = heuristic_value
        else:
            model_priors, model_value = self.model.predict(flat, action_mask)
            mixed_policy = self.search_config.prior_mix * model_priors[: len(bundles)]
            mixed_policy += (1.0 - self.search_config.prior_mix) * heuristic_priors
            blended_priors = _normalize_policy(mixed_policy)
            blended_value = float(
                self.search_config.value_mix * model_value
                + (1.0 - self.search_config.value_mix) * heuristic_value
            )
        full_priors = np.zeros(self.action_dim, dtype=np.float32)
        full_priors[: len(bundles)] = blended_priors
        return PolicyValueInference(
            priors=full_priors,
            value=float(blended_value),
            observation=flat,
            mask=action_mask,
        )

    def _predict_policy_only(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext,
        bundles: list[ActionBundle],
    ) -> np.ndarray:
        if not bundles:
            return np.zeros(self.action_dim, dtype=np.float32)
        return self._blend_policy_value(state, player, context, bundles).priors

    def _branch_indices(self, priors: np.ndarray, bundles: list[ActionBundle], limit: int) -> list[int]:
        if not bundles:
            return []
        branch_limit = min(limit, len(bundles))
        order = list(np.argsort(priors[: len(bundles)])[::-1])
        selected = order[:branch_limit]
        if 0 not in selected:
            selected.append(0)
        return sorted(set(int(index) for index in selected))

    def _expand(
        self,
        node: SearchNode,
        bundles: list[ActionBundle] | None = None,
        add_root_noise: bool = False,
        max_decision_depth: int | None = None,
    ) -> float:
        if node.expanded:
            return node.mean_value

        terminal = _terminal_value(node.state, node.to_play)
        if terminal is not None:
            node.expanded = True
            return terminal

        action_bundles = bundles or self.action_catalog.build(node.state, node.to_play, node.context, rerank=False)
        node.bundles = action_bundles
        inference = self._blend_policy_value(node.state, node.to_play, node.context, action_bundles)
        node.priors = inference.priors
        node.expanded = True

        depth_limit = self.search_config.max_depth if max_decision_depth is None else max_decision_depth
        if node.depth >= depth_limit or not action_bundles:
            return inference.value

        prior_slice = inference.priors[: len(action_bundles)].astype(np.float32, copy=True)
        if add_root_noise and len(action_bundles) > 1 and self.search_config.dirichlet_epsilon > 0.0:
            noise = np.random.default_rng(self.rng.randrange(1 << 30)).dirichlet(
                [self.search_config.dirichlet_alpha] * len(action_bundles)
            ).astype(np.float32)
            prior_slice = (
                (1.0 - self.search_config.dirichlet_epsilon) * prior_slice
                + self.search_config.dirichlet_epsilon * noise
            )
            prior_slice = _normalize_policy(prior_slice)
            node.priors[: len(action_bundles)] = prior_slice

        limit = self.search_config.root_action_limit if node.depth == 0 else self.search_config.child_action_limit
        for action_index in self._branch_indices(node.priors, action_bundles, limit):
            child_state = node.state.clone()
            bundle = action_bundles[action_index]
            child_state.apply_operation_list(node.to_play, bundle.operations)
            child_context = node.context.next_turn()
            if node.context.settles_after_action and not child_state.terminal:
                child_state.advance_round()
            node.children.append(
                SearchNode(
                    state=child_state,
                    context=child_context,
                    prior=float(node.priors[action_index]),
                    bundle=bundle,
                    action_index=action_index,
                    depth=node.depth + 1,
                )
            )
        return inference.value

    def _puct(self, parent: SearchNode, child: SearchNode) -> float:
        explore = self.search_config.c_puct * child.prior * math.sqrt(parent.visits + 1.0) / (child.visits + 1.0)
        return -child.mean_value + explore

    def _backpropagate(self, path: list[SearchNode], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _max_decision_depth(self, context: DecisionContext) -> int:
        base = max(int(self.search_config.max_depth), 1)
        if context.settles_after_action:
            return base * 2 - 1
        return base * 2

    def _sample_from_policy(self, policy: np.ndarray) -> int:
        threshold = self.rng.random()
        cumulative = 0.0
        for index, probability in enumerate(policy.tolist()):
            cumulative += probability
            if threshold <= cumulative:
                return index
        return int(np.argmax(policy))

    def _policy_from_visits(self, visits: np.ndarray, temperature: float) -> np.ndarray:
        if visits.size == 0:
            return visits.astype(np.float32, copy=False)
        if temperature <= 1e-6:
            policy = np.zeros_like(visits, dtype=np.float32)
            policy[int(np.argmax(visits))] = 1.0
            return policy
        scaled = np.power(np.maximum(visits, 1e-6), 1.0 / max(temperature, 1e-6)).astype(np.float32, copy=False)
        return _normalize_policy(scaled)

    def search(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
        context: DecisionContext | None = None,
        temperature: float = 0.0,
        add_root_noise: bool = False,
    ) -> SearchResult:
        if context is None:
            context = DecisionContext.for_player(player)
        root = SearchNode(state=state.clone(), context=context)
        max_decision_depth = self._max_decision_depth(context)
        root_value = self._expand(root, bundles=bundles, add_root_noise=add_root_noise, max_decision_depth=max_decision_depth)
        if not root.bundles:
            fallback = ActionBundle(name="hold", score=0.0, tags=("noop",))
            return SearchResult(
                action_index=0,
                bundle=fallback,
                policy=np.zeros(self.action_dim, dtype=np.float32),
                root_value=float(root_value),
                visit_count=0,
                priors=np.zeros(self.action_dim, dtype=np.float32),
            )

        for _ in range(self.search_config.iterations):
            node = root
            path = [root]
            while node.expanded and node.children and node.depth < max_decision_depth and not node.state.terminal:
                node = max(node.children, key=lambda child: self._puct(path[-1], child))
                path.append(node)
            if node.state.terminal or node.depth >= max_decision_depth:
                value = self._heuristic_value(node.state, node.to_play, context=node.context)
            else:
                value = self._expand(node, max_decision_depth=max_decision_depth)
            self._backpropagate(path, value)

        visit_counts = np.zeros(len(root.bundles), dtype=np.float32)
        for child in root.children:
            visit_counts[child.action_index] = float(child.visits)
        if float(np.sum(visit_counts)) <= 0.0:
            visit_counts = root.priors[: len(root.bundles)] if root.priors is not None else np.ones(len(root.bundles), dtype=np.float32)

        root_policy = self._policy_from_visits(visit_counts, temperature=temperature)
        if temperature <= 1e-6:
            action_index = int(np.argmax(visit_counts))
        else:
            action_index = self._sample_from_policy(root_policy)
        selected_bundle = root.bundles[action_index]
        full_policy = np.zeros(self.action_dim, dtype=np.float32)
        full_policy[: len(root.bundles)] = root_policy
        full_priors = np.zeros(self.action_dim, dtype=np.float32)
        if root.priors is not None:
            full_priors[: len(root.bundles)] = root.priors[: len(root.bundles)]
        return SearchResult(
            action_index=action_index,
            bundle=selected_bundle,
            policy=full_policy,
            root_value=float(root.mean_value if root.visits else root_value),
            visit_count=int(visit_counts[action_index]),
            priors=full_priors,
        )


def build_policy_value_net(
    feature_extractor: FeatureExtractor,
    action_dim: int,
    config: PolicyValueNetConfig | None = None,
) -> PolicyValueNet:
    state = create_python_backend_state()
    mask = np.zeros(action_dim, dtype=np.float32)
    observation = feature_extractor.encode_observation(state, 0, mask)
    obs_dim = len(feature_extractor.flatten_observation(observation))
    return PolicyValueNet(obs_dim=obs_dim, action_dim=action_dim, config=config)


def infer_observation_dim(
    feature_extractor: FeatureExtractor,
    action_dim: int,
) -> int:
    state = create_python_backend_state()
    mask = np.zeros(action_dim, dtype=np.float32)
    observation = feature_extractor.encode_observation(state, 0, mask)
    return len(feature_extractor.flatten_observation(observation))
