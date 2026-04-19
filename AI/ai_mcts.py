from __future__ import annotations

import os
from pathlib import Path

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
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


class AI(MCTSAgent):
    pass
