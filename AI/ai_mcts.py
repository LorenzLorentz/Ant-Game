from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.alphazero import PolicyValueNet, PriorGuidedMCTS, SearchConfig, infer_observation_dim
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle
from SDK.utils.constants import MAX_ACTIONS, OperationType
from SDK.utils.turns import DecisionContext


def _patch_backend_state() -> None:
    if getattr(BackendState, "_zz_agent_patch_applied", False):
        return

    original_can_apply = BackendState.can_apply_operation

    def patched_can_apply(self, player: int, operation, pending=None) -> bool:
        op_type = getattr(operation, "op_type", getattr(operation, "type", None))
        if op_type == OperationType.UPGRADE_GENERATION_SPEED and self.bases[player].generation_level >= 2:
            return False
        if op_type == OperationType.UPGRADE_GENERATED_ANT and self.bases[player].ant_level >= 2:
            return False
        try:
            return original_can_apply(self, player, operation, pending)
        except IndexError:
            return False

    BackendState.can_apply_operation = patched_can_apply
    BackendState._zz_agent_patch_applied = True


@dataclass(slots=True)
class SearchProfile:
    iterations: int
    max_depth: int
    root_action_limit: int
    child_action_limit: int
    c_puct: float
    prior_mix: float
    value_mix: float


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        iterations: int = 96,
        max_depth: int = 5,
        seed: int | None = None,
        max_actions: int = MAX_ACTIONS,
        model_path: str | os.PathLike[str] | None = None,
        shortlist_size: int = 18,
    ) -> None:
        _patch_backend_state()
        super().__init__(seed=seed, max_actions=max_actions)
        self.base_iterations = iterations
        self.base_depth = max_depth
        self.shortlist_size = max(8, shortlist_size)
        self.model = self._load_model(model_path)
        self.search = PriorGuidedMCTS(
            model=self.model,
            search_config=SearchConfig(
                iterations=iterations,
                max_depth=max_depth,
                c_puct=1.2,
                root_action_limit=min(self.shortlist_size, 16),
                child_action_limit=8,
                prior_mix=0.75,
                value_mix=0.7,
                seed=seed or 0,
            ),
            feature_extractor=self.feature_extractor,
            action_catalog=self.catalog,
        )

    def _candidate_model_paths(self, override: str | os.PathLike[str] | None) -> list[Path]:
        candidates: list[Path] = []
        if override is not None:
            return [Path(override)]
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

    def _search_profile(self, state: BackendState, bundles: list[ActionBundle]) -> SearchProfile:
        round_index = state.round_index
        bundle_count = len(bundles)
        if round_index < 80:
            return SearchProfile(
                iterations=max(self.base_iterations - 24, 48),
                max_depth=max(self.base_depth - 1, 3),
                root_action_limit=min(bundle_count, 12),
                child_action_limit=6,
                c_puct=1.05,
                prior_mix=0.82,
                value_mix=0.60,
            )
        if round_index < 220:
            return SearchProfile(
                iterations=self.base_iterations,
                max_depth=self.base_depth,
                root_action_limit=min(bundle_count, 16),
                child_action_limit=8,
                c_puct=1.15,
                prior_mix=0.76,
                value_mix=0.70,
            )
        return SearchProfile(
            iterations=self.base_iterations + 24,
            max_depth=self.base_depth + 1,
            root_action_limit=min(bundle_count, 20),
            child_action_limit=10,
            c_puct=1.3,
            prior_mix=0.68,
            value_mix=0.78,
        )

    def _tag_bonus(self, bundle: ActionBundle, round_index: int) -> float:
        tags = set(bundle.tags)
        if round_index < 80:
            if "base" in tags:
                return 5.0
            if "build" in tags:
                return 3.0
            if "weapon" in tags:
                return -2.0
            return 0.0
        if round_index < 220:
            if "upgrade" in tags:
                return 2.5
            if "combo" in tags:
                return 1.5
            if "weapon" in tags:
                return 1.0
            return 0.0
        if "weapon" in tags:
            return 4.0
        if "sell" in tags:
            return -3.0
        if "base" in tags:
            return -1.5
        return 0.0

    def _shortlist_bundles(self, state: BackendState, bundles: list[ActionBundle]) -> list[ActionBundle]:
        if len(bundles) <= self.shortlist_size:
            return bundles

        round_index = state.round_index
        hold_bundle = bundles[0]
        rest = bundles[1:]
        ranked = sorted(
            rest,
            key=lambda bundle: bundle.score + self._tag_bonus(bundle, round_index),
            reverse=True,
        )

        keep: list[ActionBundle] = [hold_bundle]
        keep.extend(ranked[: self.shortlist_size - 1])

        # Always keep the strongest weapon line in the shortlist if one exists.
        weapon_bundle = next((bundle for bundle in ranked if "weapon" in bundle.tags), None)
        if weapon_bundle is not None and weapon_bundle not in keep:
            keep[-1] = weapon_bundle

        return keep

    def list_bundles(self, state: BackendState, player: int) -> list[ActionBundle]:
        bundles = self.catalog.build(
            state,
            player,
            context=DecisionContext.for_player(player),
            rerank=False,
        )
        return self._shortlist_bundles(state, bundles)

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        if not bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))

        bundles = self._shortlist_bundles(state, bundles)
        profile = self._search_profile(state, bundles)
        config = self.search.search_config
        config.iterations = profile.iterations
        config.max_depth = profile.max_depth
        config.root_action_limit = profile.root_action_limit
        config.child_action_limit = profile.child_action_limit
        config.c_puct = profile.c_puct
        config.prior_mix = profile.prior_mix
        config.value_mix = profile.value_mix

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
