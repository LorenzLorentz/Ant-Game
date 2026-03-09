from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from SDK.actions import ActionBundle, ActionCatalog
from SDK.constants import MAX_ACTIONS
from SDK.engine import GameState
from SDK.features import FeatureExtractor
from SDK.model import Operation


@dataclass(slots=True)
class AgentContext:
    state: GameState
    player: int
    bundles: list[ActionBundle]


class MatchSession(ABC):
    @property
    @abstractmethod
    def player(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def perform_self_turn(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def receive_opponent_turn(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def sync_round(self) -> bool:
        raise NotImplementedError


class BaseAgent(ABC):
    def __init__(self, seed: int | None = None, max_actions: int = MAX_ACTIONS) -> None:
        self._seed_override = seed
        self.rng = random.Random(seed)
        self.feature_extractor = FeatureExtractor(max_actions=max_actions)
        self.catalog = ActionCatalog(max_actions=max_actions, feature_extractor=self.feature_extractor)

    def list_bundles(self, state: GameState, player: int) -> list[ActionBundle]:
        return self.catalog.build(state, player)

    def on_match_start(self, player: int, seed: int) -> None:
        if self._seed_override is None:
            self.rng.seed((seed << 1) ^ player)

    def on_self_operations(self, operations) -> None:
        del operations

    def on_opponent_operations(self, operations) -> None:
        del operations

    def on_round_state(self, public_round_state) -> None:
        del public_round_state

    @abstractmethod
    def choose_bundle(self, state: GameState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        raise NotImplementedError

    def choose_operations(self, state: GameState, player: int, bundles: list[ActionBundle] | None = None) -> list[Operation]:
        return list(self.choose_bundle(state, player, bundles=bundles).operations)

    def choose_action_index(self, state: GameState, player: int, bundles: list[ActionBundle] | None = None) -> int:
        bundles = bundles or self.list_bundles(state, player)
        target = self.choose_bundle(state, player, bundles=bundles)
        for index, bundle in enumerate(bundles):
            if bundle.operations == target.operations:
                return index
        return 0
