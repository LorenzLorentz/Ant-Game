from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from SDK.backend import EngineBackend, load_backend
from SDK.engine import GameState, PublicRoundState
from SDK.model import Operation


@dataclass(slots=True)
class MatchRuntime:
    player: int
    state: GameState

    @classmethod
    def create(
        cls,
        player: int,
        seed: int,
        *,
        prefer_native: bool = False,
        backend: EngineBackend | None = None,
    ) -> MatchRuntime:
        engine = backend or load_backend(prefer_native=prefer_native)
        return cls(player=player, state=engine.initial_state(seed=seed))

    @property
    def opponent(self) -> int:
        return 1 - self.player

    def apply_operations(self, player: int, operations: Iterable[Operation]) -> list[Operation]:
        return self.state.apply_operation_list(player, operations)

    def apply_self_operations(self, operations: Iterable[Operation]) -> list[Operation]:
        return self.apply_operations(self.player, operations)

    def apply_opponent_operations(self, operations: Iterable[Operation]) -> list[Operation]:
        return self.apply_operations(self.opponent, operations)

    def finish_round(self, public_round_state: PublicRoundState) -> None:
        self.state.advance_round()
        self.state.sync_public_round_state(public_round_state)

