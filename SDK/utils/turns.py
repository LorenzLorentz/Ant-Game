from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TurnPhase(str, Enum):
    PLAYER_0_DECISION = "player_0_decision"
    PLAYER_1_DECISION = "player_1_decision"


@dataclass(slots=True, frozen=True)
class DecisionContext:
    phase: TurnPhase = TurnPhase.PLAYER_0_DECISION

    @classmethod
    def initial(cls) -> DecisionContext:
        return cls(TurnPhase.PLAYER_0_DECISION)

    @classmethod
    def for_player(cls, player: int) -> DecisionContext:
        return cls(TurnPhase.PLAYER_0_DECISION if int(player) == 0 else TurnPhase.PLAYER_1_DECISION)

    @property
    def to_play(self) -> int:
        return 0 if self.phase == TurnPhase.PLAYER_0_DECISION else 1

    @property
    def settles_after_action(self) -> bool:
        return self.phase == TurnPhase.PLAYER_1_DECISION

    @property
    def opponent_already_acted(self) -> bool:
        return self.phase == TurnPhase.PLAYER_1_DECISION

    def next_turn(self) -> DecisionContext:
        if self.phase == TurnPhase.PLAYER_0_DECISION:
            return DecisionContext(TurnPhase.PLAYER_1_DECISION)
        return DecisionContext(TurnPhase.PLAYER_0_DECISION)

