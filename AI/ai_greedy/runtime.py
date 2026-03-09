from __future__ import annotations

try:
    from common import MatchSession
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import MatchSession

try:
    from protocol import ProtocolIO
except ModuleNotFoundError as exc:
    if exc.name != "protocol":
        raise
from AI.protocol import ProtocolIO

from SDK.constants import OperationType as SDKOperationType
from SDK.backend.forecast import ForecastOperation as Operation, ForecastState as GameInfo, build_forecast_state
from SDK.backend.runtime import MatchRuntime
from SDK.model import Operation as SDKOperation


def _to_sdk_operation(operation: Operation) -> SDKOperation:
    return SDKOperation(SDKOperationType(int(operation.type)), int(operation.arg0), int(operation.arg1))


def _to_greedy_info(state) -> GameInfo:
    return build_forecast_state(state)


class GreedySession(MatchSession):
    def __init__(self, agent, io: ProtocolIO | None = None) -> None:
        self.agent = agent
        self.io = io or ProtocolIO()
        player, seed = self.io.recv_init()
        self.runtime = MatchRuntime.create(player=player, seed=seed, prefer_native=False)

    @property
    def player(self) -> int:
        return self.runtime.player

    def perform_self_turn(self) -> None:
        proposed = [_to_sdk_operation(operation) for operation in self.agent(self.player, _to_greedy_info(self.runtime.state))]
        accepted: list[SDKOperation] = []
        for operation in proposed:
            if self.runtime.state.can_apply_operation(self.player, operation, accepted):
                accepted.append(operation)
        self.runtime.apply_self_operations(accepted)
        self.io.send_operations(accepted)

    def receive_opponent_turn(self) -> bool:
        try:
            opponent_operations = self.io.recv_operations()
        except Exception:
            return False
        self.runtime.apply_opponent_operations(opponent_operations)
        return True

    def sync_round(self) -> bool:
        round_state = self.io.recv_round_state()
        if round_state is None:
            return False
        self.runtime.finish_round(round_state)
        return True
