from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from typing import Iterable

try:
    from common import BaseAgent, MatchSession
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent, MatchSession

from SDK.engine import PublicRoundState
from SDK.runtime import MatchRuntime
from SDK.model import Operation
from SDK.constants import OperationType


@dataclass(slots=True)
class ProtocolController:
    runtime: MatchRuntime
    agent: BaseAgent

    @property
    def player(self) -> int:
        return self.runtime.player

    @property
    def state(self):
        return self.runtime.state

    def decide(self) -> list[Operation]:
        return self.agent.choose_operations(self.state, self.player)

    def apply_self_operations(self, operations: Iterable[Operation]) -> list[Operation]:
        return self.runtime.apply_self_operations(operations)

    def apply_opponent_operations(self, operations: Iterable[Operation]) -> list[Operation]:
        return self.runtime.apply_opponent_operations(operations)

    def finish_round(self, public_round_state: PublicRoundState) -> None:
        self.runtime.finish_round(public_round_state)


class ProtocolIO:
    def __init__(self, stdin=None, stdout=None, stderr=None) -> None:
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer
        self.stderr = stderr or sys.stderr

    def log(self, message: str) -> None:
        self.stderr.write(f"[AI] {message}\n")
        self.stderr.flush()

    def recv_line(self) -> str | None:
        raw = self.stdin.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace").rstrip("\n")

    def send_packet(self, payload: str) -> None:
        if not payload.endswith("\n"):
            payload += "\n"
        data = payload.encode("utf-8")
        self.stdout.write(struct.pack(">I", len(data)))
        self.stdout.write(data)
        self.stdout.flush()

    def recv_init(self) -> tuple[int, int]:
        line = self.recv_line()
        if line is None:
            raise RuntimeError("missing init line")
        player, seed = map(int, line.split())
        return player, seed

    def recv_operations(self) -> list[Operation]:
        line = self.recv_line()
        if line is None:
            raise RuntimeError("missing operation count")
        count = int(line.strip())
        operations: list[Operation] = []
        for _ in range(count):
            payload = self.recv_line()
            if payload is None:
                raise RuntimeError("unexpected EOF while reading operations")
            parts = [int(item) for item in payload.split()]
            op_type = OperationType(parts[0])
            if len(parts) == 1:
                operations.append(Operation(op_type))
            elif len(parts) == 2:
                operations.append(Operation(op_type, parts[1]))
            else:
                operations.append(Operation(op_type, parts[1], parts[2]))
        return operations

    def recv_round_state(self) -> PublicRoundState | None:
        line = self.recv_line()
        if line is None:
            return None
        round_index = int(line.strip())
        tower_count = int((self.recv_line() or "0").strip())
        towers = []
        for _ in range(tower_count):
            towers.append(tuple(map(int, (self.recv_line() or "").split())))
        ant_count = int((self.recv_line() or "0").strip())
        ants = []
        for _ in range(ant_count):
            ants.append(tuple(map(int, (self.recv_line() or "").split())))
        coins = tuple(map(int, (self.recv_line() or "0 0").split()[:2]))
        camps_hp = tuple(map(int, (self.recv_line() or "0 0").split()[:2]))
        return PublicRoundState(round_index=round_index, towers=towers, ants=ants, coins=coins, camps_hp=camps_hp)

    def send_operations(self, operations: Iterable[Operation]) -> None:
        items = list(operations)
        lines = [str(len(items))]
        lines.extend(" ".join(str(token) for token in operation.to_protocol_tokens()) for operation in items)
        self.send_packet("\n".join(lines) + "\n")


class ProtocolSession(MatchSession):
    def __init__(self, agent: BaseAgent, io: ProtocolIO | None = None) -> None:
        self.io = io or ProtocolIO()
        player, seed = self.io.recv_init()
        self.controller = ProtocolController(
            runtime=MatchRuntime.create(player=player, seed=seed, prefer_native=False),
            agent=agent,
        )
        agent.on_match_start(player, seed)

    @property
    def player(self) -> int:
        return self.controller.player

    def perform_self_turn(self) -> None:
        my_ops = self.controller.decide()
        self.controller.apply_self_operations(my_ops)
        self.controller.agent.on_self_operations(my_ops)
        self.io.send_operations(my_ops)

    def receive_opponent_turn(self) -> bool:
        try:
            opponent_ops = self.io.recv_operations()
        except Exception:
            return False
        self.controller.apply_opponent_operations(opponent_ops)
        self.controller.agent.on_opponent_operations(opponent_ops)
        return True

    def sync_round(self) -> bool:
        round_state = self.io.recv_round_state()
        if round_state is None:
            return False
        self.controller.finish_round(round_state)
        self.controller.agent.on_round_state(round_state)
        return True


def run_agent(agent: BaseAgent, io: ProtocolIO | None = None) -> None:
    try:
        from main import run_session
    except ModuleNotFoundError as exc:  # pragma: no cover - repository layout
        if exc.name != "main":
            raise
        from AI.main import run_session

    run_session(ProtocolSession(agent, io=io))
