from __future__ import annotations

import io
import struct

from AI.protocol import ProtocolIO
from SDK.model import Operation
from SDK.training import AntWarParallelEnv
from SDK.constants import OperationType


def test_env_reset_and_step() -> None:
    env = AntWarParallelEnv(seed=21)
    observations, infos = env.reset(seed=21)
    assert set(observations) == {"player_0", "player_1"}
    actions = {"player_0": 0, "player_1": 0}
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    assert set(next_obs) == {"player_0", "player_1"}
    assert all(isinstance(value, float) for value in rewards.values())
    assert all(not flag for flag in truncations.values())
    env.close()


def test_protocol_send_and_receive_round_state() -> None:
    stdin = io.BytesIO(
        b"0 7\n"
        b"1\n"
        b"11 6 9\n"
        b"1\n"
        b"1\n"
        b"0 0 6 9 0 1\n"
        b"1\n"
        b"0 0 2 9 10 0 0 0 0\n"
        b"51 51\n"
        b"50 50\n"
    )
    stdout = io.BytesIO()
    proto = ProtocolIO(stdin=stdin, stdout=stdout, stderr=io.StringIO())
    assert proto.recv_init() == (0, 7)
    assert len(proto.recv_operations()) == 1
    round_state = proto.recv_round_state()
    assert round_state is not None
    assert round_state.round_index == 1
    proto.send_operations([Operation(OperationType.BUILD_TOWER, 6, 9)])
    payload = stdout.getvalue()
    packet_len = struct.unpack(">I", payload[:4])[0]
    assert packet_len == len(payload[4:])
