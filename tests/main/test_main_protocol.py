import io
import json
import struct
import sys

import pytest


def _pack_incoming(payload: bytes) -> bytes:
    """Judger -> logic: [4B len][payload]"""
    return struct.pack(">I", len(payload)) + payload


class _FakeIn:
    def __init__(self, data: bytes):
        self.buffer = io.BytesIO(data)


class _FakeOut:
    def __init__(self):
        self.buffer = io.BytesIO()

    def flush(self):
        # match sys.stdout API
        return None


def test_send_to_judger_header_and_body(monkeypatch):
    import main as m

    fake_out = _FakeOut()
    monkeypatch.setattr(sys, "stdout", fake_out, raising=False)

    payload = b"{\"hello\":1}"
    m.send_to_judger(payload, target=7)

    data = fake_out.buffer.getvalue()
    assert len(data) == 8 + len(payload)
    length, target = struct.unpack(">Ii", data[:8])
    assert length == len(payload)
    assert target == 7
    assert data[8:] == payload


def test_recv_from_judger_reads_full_header(monkeypatch):
    import main as m

    p = b"{\"a\":1}"
    fake_in = _FakeIn(_pack_incoming(p))
    monkeypatch.setattr(sys, "stdin", fake_in, raising=False)

    out = m.recv_from_judger()
    assert out == p


def test_recv_from_judger_alignment_two_messages(monkeypatch):
    import main as m

    p1 = b"{\"x\":1}"
    p2 = b"{\"y\":2}"
    data = _pack_incoming(p1) + _pack_incoming(p2)
    fake_in = _FakeIn(data)
    monkeypatch.setattr(sys, "stdin", fake_in, raising=False)

    out1 = m.recv_from_judger()
    out2 = m.recv_from_judger()
    assert out1 == p1
    assert out2 == p2


def test_recv_ai_msg_and_round_messages(monkeypatch):
    import main as m

    # Prepare a fake incoming AI message (judger->logic channel)
    ai_msg = {"player": 0, "content": "8\n"}
    fake_in = _FakeIn(_pack_incoming(json.dumps(ai_msg).encode("utf-8")))
    fake_out = _FakeOut()
    monkeypatch.setattr(sys, "stdin", fake_in, raising=False)
    monkeypatch.setattr(sys, "stdout", fake_out, raising=False)

    got = m.recv_player_msg()
    assert got == ai_msg

    # Verify round config and round message are correctly framed and targeted (-1)
    m.send_round_config(1.5, 1024)
    m.send_round_message(3, [0], [0, 1], ["a", "b"])

    buf = fake_out.buffer.getvalue()
    # Scan messages in the outgoing stream; each is [8B header][json payload]
    bio = io.BytesIO(buf)

    def read_one(b):
        hdr = b.read(8)
        if not hdr:
            return None
        L, tgt = struct.unpack(">Ii", hdr)
        body = b.read(L)
        return tgt, json.loads(body.decode("utf-8"))

    m1 = read_one(bio)
    m2 = read_one(bio)
    assert m1 is not None and m2 is not None
    # Order preserved: first config then message
    (t1, j1), (t2, j2) = m1, m2
    assert t1 == -1 and t2 == -1
    assert j1["state"] == 0 and j2["state"] == 3
    assert j2["listen"] == [0]
    assert j2["player"] == [0, 1]
    assert j2["content"] == ["a", "b"]


def test_forward_raw_to_ai_appends_newline_and_target(monkeypatch):
    import main as m

    fake_out = _FakeOut()
    monkeypatch.setattr(sys, "stdout", fake_out, raising=False)
    m.forward_raw_to_ai(5, "PING")

    data = fake_out.buffer.getvalue()
    L, tgt = struct.unpack(">Ii", data[:8])
    assert tgt == 5
    payload = data[8:8 + L]
    assert payload == b"PING\n"


def test_parse_ops_lines_appends_end_and_stops():
    import main as m

    text = "1 0 0 1 1\n2 1 1 3\n8\n1 0 0 1 1\n"
    ops = m.parse_ai_text_ops(text)
    # Should stop at the first 8 and not include any following lines
    assert ops[-1] == [8]
    assert all(isinstance(op, list) for op in ops)
