import json
import select
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _pack(obj: dict[str, Any]) -> bytes:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _recv_one(proc: subprocess.Popen, timeout: float = 1.0):
    rlist, _, _ = select.select([proc.stdout], [], [], timeout)
    if not rlist:
        return None
    hdr = proc.stdout.read(8)
    if not hdr or len(hdr) < 8:
        return None
    length, target = struct.unpack(">Ii", hdr)
    body = proc.stdout.read(length)
    if len(body) < length:
        return None
    try:
        obj = json.loads(body.decode("utf-8"))
    except Exception:
        obj = body.decode("utf-8", errors="replace")
    return target, obj


def _collect(proc: subprocess.Popen, timeout: float = 1.0):
    out = []
    end = time.time() + timeout
    while time.time() < end:
        msg = _recv_one(proc, timeout=0.05)
        if msg is None:
            continue
        out.append(msg)
    return out


def _parse_content_items(round_msg: dict[str, Any]) -> list[Any]:
    parsed: list[Any] = []
    for item in round_msg.get("content", []):
        if not isinstance(item, str):
            continue
        try:
            parsed.append(json.loads(item))
        except Exception:
            pass
    return parsed


def _get_web_round_state(round_msg: dict[str, Any]) -> dict[str, Any] | None:
    for obj in _parse_content_items(round_msg):
        if isinstance(obj, dict) and "towers" in obj and "ants" in obj:
            return obj
    return None


def _start_logic(tmp_path: Path):
    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    replay_file = tmp_path / "replay.json"
    init = {
        "player_list": [2, 1],  # p0 web, p1 ai
        "player_num": 2,
        "config": {"random_seed": 1},
        "replay": str(replay_file),
    }
    proc.stdin.write(_pack(init))
    proc.stdin.flush()
    return proc


def _stop_logic(proc: subprocess.Popen):
    try:
        # send abnormal to force end
        end_msg = {"player": -1, "content": json.dumps({"player": 0, "error": 0})}
        proc.stdin.write(_pack(end_msg))
        proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=1.0)
    except Exception:
        proc.kill()


def test_turn_switch_web_ai_has_no_noop_action_packet():
    with tempfile.TemporaryDirectory() as td:
        proc = _start_logic(Path(td))
        try:
            _collect(proc, timeout=0.8)  # initial config + round message + AI KM message

            # web player ends turn (empty action list; logic auto-appends end marker)
            proc.stdin.write(_pack({"player": 0, "content": "[]"}))
            proc.stdin.flush()
            msgs = _collect(proc, timeout=0.8)

            # no round-message content should be explicit [] no-op packets
            for tgt, obj in msgs:
                if tgt != -1 or not isinstance(obj, dict) or "content" not in obj:
                    continue
                for content_obj in _parse_content_items(obj):
                    assert content_obj != []

            # and must contain a proper round_state for web with player == 1 (AI's turn)
            round_msgs = [obj for tgt, obj in msgs if tgt == -1 and isinstance(obj, dict) and "content" in obj]
            assert round_msgs, "expected at least one round message after web turn end"
            web_states = [_get_web_round_state(m) for m in round_msgs]
            web_states = [s for s in web_states if s is not None]
            assert web_states, "expected a web round_state payload"
            assert any(int(s.get("player", -1)) == 1 for s in web_states)
        finally:
            _stop_logic(proc)


def test_turn_switch_back_to_web_after_ai_end_turn():
    with tempfile.TemporaryDirectory() as td:
        proc = _start_logic(Path(td))
        try:
            _collect(proc, timeout=0.8)

            proc.stdin.write(_pack({"player": 0, "content": "[]"}))
            proc.stdin.flush()
            _collect(proc, timeout=0.8)

            # ai ends its turn
            proc.stdin.write(_pack({"player": 1, "content": "8\n"}))
            proc.stdin.flush()
            msgs = _collect(proc, timeout=0.8)

            round_msgs = [obj for tgt, obj in msgs if tgt == -1 and isinstance(obj, dict) and "content" in obj]
            assert round_msgs, "expected round message after ai ends turn"
            web_states = [_get_web_round_state(m) for m in round_msgs]
            web_states = [s for s in web_states if s is not None]
            assert web_states, "expected web round_state after ai turn"
            assert any(int(s.get("player", -1)) == 0 for s in web_states)
        finally:
            _stop_logic(proc)
