from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import struct
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = REPO_ROOT / "game"
GAME_BIN = GAME_DIR / "output" / "main"


@lru_cache(maxsize=1)
def _ensure_game_binary() -> None:
    subprocess.run(["make"], cwd=GAME_DIR, check=True, capture_output=True, text=True)


def _packet(message: object) -> bytes:
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _prefixed_text_packet(text: str) -> str:
    payload = text.encode("utf-8")
    return (struct.pack(">I", len(payload)) + payload).decode("latin1")


def _run_game(input_packets: bytes) -> subprocess.CompletedProcess[bytes]:
    _ensure_game_binary()
    return subprocess.run(
        [str(GAME_BIN)],
        cwd=GAME_DIR,
        input=input_packets,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_cpp_game_accepts_null_random_seed(tmp_path: Path) -> None:
    replay_path = tmp_path / "null-seed-replay.json"
    init_packet = _packet(
        {
            "player_list": [1, 1],
            "player_num": 2,
            "config": {"random_seed": None},
            "replay": str(replay_path),
        }
    )
    error_packet = _packet(
        {
            "player": -1,
            "content": json.dumps({"player": 0, "error": 0}),
            "time": 0,
        }
    )

    completed = _run_game(init_packet + error_packet)
    stderr = completed.stderr.decode("utf-8", errors="replace")

    assert completed.returncode == 0
    assert "type_error" not in stderr
    assert replay_path.exists()


def test_cpp_game_decodes_length_prefixed_ai_operations(tmp_path: Path) -> None:
    replay_path = tmp_path / "prefixed-ops-replay.json"
    init_packet = _packet(
        {
            "player_list": [1, 1],
            "player_num": 2,
            "config": {"random_seed": 7},
            "replay": str(replay_path),
        }
    )
    round0_packet = _packet(
        {
            "player": 0,
            "content": _prefixed_text_packet("1\n11 6 9\n"),
            "time": 0,
        }
    )
    round1_packet = _packet(
        {
            "player": 1,
            "content": _prefixed_text_packet("0\n"),
            "time": 0,
        }
    )
    error_packet = _packet(
        {
            "player": -1,
            "content": json.dumps({"player": 0, "error": 0}),
            "time": 0,
        }
    )

    completed = _run_game(init_packet + round0_packet + round1_packet + error_packet)
    stderr = completed.stderr.decode("utf-8", errors="replace")

    assert completed.returncode == 0
    assert "Undefined type" not in stderr
    assert "read from judger error" not in stderr

    replay = json.loads(replay_path.read_text())
    assert any(op["type"] == 11 for entry in replay for op in entry.get("op0", []))
    assert any(
        tower["pos"]["x"] == 6 and tower["pos"]["y"] == 9
        for entry in replay
        for tower in entry.get("round_state", {}).get("towers", [])
    )
    assert all("pheromone" not in entry.get("round_state", {}) for entry in replay)
