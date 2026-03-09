#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = REPO_ROOT / "game"
DEFAULT_GAME_BIN = GAME_DIR / "output" / "main"
PACKAGE_AI = REPO_ROOT / "AI" / "package_ai.sh"
TIMEOUT_SECONDS = 20.0


def make_game(game_bin: Path) -> None:
    if game_bin == DEFAULT_GAME_BIN:
        subprocess.run(["make"], cwd=GAME_DIR, check=True)


def packet(payload: object) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def read_exact(stream, size: int, proc: subprocess.Popen[bytes], label: str,
               timeout: float = TIMEOUT_SECONDS) -> bytes:
    fd = stream.fileno()
    data = bytearray()
    deadline = time.monotonic() + timeout
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out while reading {label}")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            continue
        chunk = os.read(fd, size - len(data))
        if not chunk:
            code = proc.poll()
            if code is None:
                raise EOFError(f"unexpected EOF while reading {label}")
            raise EOFError(f"{label} closed with exit code {code}")
        data.extend(chunk)
    return bytes(data)


def read_game_packet(game: subprocess.Popen[bytes]) -> tuple[int, bytes]:
    size = struct.unpack(">I", read_exact(game.stdout, 4, game, "game packet length"))[0]
    obj = struct.unpack(">i", read_exact(game.stdout, 4, game, "game packet object"))[0]
    payload = read_exact(game.stdout, size, game, "game packet payload")
    return obj, payload


def read_ai_packet(ai: subprocess.Popen[bytes], name: str) -> bytes:
    size = struct.unpack(">I", read_exact(ai.stdout, 4, ai, f"{name} packet length"))[0]
    payload = read_exact(ai.stdout, size, ai, f"{name} packet payload")
    return struct.pack(">I", size) + payload


def write_all(stream, payload: bytes) -> None:
    stream.write(payload)
    stream.flush()


def stage_ai(target: str, parent: Path, label: str) -> Path:
    output_dir = parent / f"{label}-{target}"
    output_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run([str(PACKAGE_AI), target, str(output_dir)], cwd=REPO_ROOT, check=True)
    return output_dir


def launch_ai(ai_dir: Path, stderr_path: Path) -> subprocess.Popen[bytes]:
    stderr_handle = stderr_path.open("wb")
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=ai_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
    )


def terminate(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def close_stdin(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.stdin is None:
        return
    try:
        proc.stdin.close()
    except OSError:
        pass


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a full local game<->AI<->judger match")
    parser.add_argument("--ai0", default="greedy", choices=["random", "mcts", "greedy"])
    parser.add_argument("--ai1", default="greedy", choices=["random", "mcts", "greedy"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep-dir", type=Path, default=None,
                        help="Keep packaged AIs, replay, and stderr logs in this directory")
    parser.add_argument("--game-bin", type=Path, default=DEFAULT_GAME_BIN)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    game_bin = args.game_bin.resolve()
    make_game(game_bin)

    workdir_obj = args.keep_dir
    tempdir_obj = None
    if workdir_obj is None:
        tempdir_obj = tempfile.TemporaryDirectory(prefix="agent-tradition-match-")
        workdir_obj = Path(tempdir_obj.name)
    workdir = workdir_obj.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    replay_path = workdir / "replay.json"
    game_stderr_path = workdir / "game.stderr.log"
    ai0_stderr_path = workdir / "ai0.stderr.log"
    ai1_stderr_path = workdir / "ai1.stderr.log"

    ai_stage_root = workdir / "ais"
    if ai_stage_root.exists():
        shutil.rmtree(ai_stage_root)
    ai_stage_root.mkdir(parents=True)

    ai0_dir = stage_ai(args.ai0, ai_stage_root, "ai0")
    ai1_dir = stage_ai(args.ai1, ai_stage_root, "ai1")

    game_stderr_handle = game_stderr_path.open("wb")
    game = None
    ai0 = None
    ai1 = None
    result: dict[str, object] = {
        "ai0": args.ai0,
        "ai1": args.ai1,
        "seed": args.seed,
        "workdir": str(workdir),
        "replay": str(replay_path),
    }
    events: list[dict[str, object]] = []

    def record_event(kind: str, **payload: object) -> None:
        entry = {"kind": kind, **payload}
        events.append(entry)
        if len(events) > 30:
            del events[0]
        if args.verbose:
            print(json.dumps(entry, ensure_ascii=False), flush=True)

    try:
        ai0 = launch_ai(ai0_dir, ai0_stderr_path)
        ai1 = launch_ai(ai1_dir, ai1_stderr_path)
        ais = {0: ai0, 1: ai1}

        game = subprocess.Popen(
            [str(game_bin)],
            cwd=GAME_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=game_stderr_handle,
        )

        init = {
            "player_list": [1, 1],
            "player_num": 2,
            "config": {"random_seed": args.seed},
            "replay": str(replay_path),
        }
        init_packet = packet(init)
        write_all(game.stdin, init_packet)
        record_event("send_init", size=len(init_packet))

        while True:
            obj, payload = read_game_packet(game)
            record_event("game_packet", object=obj, size=len(payload))
            if obj in (0, 1):
                write_all(ais[obj].stdin, payload)
                record_event("forward_to_ai", player=obj, size=len(payload))
                continue

            message = json.loads(payload.decode("utf-8"))
            if isinstance(message, dict) and "player" in message and "content" in message:
                for player, content in zip(message["player"], message["content"]):
                    write_all(ais[int(player)].stdin, content.encode("utf-8"))
                    record_event("broadcast_to_ai", player=int(player), size=len(content))
            if isinstance(message, dict) and message.get("listen"):
                for player in message["listen"]:
                    ai_packet = read_ai_packet(ais[int(player)], f"ai{player}")
                    record_event("ai_reply", player=int(player), size=len(ai_packet))
                    reply = {
                        "player": int(player),
                        "content": ai_packet.decode("latin1"),
                        "time": 0,
                    }
                    write_all(game.stdin, packet(reply))
                    record_event("send_to_game", player=int(player), size=len(reply["content"]))
            if isinstance(message, dict) and "end_state" in message:
                result["end_state"] = message["end_state"]
                result["end_info"] = message.get("end_info")
                break

        game.wait(timeout=3)
        close_stdin(ai0)
        close_stdin(ai1)
        for ai in (ai0, ai1):
            try:
                ai.wait(timeout=3)
            except subprocess.TimeoutExpired:
                terminate(ai)
        result["game_returncode"] = game.returncode
        result["ai0_returncode"] = ai0.returncode
        result["ai1_returncode"] = ai1.returncode
        result["game_stderr"] = read_text(game_stderr_path)
        result["ai0_stderr"] = read_text(ai0_stderr_path)
        result["ai1_stderr"] = read_text(ai1_stderr_path)
        if replay_path.exists():
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            result["rounds_recorded"] = len(replay)
            if replay:
                last_round = replay[-1].get("round_state", {})
                result["last_winner"] = last_round.get("winner")
                result["last_error"] = last_round.get("error")
        result["events"] = events
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["events"] = events
        if game is not None:
            result["game_returncode"] = game.poll()
        if ai0 is not None:
            result["ai0_returncode"] = ai0.poll()
        if ai1 is not None:
            result["ai1_returncode"] = ai1.poll()
        result["game_stderr"] = read_text(game_stderr_path)
        result["ai0_stderr"] = read_text(ai0_stderr_path)
        result["ai1_stderr"] = read_text(ai1_stderr_path)
        result["replay_exists"] = replay_path.exists()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    finally:
        terminate(ai0)
        terminate(ai1)
        terminate(game)
        game_stderr_handle.close()
        if tempdir_obj is not None:
            tempdir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
