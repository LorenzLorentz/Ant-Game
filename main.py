import sys
import json
import struct
import traceback
from typing import Any, Dict, List


# =========================
# Debug
# =========================
def log(msg: str) -> None:
    sys.stderr.write(f"[logic] {msg}\n")
    sys.stderr.flush()


# =========================
# IO (platform doc)
# judger -> logic: [4B len][payload]
# logic  -> judger: [4B len][4B target][payload]
# =========================
def _read_exact_stdin(n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def recv_from_judger() -> bytes:
    header = _read_exact_stdin(4)
    if len(header) < 4:
        return b""
    length = struct.unpack(">I", header)[0]
    if length > 50_000_000:
        raise RuntimeError(f"invalid length from judger: {length}")
    payload = _read_exact_stdin(length)
    if len(payload) < length:
        return b""
    return payload


def send_to_judger(payload: bytes, target: int) -> None:
    header = struct.pack(">Ii", len(payload), int(target))
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(payload)
    sys.stdout.flush()


# =========================
# Judger<->Logic schemas
# =========================
def recv_init_info() -> Dict[str, Any]:
    data = recv_from_judger()
    if not data:
        return {}
    try:
        d = json.loads(data.decode("utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def recv_ai_msg() -> Dict[str, Any]:
    data = recv_from_judger()
    if not data:
        return {"player": -1, "content": ""}
    try:
        d = json.loads(data.decode("utf-8"))
        return d if isinstance(d, dict) else {"player": -1, "content": ""}
    except Exception:
        return {"player": -1, "content": ""}


def send_round_config(time_limit_sec: float, max_length_bytes: int) -> None:
    msg = {"state": 0, "time": float(time_limit_sec), "length": int(max_length_bytes)}
    send_to_judger(json.dumps(msg, ensure_ascii=False).encode("utf-8"), target=-1)


def send_round_message(state_id: int,
                       listen: List[int],
                       player: List[int],
                       content: List[str]) -> None:
    msg = {
        "state": int(state_id),
        "listen": listen,
        "player": player,
        "content": content,
    }
    send_to_judger(json.dumps(msg, ensure_ascii=False).encode("utf-8"), target=-1)


def forward_raw_to_ai(ai_id: int, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    send_to_judger(text.encode("utf-8"), target=ai_id)


def send_game_end(end_info_obj: Dict[str, Any], end_state_list: List[str]) -> None:
    keys_sorted = sorted(end_info_obj.keys(), key=lambda x: int(x))
    ordered = {k: end_info_obj[k] for k in keys_sorted}
    msg = {
        "state": -1,
        "end_info": json.dumps(ordered, ensure_ascii=False),
        "end_state": json.dumps(end_state_list, ensure_ascii=False),
    }
    send_to_judger(json.dumps(msg, ensure_ascii=False).encode("utf-8"), target=-1)


# =========================
# Ops parsing (AI 输出多行，末尾 8)
# =========================
def parse_ops_lines(text: str) -> List[List[int]]:
    ops: List[List[int]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            op = [int(x) for x in parts]
        except Exception:
            raise ValueError(f"non-integer op line: {ln!r}")
        ops.append(op)
        if op and op[0] == 8:
            break
    if not ops or ops[-1][0] != 8:
        ops.append([8])
    return ops


ERROR_MAP = {0: "RE", 1: "TLE", 2: "OLE"}


def main() -> int:
    gamestate = None
    try:
        from logic.gamestate import GameState, init_generals, update_round
        from logic.ai2logic import execute_single_command
        from logic.game_rules import is_game_over

        init_info = recv_init_info()
        if not init_info:
            raise RuntimeError("no init info from judger")

        player_list = init_info.get("player_list", [])
        player_num = int(init_info.get("player_num", len(player_list) if isinstance(player_list, list) else 2))
        cfg = init_info.get("config", {}) if isinstance(init_info.get("config", {}), dict) else {}
        seed = int(cfg.get("random_seed", 0))
        replay_path = init_info.get("replay", "replay.json")

        if player_num < 2:
            raise RuntimeError(f"player_num < 2: {player_num}")

        def ptype(i: int) -> int:
            if isinstance(player_list, list) and i < len(player_list):
                try:
                    return int(player_list[i])
                except Exception:
                    return 1
            return 1

        # init gamestate
        gamestate = GameState()
        init_generals(gamestate)
        gamestate.replay_file = replay_path

        # open replay (must be closed before end message)
        gamestate.replay_open(seed)

        # optional: send KM init to AI players (type==1)
        for pid in (0, 1):
            if ptype(pid) == 1:
                forward_raw_to_ai(pid, f"{pid} {seed}")

        # time limits
        TIME_LIMIT_AI = 3.0
        TIME_LIMIT_WEB = 180.0
        MAX_LEN = 2048

        def time_limit_for(pid: int) -> float:
            return TIME_LIMIT_WEB if ptype(pid) == 2 else TIME_LIMIT_AI

        # helper: build rep JSON for each player; include Turn field to gate action
        def rep_for(pid: int, turn: int) -> str:
            rep = gamestate.trans_state_to_init_json(pid)
            if not isinstance(rep, dict):
                rep = {}
            rep["Player"] = pid
            rep["Turn"] = turn
            return json.dumps(rep, ensure_ascii=False, separators=(",", ":")) + "\n"

        tick_state = 1
        turn = 0  # player0 starts
        end_state = ["OK", "OK"]
        winner = -1

        # first broadcast state + listen current turn
        send_round_config(time_limit_for(turn), MAX_LEN)
        send_round_message(
            state_id=tick_state,
            listen=[turn],
            player=[0, 1],
            content=[rep_for(0, turn), rep_for(1, turn)],
        )
        tick_state += 1

        while True:
            msg = recv_ai_msg()
            p = int(msg.get("player", -1))

            if p == -1:
                # abnormal
                try:
                    inner = json.loads(msg.get("content", "{}"))
                    bad = int(inner.get("player", turn))
                    err_code = int(inner.get("error", 0))
                    end_state[bad] = ERROR_MAP.get(err_code, "RE")
                    winner = 1 - bad
                except Exception:
                    end_state[turn] = "RE"
                    winner = 1 - turn

                gamestate.winner = winner
                # 补帧（异常时可能还没走到 update_round）
                gamestate.append_ant_replay_frame(force=True)
                break

            if p != turn:
                # ignore out-of-turn packets (robust)
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            try:
                ops = parse_ops_lines(content)
            except Exception:
                end_state[turn] = "RE"
                winner = 1 - turn
                gamestate.winner = winner
                gamestate.set_last_ops(turn, [[8]])
                gamestate.append_ant_replay_frame(force=True)
                break

            # record ops for replay
            gamestate.set_last_ops(turn, ops)

            # execute ops
            ok_all = True
            for op in ops:
                if not op:
                    continue
                if op[0] == 8:
                    break
                if not execute_single_command(turn, gamestate, op[0], op[1:]):
                    ok_all = False
                    break
                w = is_game_over(gamestate)
                if w in (0, 1):
                    winner = w
                    break

            if not ok_all:
                end_state[turn] = "IA"
                winner = 1 - turn
                gamestate.winner = winner
                gamestate.append_ant_replay_frame(force=True)
                break

            if winner in (0, 1):
                gamestate.winner = winner
                gamestate.append_ant_replay_frame(force=True)
                break

            # advance turn; settle after player1 acted
            if turn == 1:
                update_round(gamestate)  # this will append one replay frame (non-empty)
                w = is_game_over(gamestate)
                if w in (0, 1):
                    winner = w
                    gamestate.winner = winner
                    # update_round already wrote frame; no need to force another
                    break
                turn = 0
            else:
                turn = 1

            # send next state
            send_round_config(time_limit_for(turn), MAX_LEN)
            send_round_message(
                state_id=tick_state,
                listen=[turn],
                player=[0, 1],
                content=[rep_for(0, turn), rep_for(1, turn)],
            )
            tick_state += 1

        if winner not in (0, 1):
            winner = 0

        # close replay before end message (finish IO)
        gamestate.replay_close()

        # simple scoring
        end_info = {"0": 1 if winner == 0 else 0, "1": 1 if winner == 1 else 0}
        send_game_end(end_info, end_state)
        return 0

    except Exception:
        log("!!! FATAL ERROR !!!")
        log(traceback.format_exc())
        try:
            if gamestate is not None:
                gamestate.replay_close()
        except Exception:
            pass
        try:
            send_game_end({"0": 0, "1": 0}, ["RE", "RE"])
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
