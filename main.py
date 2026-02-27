import sys
import json
import struct
import traceback
from typing import Any, Dict, List, Tuple


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


def recv_player_msg() -> Dict[str, Any]:
    """
    judger -> logic: normal message {player, content, time?}
    abnormal: {player:-1, content:"<json-string>"}
    """
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
    # target=ai_id => judger raw-forward to that AI (no wrapper)
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
# Parse ops
# - AI text: multi-line ints, usually includes 8
# - Web human: JSON string of list[{id,type,pos,args}]
# =========================
END_TYPES = {8, 255}  # 255: surrender on frontend


def _map_web_action_to_internal(item: Dict[str, Any]) -> List[int] | None:
    """
    Map Unity ActionRequest payloads to logic internal commands.
    Unsupported actions are ignored (None).
    """
    t = int(item.get("type", -1))
    if t == 255:
        return [255]
    if t == 8:
        return [8]

    tid = int(item.get("id", -1))
    args = int(item.get("args", -1))
    pos = item.get("pos", {}) if isinstance(item.get("pos", {}), dict) else {}
    x = int(pos.get("x", -1))
    y = int(pos.get("y", -1))

    # Tower build -> call general at cell
    if t == 11:
        if x != -1 and y != -1:
            return [7, x, y]
        return None

    # Tower upgrade -> general upgrade kind
    if t == 12:
        if tid == -1:
            return None
        kind = abs(args) % 10
        if kind in (1, 2, 3):
            return [3, tid, kind]
        return None

    # Base upgrade -> tech update(1-based in execute_single_command)
    if t == 31:
        return [5, 1]
    if t == 32:
        return [5, 2]

    # Props -> super weapon best effort
    if t == 21 and x != -1 and y != -1:
        return [6, 1, x, y]
    if t == 22 and x != -1 and y != -1:
        return [6, 4, x, y]
    if t == 23 and x != -1 and y != -1:
        return [6, 2, x, y]
    if t == 24 and x != -1 and y != -1:
        return [6, 2, x, y]

    # Already internal fallback (for mixed frontends)
    if 1 <= t <= 7:
        if x != -1 and y != -1:
            if args != -1:
                return [t, x, y, args]
            return [t, x, y]
        if tid != -1:
            if args != -1:
                return [t, tid, args]
            return [t, tid]
        return [t]

    return None

def parse_ai_text_ops(text: str) -> List[List[int]]:
    ops: List[List[int]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        op = [int(x) for x in parts]
        ops.append(op)
        if op and op[0] == 8:
            break
    # AI 协议常规要求末尾有 8；缺失就补上
    if not ops or ops[-1][0] != 8:
        ops.append([8])
    return ops


def parse_web_json_ops(text: str) -> List[List[int]]:
    """
    text is like:
      '[{"id":-1,"type":11,"pos":{"x":6,"y":4},"args":-1}]'
    Return internal op list-of-ints.
    Unity frontend sends one packet per committed turn, so append [8] if absent.
    """
    arr = json.loads(text)
    if isinstance(arr, dict):
        arr = [arr]
    if not isinstance(arr, list):
        arr = []

    ops: List[List[int]] = []
    has_end = False
    for item in arr:
        if not isinstance(item, dict):
            continue
        op = _map_web_action_to_internal(item)
        if op is None:
            continue
        ops.append(op)
        if op and op[0] in END_TYPES:
            has_end = True

    if not has_end:
        ops.append([8])
    return ops


def ensure_end_marker_if_turn_end(buf: List[List[int]]) -> None:
    if not buf:
        return
    if buf[-1][0] != 8:
        buf.append([8])


# =========================
# Content builders
# =========================
def json_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


# =========================
# Main
# =========================
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

        # init game
        gamestate = GameState()
        init_generals(gamestate)
        gamestate.replay_file = replay_path

        # align with docs if your init_coin() still returns 0
        if hasattr(gamestate, "coin") and isinstance(gamestate.coin, list) and len(gamestate.coin) == 2:
            if gamestate.coin[0] == 0 and gamestate.coin[1] == 0:
                gamestate.coin = [50, 50]

        # open replay (must close before end message)
        if hasattr(gamestate, "replay_open"):
            gamestate.replay_open(seed)

        # optional KM init to AIs (type==1)
        for pid in (0, 1):
            if ptype(pid) == 1:
                forward_raw_to_ai(pid, f"{pid} {seed}")

        # time limits
        TIME_LIMIT_AI = 3.0
        TIME_LIMIT_WEB = 180.0
        MAX_LEN = 2048

        def time_limit_for(pid: int) -> float:
            return TIME_LIMIT_WEB if ptype(pid) == 2 else TIME_LIMIT_AI

        # recipients for broadcast (all non-zero-started players)
        recipients: List[int] = [i for i in range(player_num) if ptype(i) != 0]
        web_recipients: List[int] = [i for i in recipients if ptype(i) == 2]

        # build message content per recipient:
        # - AI (type==1): send your rep JSON (Cells/Generals...) with Turn gating
        # - Web (type==2): send AntWar-like round_state snapshot
        def content_for(pid: int, turn: int) -> str:
            if ptype(pid) == 1:
                rep = gamestate.trans_state_to_init_json(pid)
                if not isinstance(rep, dict):
                    rep = {}
                rep["Player"] = pid
                rep["Turn"] = turn
                return json_line(rep)
            else:
                # web: send RenderDatum object directly
                if hasattr(gamestate, "_build_round_state"):
                    rs = gamestate._build_round_state()  # type: ignore[attr-defined]
                else:
                    rs = {"winner": getattr(gamestate, "winner", -1)}

                if not isinstance(rs, dict):
                    rs = {}
                rs.setdefault("towers", [])
                rs.setdefault("ants", [])
                rs.setdefault("pheromone", [])
                rs.setdefault("coins", [])
                rs.setdefault("speedLv", [])
                rs.setdefault("anthpLv", [])
                rs.setdefault("camps", [])
                rs.setdefault("winner", getattr(gamestate, "winner", -1))
                rs.setdefault("message", "[,]")
                rs.setdefault("end_msg", "")
                rs["player"] = turn
                return json_line(rs)

        # per-turn op buffer (human can send multiple messages before end)
        op_buf: List[List[List[int]]] = [[], []]

        # judger timing state
        judge_state = 1
        turn = 0  # player 0 starts
        end_state = ["OK", "OK"]
        winner = -1

        # send initial state to all + listen current turn
        send_round_config(time_limit_for(turn), MAX_LEN)
        send_round_message(
            state_id=judge_state,
            listen=[turn],
            player=recipients,
            content=[content_for(pid, turn) for pid in recipients],
        )

        while True:
            msg = recv_player_msg()
            p = int(msg.get("player", -1))

            # abnormal from judger
            if p == -1:
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
                # ensure replay has a final frame
                if hasattr(gamestate, "set_last_ops"):
                    gamestate.set_last_ops(0, op_buf[0])
                    gamestate.set_last_ops(1, op_buf[1])
                if hasattr(gamestate, "append_ant_replay_frame"):
                    gamestate.append_ant_replay_frame(force=True)
                break

            # allow explicit surrender even when message is out-of-turn
            if p in (0, 1) and p != turn and ptype(p) == 2:
                raw_other = msg.get("content", "")
                if not isinstance(raw_other, str):
                    raw_other = str(raw_other)
                try:
                    other_ops = parse_web_json_ops(raw_other)
                    if any(op and op[0] == 255 for op in other_ops):
                        end_state[p] = "OK"
                        winner = 1 - p
                        gamestate.winner = winner
                        if hasattr(gamestate, "set_last_ops"):
                            gamestate.set_last_ops(0, op_buf[0])
                            gamestate.set_last_ops(1, op_buf[1])
                        if hasattr(gamestate, "append_ant_replay_frame"):
                            gamestate.append_ant_replay_frame(force=True)
                        break
                except Exception:
                    pass

            # ignore out-of-turn
            if p != turn:
                continue

            raw_content = msg.get("content", "")
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            # parse ops by player type
            try:
                if ptype(turn) == 2:
                    ops = parse_web_json_ops(raw_content)
                else:
                    ops = parse_ai_text_ops(raw_content)
            except Exception as e:
                log(f"parse ops failed: p={turn}, err={e}")
                end_state[turn] = "RE"
                winner = 1 - turn
                gamestate.winner = winner
                if hasattr(gamestate, "set_last_ops"):
                    gamestate.set_last_ops(0, op_buf[0])
                    gamestate.set_last_ops(1, op_buf[1])
                if hasattr(gamestate, "append_ant_replay_frame"):
                    gamestate.append_ant_replay_frame(force=True)
                break

            # determine whether this message ends turn
            ended = any(op and op[0] in END_TYPES for op in ops)
            surrendered = any(op and op[0] == 255 for op in ops)

            # execute ops (excluding end marker)
            ok_all = True
            is_web_turn = ptype(turn) == 2
            is_ai_turn = ptype(turn) == 1
            for op in ops:
                if not op:
                    continue
                if op[0] in END_TYPES:
                    break
                if not execute_single_command(turn, gamestate, op[0], op[1:]):
                    # Keep matches robust: ignore invalid web/AI ops instead of immediate IA.
                    # (still consumes the command slot; turn end is controlled by [8])
                    if is_web_turn or is_ai_turn:
                        continue
                    ok_all = False
                    break
                w = is_game_over(gamestate)
                if w in (0, 1):
                    winner = w
                    gamestate.winner = w
                    break

            # record ops into per-turn buffer
            if ops:
                # store all ops including end marker if it exists; otherwise only executed ops
                # to keep replay faithful, include the explicit [8] only when user/AI ended
                for op in ops:
                    if not op:
                        continue
                    if op[0] in END_TYPES:
                        if ended:
                            op_buf[turn].append([8])
                        break
                    op_buf[turn].append(op)

            if surrendered:
                end_state[turn] = "OK"
                winner = 1 - turn
                gamestate.winner = winner
                if hasattr(gamestate, "set_last_ops"):
                    gamestate.set_last_ops(0, op_buf[0])
                    gamestate.set_last_ops(1, op_buf[1])
                if hasattr(gamestate, "append_ant_replay_frame"):
                    gamestate.append_ant_replay_frame(force=True)
                break

            if not ok_all:
                end_state[turn] = "IA"
                winner = 1 - turn
                gamestate.winner = winner
                if hasattr(gamestate, "set_last_ops"):
                    gamestate.set_last_ops(0, op_buf[0])
                    gamestate.set_last_ops(1, op_buf[1])
                if hasattr(gamestate, "append_ant_replay_frame"):
                    gamestate.append_ant_replay_frame(force=True)
                break

            if winner in (0, 1):
                # finalize replay frame
                if hasattr(gamestate, "set_last_ops"):
                    gamestate.set_last_ops(0, op_buf[0])
                    gamestate.set_last_ops(1, op_buf[1])
                if hasattr(gamestate, "append_ant_replay_frame"):
                    gamestate.append_ant_replay_frame(force=True)
                break

            # --------- key behavior split ----------
            if not ended:
                # Human interactive: send intermediate update to unlock UI
                # IMPORTANT: keep judge_state unchanged to avoid timer reset
                send_round_message(
                    state_id=judge_state,
                    listen=[turn],
                    player=[turn],
                    content=[content_for(turn, turn)],
                )
                continue

            # Turn ended: if end marker not present in buffer, append it
            ensure_end_marker_if_turn_end(op_buf[turn])

            # switch turn / settle
            if turn == 1:
                # before settle, push ops to gamestate for replay
                if hasattr(gamestate, "set_last_ops"):
                    gamestate.set_last_ops(0, op_buf[0])
                    gamestate.set_last_ops(1, op_buf[1])

                update_round(gamestate)  # will append replay frame + clear internal last_ops

                # doc-like: each full round grant both +1 coin (if you want this rule)
                if hasattr(gamestate, "coin") and isinstance(gamestate.coin, list) and len(gamestate.coin) == 2:
                    gamestate.coin[0] += 1
                    gamestate.coin[1] += 1

                op_buf = [[], []]

                w = is_game_over(gamestate)
                if w in (0, 1):
                    winner = w
                    gamestate.winner = w
                    break

                turn = 0
            else:
                turn = 1

            # new timing epoch for the next player
            judge_state += 1
            send_round_config(time_limit_for(turn), MAX_LEN)
            send_round_message(
                state_id=judge_state,
                listen=[turn],
                player=recipients,
                content=[content_for(pid, turn) for pid in recipients],
            )

        # end game
        if winner not in (0, 1):
            winner = 0

        # close replay before sending end message (finish IO)
        try:
            if hasattr(gamestate, "replay_close"):
                gamestate.replay_close()
        except Exception:
            pass

        end_info = {"0": 1 if winner == 0 else 0, "1": 1 if winner == 1 else 0}
        send_game_end(end_info, end_state)
        return 0

    except Exception:
        log("!!! FATAL ERROR !!!")
        log(traceback.format_exc())
        try:
            if gamestate is not None and hasattr(gamestate, "replay_close"):
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
