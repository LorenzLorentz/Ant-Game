from __future__ import annotations

"""
Saiblo网站评测用网络Wrapper，提交ai时需要将此文件放入压缩包。

功能：
- 按照官方规则文档的“AI 编写指南”实现标准输入/输出通信协议：
  - 读取初始化行：`K M`，其中 K表示先手/后手，M 为随机种子。
  - 每回合读取一份“局面信息”（纯文本，多行），解析为 Python dict。
  - 调用暴露的策略函数（需要实现并import，本文件的示例为ai_grredy.py中的policy函数）。
    要替换使用的 AI，只需修改 `from ai_greedy import policy`这一行的导入目标，使其指向你自己的策略函数即可。
"""

import struct
import sys
from typing import Any, Dict, List, Tuple

from ai_greedy import policy


def log(msg: str) -> None:
    sys.stderr.write(f"[AI] {msg}\n")
    sys.stderr.flush()


def recv_line() -> str | None:
    b = sys.stdin.buffer.readline()
    if not b:
        return None
    try:
        return b.decode("utf-8").rstrip("\n")
    except Exception:
        return b.decode("utf-8", errors="replace").rstrip("\n")


def send_packet(text: str) -> None:
    """
    向评测系统发送一条消息
    """
    if not text.endswith("\n"):
        text += "\n"
    data = text.encode("utf-8")
    header = struct.pack(">I", len(data))
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(data)
    log(f"send packet: {text!r}")
    sys.stdout.buffer.flush()


def recv_init() -> Tuple[int, int]:
    """从评测系统读取初始化行（`K M`）。

    返回 (seat, seed)。seat==0 先手，seat==1 后手；seed 供策略使用。
    """
    line = recv_line()
    if line is None:
        raise RuntimeError("No init line (K M) received from judger.")
    parts = line.strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"Invalid init line: {line!r}")
    seat = int(parts[0])
    seed = int(parts[1])
    if seat not in (0, 1):
        raise RuntimeError(f"Invalid seat K={seat}, expected 0 or 1.")
    log(f"Init: seat={seat}, seed={seed}")
    return seat, seed


def _read_state_lines() -> list[str] | None:
    """读取原始的局面信息行并返回为字符串列表。

    该函数只负责按照协议获取
    R, N1+tower行, N2+ant行, coins, camps_hp；
    不做解析。调用者随后可以把列表传给
    ``_parse_state_from_lines`` 完成具体结构化。
    """
    lines: list[str] = []
    def _first_int(s: str, desc: str) -> int:
        parts = s.strip().split()
        if not parts:
            raise RuntimeError(f"Empty {desc} line: {s!r}")
        try:
            return int(parts[0])
        except ValueError:
            raise RuntimeError(f"Invalid {desc} line: {s!r}")

    line = recv_line()
    if line is None:
        return None
    lines.append(line)
    # 第1行：R
    R = _first_int(line, "round")

    # N1 and towers
    line = recv_line()
    if line is None:
        return None
    lines.append(line)
    N1 = _first_int(line, "N1")
    for _ in range(N1):
        l = recv_line()
        if l is None:
            raise RuntimeError("EOF while reading tower lines")
        lines.append(l)
    # N2 and ants
    line = recv_line()
    if line is None:
        return None
    lines.append(line)
    N2 = _first_int(line, "N2")
    for _ in range(N2):
        l = recv_line()
        if l is None:
            raise RuntimeError("EOF while reading ant lines")
        lines.append(l)
    # coins
    line = recv_line()
    if line is None:
        return None
    lines.append(line)
    # camps_hp
    line = recv_line()
    if line is None:
        return None
    lines.append(line)
    return lines


def _parse_state_from_lines(lines: list[str]) -> Dict[str, Any]:
    """将从 ``_read_state_lines`` 得到的原始行列表解析为状态字典。"""
    it = iter(lines)
    def _first_int_from(s: str) -> int:
        return int(s.strip().split()[0])

    R = _first_int_from(next(it))
    N1 = _first_int_from(next(it))
    towers = []
    for _ in range(N1):
        parts = next(it).strip().split()
        # accept at least 6 tokens, ignore extras
        if len(parts) < 6:
            raise RuntimeError(f"Invalid tower line: {parts}")
        tid, player, x, y, ttype, cd = map(int, parts[:6])
        towers.append((tid, player, x, y, ttype, cd))
    N2 = _first_int_from(next(it))
    ants = []
    for _ in range(N2):
        parts = next(it).strip().split()
        if len(parts) < 8:
            raise RuntimeError(f"Invalid ant line: {parts}")
        aid, player, x, y, hp, lv, age, state = map(int, parts[:8])
        ants.append((aid, player, x, y, hp, lv, age, state))
    coins_parts = next(it).strip().split()
    G0, G1 = map(int, coins_parts[:2])
    camps_parts = next(it).strip().split()
    HP0, HP1 = map(int, camps_parts[:2])
    return {
        "round": R,
        "towers": towers,
        "ants": ants,
        "coins": [G0, G1],
        "camps_hp": [HP0, HP1],
    }


def recv_state() -> Dict[str, Any] | None:
    """读取并解析一轮局面，返回状态字典或 None（EOF）。"""
    lines = _read_state_lines()
    if lines is None:
        return None
    for l in lines:
        log(f"line: {l}")
    return _parse_state_from_lines(lines)


def ops_to_text(ops: List[List[int]]) -> str:
    """把操作列表序列化为判题器/对手能理解的文本包。

    格式：首行 N 后跟 N 行每行一个操作。
    """
    n = len(ops)
    lines: List[str] = [str(n)]
    for op in ops:
        if not op:
            continue
        lines.append(" ".join(str(x) for x in op))
    return "\n".join(lines) + "\n"


def recv_ops() -> List[List[int]]:
    """从 judger 处接收一份对方的操作列表（无封装）。"""
    line = recv_line()
    if line is None:
        raise RuntimeError("EOF while waiting for opponent ops")
    log(f"opp line: {line}")
    try:
        N = int(line.strip())
    except Exception:
        raise RuntimeError(f"Invalid ops count: {line!r}")
    ops: List[List[int]] = []
    for _ in range(N):
        l = recv_line()
        if l is None:
            raise RuntimeError("EOF while reading opponent op")
        log(f"opp line: {l}")
        parts = l.strip().split()
        ops.append([int(x) for x in parts])
    return ops


def main() -> None:
    # 1) 初始化：读取 K M
    try:
        seat, seed = recv_init()
    except Exception as e:
        log(f"init failed: {e}")
        return
    del seed

    state: Dict[str, Any] | None = None

    # 2) 回合循环
    while True:
        log("round begin")
        if seat == 0:
            try:
                ops = policy(state, seat)
            except Exception as e:
                log(f"policy crashed: {e}")
                ops = []
            send_packet(ops_to_text(ops))
            # 等待后手操作
            try:
                _ = recv_ops()
            except Exception as e:
                log(f"recv opponent ops failed: {e}")
                break
            # 然后读取本回合局面
            state = recv_state()
            if state is None:
                log("state is none")
                break
        else:
            # 后手：先收对手操作
            try:
                _ = recv_ops()
            except Exception as e:
                log(f"recv opponent ops failed: {e}")
                break
            try:
                ops = policy(state, seat)
            except Exception as e:
                log(f"policy crashed: {e}")
                ops = []
            send_packet(ops_to_text(ops))
            state = recv_state()
            if state is None:
                log("state is none")
                break


if __name__ == "__main__":
    main()

