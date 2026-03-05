from __future__ import annotations

"""
AntGame 网站评测用 Python Wrapper。

功能：
- 按照官方规则文档的“AI 编写指南”实现标准输入/输出通信协议：
  - 读取初始化行：`K M`，其中 K∈{0,1} 表示先手/后手，M 为随机种子。
  - 每回合读取一份“局面信息”（纯文本，多行），解析为 Python dict。
  - 调用暴露的策略函数（当前为 `ai_greedy_ant.policy_from_site_state`），
    生成本回合的操作列表。
  - 将操作列表转为评测系统要求的文本格式：
        N
        <op1>
        ...
        <opN>
    并在前面加上 4 字节大端长度，通过 stdout 发送。

要替换使用的 AI，只需修改 `from ai_greedy_ant import policy_from_site_state`
这一行的导入目标，使其指向你自己的策略函数即可。
"""

import struct
import sys
from typing import Any, Dict, List, Tuple

from ai_greedy_ant import policy_from_site_state


def _log(msg: str) -> None:
    sys.stderr.write(f"[ANT_AI] {msg}\n")
    sys.stderr.flush()


def _recv_line() -> str | None:
    """读取一行 UTF-8 文本；EOF 时返回 None。"""
    b = sys.stdin.buffer.readline()
    if not b:
        return None
    try:
        return b.decode("utf-8").rstrip("\n")
    except Exception:
        return b.decode("utf-8", errors="replace").rstrip("\n")


def _send_packet(text: str) -> None:
    """
    向评测系统发送一条消息：
    - 内容为 text（已包含换行符），编码为 UTF-8。
    - 前缀为 4 字节大端序整数（消息字节长度）。
    """
    if not text.endswith("\n"):
        text += "\n"
    data = text.encode("utf-8")
    header = struct.pack(">I", len(data))
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _recv_init() -> Tuple[int, int]:
    """
    读取初始化行：K M
    - K: 0 表示先手玩家 P0，1 表示后手玩家 P1
    - M: 随机数种子（信息素初始化用）
    """
    line = _recv_line()
    if line is None:
        raise RuntimeError("No init line (K M) received from judger.")
    parts = line.strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"Invalid init line: {line!r}")
    seat = int(parts[0])
    seed = int(parts[1])
    if seat not in (0, 1):
        raise RuntimeError(f"Invalid seat K={seat}, expected 0 or 1.")
    _log(f"Init: seat={seat}, seed={seed}")
    return seat, seed


def _recv_state() -> Dict[str, Any] | None:
    """
    按照“局面信息”一节的格式读取一轮完整局面，解析为字典：
      {
        "round": R,
        "towers": [(id, player, x, y, type, cd), ...],
        "ants":   [(id, player, x, y, hp, lv, age, state), ...],
        "coins": [G0, G1],
        "camps_hp": [HP0, HP1],
      }
    如果遇到 EOF 则返回 None。
    """
    line = _recv_line()
    if line is None:
        return None
    s = line.strip()
    if not s:
        # 跳过空行，递归读取下一轮
        return _recv_state()
    try:
        R = int(s)
    except Exception:
        raise RuntimeError(f"Invalid round line: {line!r}")

    # 防御塔信息
    line = _recv_line()
    if line is None:
        return None
    try:
        N1 = int(line.strip())
    except Exception:
        raise RuntimeError(f"Invalid N1 line: {line!r}")
    towers: List[Tuple[int, int, int, int, int, int]] = []
    for _ in range(N1):
        l = _recv_line()
        if l is None:
            raise RuntimeError("EOF while reading tower lines")
        parts = l.strip().split()
        if len(parts) != 6:
            raise RuntimeError(f"Invalid tower line: {l!r}")
        tid, player, x, y, ttype, cd = map(int, parts)
        towers.append((tid, player, x, y, ttype, cd))

    # 蚂蚁信息
    line = _recv_line()
    if line is None:
        return None
    try:
        N2 = int(line.strip())
    except Exception:
        raise RuntimeError(f"Invalid N2 line: {line!r}")
    ants: List[Tuple[int, int, int, int, int, int, int, int]] = []
    for _ in range(N2):
        l = _recv_line()
        if l is None:
            raise RuntimeError("EOF while reading ant lines")
        parts = l.strip().split()
        if len(parts) != 8:
            raise RuntimeError(f"Invalid ant line: {l!r}")
        aid, player, x, y, hp, lv, age, state = map(int, parts)
        ants.append((aid, player, x, y, hp, lv, age, state))

    # 金币
    line = _recv_line()
    if line is None:
        return None
    parts = line.strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"Invalid coins line: {line!r}")
    G0, G1 = map(int, parts)

    # 基地血量
    line = _recv_line()
    if line is None:
        return None
    parts = line.strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"Invalid camps hp line: {line!r}")
    HP0, HP1 = map(int, parts)

    state = {
        "round": R,
        "towers": towers,
        "ants": ants,
        "coins": [G0, G1],
        "camps_hp": [HP0, HP1],
    }
    return state


def _ops_to_text(ops: List[List[int]]) -> str:
    """
    将 Operation 列表转为评测系统要求的文本：
      N\n
      T x y ...\n
      ...
    """
    n = len(ops)
    lines: List[str] = [str(n)]
    for op in ops:
        if not op:
            continue
        lines.append(" ".join(str(x) for x in op))
    return "\n".join(lines) + "\n"


def main() -> None:
    # 1) 初始化：读取 K M
    try:
        seat, seed = _recv_init()
    except Exception as e:
        _log(f"init failed: {e}")
        return
    # seed 暂时未在策略中使用，但保留以便后续扩展
    del seed

    # 2) 回合循环
    while True:
        state = _recv_state()
        if state is None:
            break
        try:
            ops = policy_from_site_state(state, seat)
        except Exception as e:
            _log(f"policy_from_site_state crashed: {e}")
            ops = []
        text = _ops_to_text(ops)
        _send_packet(text)


if __name__ == "__main__":
    main()

