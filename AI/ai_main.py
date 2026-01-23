from __future__ import annotations

import json
import os
import struct
import sys
from typing import Any, List, Optional

from logic.constant import row, col
from logic.gamedata import CellType, Farmer, MainGenerals, SubGenerals
from logic.gamestate import GameState
from ai import ai_func


def _log(msg: str) -> None:
    sys.stderr.write(f"[AI] {msg}\n")
    sys.stderr.flush()


# =========================
# AI -> judger (MUST be 4B len + payload)
# =========================
def _send_to_judger(payload: bytes) -> None:
    header = struct.pack(">I", len(payload))
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(payload)
    sys.stdout.flush()


def _send_ops_text(text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    _send_to_judger(text.encode("utf-8"))


# =========================
# judger -> AI (NO encapsulation)
# We read line-oriented messages. Logic guarantees JSON is one-line + '\n'.
# Also support the 'K M' line.
# =========================
def _recv_line() -> Optional[str]:
    b = sys.stdin.buffer.readline()
    if not b:
        return None
    try:
        return b.decode("utf-8", errors="strict")
    except Exception:
        return b.decode("utf-8", errors="replace")


class AIProcess:
    def __init__(self, seat: int = 0):
        self.seat = int(seat)
        self.state = GameState()
        self.initialized = False
        self.ai = ai_func
        self.seed: Optional[int] = None
        self.last_turn: Optional[int] = None

    # ---- state apply (same as your original) ----
    def _apply_init(self, rep: dict) -> None:
        ct = rep.get("Cell_type", "")
        if ct:
            for i in range(row * col):
                t = int(ct[i])
                self.state.board[i // col][i % col].type = CellType(t)
        for cell in rep.get("Cells", []):
            (x, y), owner, army = cell
            c = self.state.board[x][y]
            c.player = owner
            c.army = army
        self._rebuild_generals(rep.get("Generals", []))
        if "Coins" in rep:
            self.state.coin = list(rep["Coins"])  # type: ignore
        if "Tech_level" in rep:
            self.state.tech_level = [list(rep["Tech_level"][0]), list(rep["Tech_level"][1])]  # type: ignore
        if "Weapon_cds" in rep:
            self.state.super_weapon_cd = list(rep["Weapon_cds"])  # type: ignore
        self.state.round = int(rep.get("Round", 1))
        self.initialized = True

    def _apply_update(self, rep: dict) -> None:
        for cell in rep.get("Cells", []):
            (x, y), owner, army = cell
            c = self.state.board[x][y]
            c.player = owner
            c.army = army
        self._rebuild_generals(rep.get("Generals", []))
        if "Coins" in rep:
            self.state.coin = list(rep["Coins"])  # type: ignore
        if "Tech_level" in rep:
            self.state.tech_level = [list(rep["Tech_level"][0]), list(rep["Tech_level"][1])]  # type: ignore
        if "Weapon_cds" in rep:
            self.state.super_weapon_cd = list(rep["Weapon_cds"])  # type: ignore
        if "Round" in rep:
            self.state.round = int(rep["Round"])

    def _rebuild_generals(self, gens: List[dict]) -> None:
        for i in range(row):
            for j in range(col):
                self.state.board[i][j].generals = None
        self.state.generals.clear()
        for g in gens:
            gid = int(g.get("Id", 0))
            owner = int(g.get("Player", -1))
            gtype = int(g.get("Type", 3))
            x, y = list(g.get("Position", [0, 0]))
            lvl = list(g.get("Level", [1, 1, 1]))
            scd = list(g.get("Skill_cd", [0, 0, 0, 0, 0]))
            srest = list(g.get("Skill_rest", [0, 0, 0]))
            alive = int(g.get("Alive", 1))
            if alive == 0:
                continue
            if gtype == 1:
                obj = MainGenerals(id=gid, player=owner)
                obj.produce_level = max(1, 2 * (int(lvl[0]) - 1))
                obj.defense_level = int(lvl[1])
                obj.mobility_level = max(1, int(lvl[2]))
            elif gtype == 2:
                obj = SubGenerals(id=gid, player=owner)
                obj.produce_level = max(1, 2 * (int(lvl[0]) - 1))
                obj.defense_level = int(lvl[1])
                obj.mobility_level = max(1, int(lvl[2]))
            else:
                obj = Farmer(id=gid, player=owner)
                obj.produce_level = int(lvl[0])
                if int(lvl[1]) == 2:
                    obj.defense_level = 1.5  # type: ignore[assignment]
                elif int(lvl[1]) >= 3:
                    obj.defense_level = int(lvl[1]) - 1
                else:
                    obj.defense_level = 1
                obj.mobility_level = 0
            obj.position = [x, y]
            obj.skills_cd = scd
            obj.skill_duration = srest
            self.state.generals.append(obj)
            self.state.board[x][y].generals = obj
            if owner != -1:
                self.state.board[x][y].player = owner

    @staticmethod
    def _ops_to_str(ops: List[List[int]]) -> str:
        lines: List[str] = []
        for op in ops:
            if not op:
                continue
            lines.append(" ".join(str(x) for x in op))
            if op[0] == 8:
                break
        if not lines or lines[-1].split()[0] != "8":
            lines.append("8")
        return "\n".join(lines) + "\n"

    def _handle_km(self, line: str) -> bool:
        s = line.strip()
        parts = s.split()
        if len(parts) != 2:
            return False
        try:
            k = int(parts[0])
            m = int(parts[1])
        except Exception:
            return False
        # accept K in {0,1}
        if k in (0, 1):
            self.seat = k
            self.seed = m
            _log(f"KM init: seat={self.seat}, seed={self.seed}")
            return True
        return False

    def loop(self) -> None:
        while True:
            line = _recv_line()
            if line is None:
                break

            # 1) KM init line
            if self._handle_km(line):
                continue

            # 2) State rep JSON (one-line)
            line_strip = line.strip()
            if not line_strip:
                continue
            if not (line_strip.startswith("{") and line_strip.endswith("}")):
                # ignore non-json messages (e.g., opponent ops)
                continue

            try:
                rep = json.loads(line_strip)
            except Exception:
                continue
            if not isinstance(rep, dict):
                continue

            # apply init/update
            try:
                if not self.initialized:
                    self._apply_init(rep)
                else:
                    self._apply_update(rep)
            except Exception as e:
                _log(f"apply rep failed: {e}")
                continue

            # Turn gating: only act when Turn == seat
            turn = rep.get("Turn", None)
            try:
                turn_i = int(turn) if turn is not None else None
            except Exception:
                turn_i = None

            if turn_i is not None:
                self.last_turn = turn_i
                if turn_i != self.seat:
                    continue

            # If Turn not provided, assume "message implies need action"
            try:
                ops = self.ai(self.state)
            except Exception as e:
                _log(f"ai_func crashed: {e}")
                ops = [[8]]

            out_text = self._ops_to_str(ops)
            _send_ops_text(out_text)


def main() -> None:
    # env fallback; official seat should come from KM
    try:
        seat = int(os.environ.get("AI_SEAT", "0"))
    except Exception:
        seat = 0
    AIProcess(seat=seat).loop()


if __name__ == "__main__":
    main()
