"""
说明：
- Python 端扮演评测平台/裁判角色，直接通过 comm_judger.h 所定义的
  JSON+长度前缀 协议与 C++ `Game` 类通信。
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pettingzoo import AECEnv
from gymnasium import spaces



@dataclass
class EnvConfig:
    max_rounds: int = 512
    max_ops_per_turn: int = 16


class CppAntGameProcess:
    """
    封装与 C++ ant_game 可执行文件通信
    其方式与saiblo上的通信方式类似
    """

    def __init__(self, path: str):
        self.path = path
        self.proc: Optional[subprocess.Popen] = None

    # ---- 基础 IO ----
    def _ensure_started(self):
        # restart the subprocess if it has died or hasn't been started yet.
        if self.proc is not None:
            if self.proc.poll() is None:
                return
            # previous process terminated unexpectedly; clear for restart
            self.proc = None
        cwd = str(Path(__file__).resolve().parent.parent)
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=cwd,
        )

    def _write_json(self, obj: dict):
        assert self.proc is not None and self.proc.stdin is not None
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.proc.stdin.write(struct.pack(">I", len(data)))
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _read_json(self) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        # if the process already died, fail fast
        if self.proc.poll() is not None:
            raise RuntimeError(f"C++ ant_game process terminated with code {self.proc.returncode}")

        hdr = self.proc.stdout.read(4)
        if not hdr:
            # the process closed stdout unexpectedly
            raise RuntimeError("C++ ant_game process closed stdout.")
        (length,) = struct.unpack(">I", hdr)
        # consume the extra object tag (ignored by Python)
        _ = self.proc.stdout.read(4)

        # read the expected payload; if the process dies mid‑write we'll get a
        # shorter result and can include diagnostics.
        body = self.proc.stdout.read(length)
        if len(body) != length:
            # gather additional info for debugging
            remainder = b""
            try:
                # attempt to read whatever else might still be available
                remainder = self.proc.stdout.read()
            except Exception:
                pass
            stderr = b""
            if self.proc.stderr is not None:
                stderr = self.proc.stderr.read() or b""
            raise RuntimeError(
                f"Failed to read full JSON body from C++ ant_game. "
                f"expected={length} got={len(body)} leftover={len(remainder)} "
                f"returncode={self.proc.poll()} stderr={stderr!r}"
            )
        return json.loads(body.decode("utf-8"))

    # ---- 协议：初始化与回合 ----
    def send_init(self, seed: int | None, replay_path: str) -> dict:
        """
        发送 from_judger_init 对应的 JSON，读取 C++ 返回的初始状态 JSON。
        player_list 含义参见 Game::init：
        - 1: AI 玩家（有运行时/输出限制）
        - 2: HUMAN_PLAYER（无 AI 相关限制）
        我们在训练环境中不需要 saiblo 那些限制，因此统一标为 2。
        """
        print(f"Ensuring C++ ant_game process is started...")
        self._ensure_started()
        print(f"Sending initialization message with seed={seed}...")
        init_msg = {
            "player_list": [2, 2],  # 视作 human，便于自由发送 Operation JSON
            "player_num": 2,
            "config": {"random_seed": int(seed or 0)},
            "replay": replay_path,
        }
        print(f"Initialization message: {init_msg}")
        self._write_json(init_msg)
        print(f"Waiting for initial state from C++...")
        # C++ 在 Game::init 中通常会通过 Output/Output_to_judger 输出首帧信息。
        # 然而在第一回合该消息有时不会包含 round_state 数据；
        # 因此我们只读取一次并立即返回。后续的 round_state 会在
        # send_round 中到来，环境无需在 reset 时等待它。
        state = self._read_json()
        print(f"Received initial state from C++: {state}")
        return state

    def send_round(self, p0_ops: List[dict], p1_ops: List[dict]) -> dict:
        """
        发送一轮双方的操作列表：
        - 先发 player=0 的 from_judger_round
        - 再发 player=1 的 from_judger_round
        """
        self._ensure_started()

        def _round_msg(player: int, ops: List[dict]) -> dict:
            # content 是字符串；内部再是一整个 Operation 数组的 JSON
            ops_str = json.dumps(ops, ensure_ascii=False)
            return {
                "player": int(player),
                "content": ops_str,
                "time": 0,
            }

        self._write_json(_round_msg(0, p0_ops))
        self._write_json(_round_msg(1, p1_ops))
        # Game::next_round + dump_round_state 之后会输出当前回合的 JSON
        state = self._read_json()
        return state

    def close(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None


class AntGameAECEnv(AECEnv):
    """
    C++ AntGame 的 PettingZoo AEC 封装。
    """

    metadata = {
        "render_modes": ["ansi"],
        "name": "antgame_cpp_pettingzoo_v0",
    }

    def __init__(self, render_mode: str | None = None, config: EnvConfig | None = None):
        self.render_mode = render_mode
        self.config = config or EnvConfig()

        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents.copy()

        # 进程与内部状态
        root = Path(__file__).resolve().parent.parent
        exe_name = "main.exe" if os.name == "nt" else "main"
        path = root / "game" / "output" / exe_name
        # on Windows the binary may be named "main" without extension; try a
        # fallback if the expected file is missing.
        if not path.exists():
            alt = root / "game" / "output" / ("main" if exe_name.endswith(".exe") else "main.exe")
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(f"AntGame executable not found at {path} or {alt}")
        self._cpp = CppAntGameProcess(str(path))

        self._last_state: Optional[dict] = None  # 最新一轮的完整 JSON（含 round_state）
        self._round_idx: int = 0
        self._pending_ops: Dict[int, List[dict]] = {0: [], 1: []}
        self._ended_this_round: Dict[int, bool] = {0: False, 1: False}

        # AEC 必需字段
        self.rewards: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self.terminations: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.truncations: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.infos: Dict[str, Dict[str, Any]] = {a: {} for a in self.possible_agents}
        self._cumulative_rewards: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self._current: int = 0
        self.agent_selection: str = self.possible_agents[self._current]

        self._build_spaces()

    # ------------------------------------------------------------------
    # PettingZoo Core API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        options = options or {}
        del options

        self.agents = self.possible_agents.copy()
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self._current = 0
        self.agent_selection = self.possible_agents[self._current]
        self._pending_ops = {0: [], 1: []}
        self._ended_this_round = {0: False, 1: False}
        self._round_idx = 0
        print(f"Resetting environment with seed={seed}")
        # 发送初始化消息给 C++，读取首帧状态
        replay_dir = Path("replays")
        replay_dir.mkdir(exist_ok=True)
        replay_path = str(replay_dir / "antgame_cpp_pettingzoo.json")
        self._last_state = self._cpp.send_init(seed, replay_path)
        print("Initial state from C++:", self._last_state)

    def observe(self, agent: str):
        player = 0 if agent == "player_0" else 1
        return self._build_obs(player)

    def step(self, action: List[int] | np.ndarray):
        if any(self.terminations.values()) or any(self.truncations.values()):
            return

        if isinstance(action, np.ndarray):
            action = action.tolist()
        if not isinstance(action, list) or len(action) < 1:
            raise ValueError("action must be a non-empty list or np.ndarray")

        player = self._current
        agent = self.possible_agents[player]
        if agent != self.agent_selection:
            # 顺序错误的 step 忽略
            return

        atype = int(action[0])
        # 清空该 agent 本步 reward
        self.rewards[agent] = 0.0

        if atype == 0:
            # 结束本玩家本轮
            self._ended_this_round[player] = True
        else:
            op = self._decode_to_operation(action)
            self._pending_ops[player].append(op)

        # 若双方都结束此轮，则把整轮 ops 发送给 C++ 并更新状态
        if self._ended_this_round[0] and self._ended_this_round[1]:
            self._flush_round_and_update_state()
            self._ended_this_round = {0: False, 1: False}
            self._pending_ops = {0: [], 1: []}
            self._round_idx += 1

        # 交替到另一名玩家
        self._current = 1 - player
        self.agent_selection = self.possible_agents[self._current]

    def render(self):
        if self.render_mode != "ansi":
            return None
        if self._last_state is None:
            return "<uninitialized>"
        coins = self._extract_coins(self._last_state)
        camps_hp = self._extract_camps_hp(self._last_state)
        return f"Round={self._round_idx} Coins={coins.tolist()} CampsHP={camps_hp.tolist()}"

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _build_spaces(self):
        # 动作空间：[type, x, y, id, args]
        self._action_space = spaces.MultiDiscrete(
            np.array(
                [
                    64,   # type（0=EndTurn，其它映射到 Operation::Type）
                    19,   # x in [0, MAP_SIZE-1]
                    19,   # y
                    256,  # id
                    64,   # args
                ],
                dtype=np.int64,
            )
        )

        # 观测空间：对回放 round_state 做一个轻量、通用的投影
        self._observation_space = spaces.Dict(
            {
                "coins": spaces.Box(low=0, high=10_000, shape=(2,), dtype=np.int32),
                "camps_hp": spaces.Box(low=0, high=1_000, shape=(2,), dtype=np.int32),
                "current_player": spaces.Discrete(2),
                "round": spaces.Discrete(self.config.max_rounds + 1),
                # 原始 JSON 压缩成字节向量，便于需要时自行解析
                "raw_state_bytes": spaces.Box(
                    low=0, high=255, shape=(4096,), dtype=np.uint8
                ),
            }
        )

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self._observation_space

    def _build_obs(self, player: int) -> Dict[str, Any]:
        state = self._last_state or {}
        coins = self._extract_coins(state)
        camps_hp = self._extract_camps_hp(state)

        raw_bytes = json.dumps(state, ensure_ascii=False).encode("utf-8")
        buf = np.zeros((4096,), dtype=np.uint8)
        n = min(len(raw_bytes), 4096)
        buf[:n] = np.frombuffer(raw_bytes[:n], dtype=np.uint8)

        obs = {
            "coins": coins,
            "camps_hp": camps_hp,
            "current_player": int(player),
            "round": int(self._round_idx),
            "raw_state_bytes": buf,
        }
        return obs

    def _extract_coins(self, state: dict) -> np.ndarray:
        """
        根据官方回放格式：
        - 外层为 {seed?, op0, op1, round_state}
        - coins 位于 round_state["coins"]，为 [c0, c1]
        """
        rs = state.get("round_state", state)
        coins = rs.get("coins")
        if isinstance(coins, list) and len(coins) == 2:
            try:
                return np.array([int(coins[0]), int(coins[1])], dtype=np.int32)
            except Exception:
                pass
        return np.zeros((2,), dtype=np.int32)

    def _extract_camps_hp(self, state: dict) -> np.ndarray:
        """
        根据官方回放格式：
        - round_state["camps"] 为 [hp0, hp1]
        """
        rs = state.get("round_state", state)
        camps = rs.get("camps")
        if isinstance(camps, list) and len(camps) == 2:
            try:
                hp0 = int(camps[0])
                hp1 = int(camps[1])
                return np.array([hp0, hp1], dtype=np.int32)
            except Exception:
                pass
        return np.array([50, 50], dtype=np.int32)

    def _decode_to_operation(self, action: List[int]) -> dict:
        """
        将 MultiDiscrete 动作解码为单条 Operation：
          { "type": int, "id": int, "args": int, "pos": {"x": int, "y": int} }
        对应 ant_game - deploy/include/operation.h 中 Operation 的 JSON 结构。
        """
        op_type = int(action[0])
        x = int(action[1])
        y = int(action[2])
        op_id = int(action[3])
        args = int(action[4])
        return {
            "type": op_type,
            "id": op_id,
            "args": args,
            "pos": {"x": x, "y": y},
        }

    def _flush_round_and_update_state(self) -> None:
        """
        将当前轮双方累积的 Operation 列表发送给 C++，并依据 round_state["winner"]
        更新胜负与奖励。
        """
        try:
            new_state = self._cpp.send_round(self._pending_ops[0], self._pending_ops[1])
        except Exception as exc:
            self._last_state = None
            self.terminations = {a: True for a in self.agents}
            self.rewards = {a: 0.0 for a in self.agents}
            self.infos["player_0"]["error"] = str(exc)
            self.infos["player_1"]["error"] = str(exc)
            return

        self._last_state = new_state

        rs = new_state.get("round_state", new_state)
        winner = rs.get("winner", -1)
        if winner in (0, 1):
            self.terminations = {a: True for a in self.agents}
            self.rewards = {a: 0.0 for a in self.agents}
            win_agent = self.agents[int(winner)]
            lose_agent = self.agents[1 - int(winner)]
            self.rewards[win_agent] = 1.0
            self.rewards[lose_agent] = -1.0


def env(render_mode: str | None = None, config: EnvConfig | None = None) -> AntGameAECEnv:
    return AntGameAECEnv(render_mode=render_mode, config=config)


def parallel_env(*args, **kwargs):
    try:
        from pettingzoo.utils.conversions import aec_to_parallel
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pettingzoo not installed for parallel conversion") from e
    return aec_to_parallel(AntGameAECEnv(*args, **kwargs))

