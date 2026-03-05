from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# 参见 ant_game - deploy/include/operation.h 中 Operation::Type
TOWER_BUILD = 11
END_TURN = 0  # 在 SDK-antgame/pettingzoo_env 中约定：type==0 表示结束本轮

# 参见 ant_game - deploy/include/map.h 中基地方坐标
PLAYER_0_BASE_CAMP_X = 2
PLAYER_0_BASE_CAMP_Y = 10 - 1  # SIDE_LENGTH-1 = 9
PLAYER_1_BASE_CAMP_X = 2 * 10 - 1 - 3  # MAP_SIZE-3 = 19-3 = 16
PLAYER_1_BASE_CAMP_Y = 10 - 1


def _base_pos(player: int) -> tuple[int, int]:
    if player == 0:
        return PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y
    return PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y


def policy(obs: Dict[str, Any], action_space) -> np.ndarray:
    """
    AntGame C++ 规则下的一个极简“贪心”规则 AI：

    - 只依赖 SDK-antgame/pettingzoo_env.py 暴露的 obs 字段：
        * coins: [coin0, coin1]
        * camps_hp: [hp0, hp1]
        * current_player: 0 or 1
    - 策略：
        * 若当前玩家金币充足（>=50）且基地周围有推断出来的空位，则尝试在
          基地附近建一座塔（Operation::TowerBuild），期望 C++ 按规则判断。
        * 否则直接结束回合（type=0）。

    该策略保证：
    - 所有动作均为 C++ Operation 允许的形式（type=0 或 11）。
    - 不依赖 Python 版 GameState，也不调用 logic/ 下的任何函数。
    """
    a = action_space.sample()
    # 默认：立即结束回合
    a[0] = END_TURN

    coins = obs.get("coins", [0, 0])
    if not isinstance(coins, (list, tuple)) or len(coins) != 2:
        return a

    my_seat = int(obs.get("current_player", 0))
    my_coin = int(coins[my_seat])
    # 简单约束：金币不足则绝不尝试建塔，避免频繁触发非法操作
    if my_coin < 50:
        return a

    bx, by = _base_pos(my_seat)

    # 在基地周围一圈尝试放一座塔；坐标是否真的可用由 C++ 自行裁决
    # 这里不去解析 raw_state_bytes，避免与具体 JSON 结构过度耦合。
    candidate_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in candidate_offsets:
        x = bx + dx
        y = by + dy
        if x < 0 or y < 0 or x >= 20 or y >= 20:
            continue
        a[0] = TOWER_BUILD  # Operation::TowerBuild
        a[1] = x            # x 坐标
        a[2] = y            # y 坐标
        a[3] = 0            # id（建塔时由 C++ 自行分配）
        a[4] = 0            # args 无特殊含义
        break

    return a


def policy_from_site_state(state: Dict[str, Any], my_seat: int) -> List[List[int]]:
    """
    针对“网站评测协议”的规则 AI：
    - 输入为根据规则文档解析出的 Python dict：
        {
          "round": R,
          "towers": [...],
          "ants": [...],
          "coins": [G0, G1],
          "camps_hp": [HP0, HP1],
        }
      其中 coins/camps_hp 的含义与文档完全一致。
    - 输出为 Operation 列表，每条形如 [T, ...]，将由 wrapper 转换成
      文档中规定的 N + N 行操作文本。

    当前策略（极简、防御向）：
    - 若金币不足 50，则本轮不执行任何操作（N=0）。
    - 否则在本方基地周围四个相邻格尝试建一座塔（11 x y），只返回一条操作。
    """
    coins = state.get("coins", [0, 0])
    if not isinstance(coins, (list, tuple)) or len(coins) != 2:
        return []
    try:
        my_coin = int(coins[my_seat])
    except Exception:
        return []

    if my_coin < 50:
        # 金币不足，直接不出手
        return []

    bx, by = _base_pos(my_seat)
    ops: List[List[int]] = []
    candidate_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in candidate_offsets:
        x = bx + dx
        y = by + dy
        if 0 <= x < 19 and 0 <= y < 19:
            # 11 x y：在 (x,y) 建造防御塔（合法性由 C++ 判定）
            ops.append([TOWER_BUILD, x, y])
            break
    return ops


