from __future__ import annotations

from typing import Any, Dict, List
import random

TOWER_BUILD = 11
END_TURN = 0  

# 参见 ant_game - deploy/include/map.h 中基地方坐标
PLAYER_0_BASE_CAMP_X = 2
PLAYER_0_BASE_CAMP_Y = 10 - 1  # SIDE_LENGTH-1 = 9
PLAYER_1_BASE_CAMP_X = 2 * 10 - 1 - 3  # MAP_SIZE-3 = 19-3 = 16
PLAYER_1_BASE_CAMP_Y = 10 - 1


def _base_pos(player: int) -> tuple[int, int]:
    if player == 0:
        return PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y
    return PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y


def policy(state: Dict[str, Any], my_seat: int) -> List[List[int]]:
    """
    - 输入为judger发送的json
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
    """
    if state is None:
        return []

    coins = state.get("coins")
    try:
        my_coin = int(coins[my_seat])
    except Exception:
        return []

    towers = state.get("towers", [])
    own = 0
    for tower in towers:
        if tower[1] == my_seat:
            own += 1
    if my_coin < 2 ** (own + 1) * 15:
        return []
    bx, by = _base_pos(my_seat)
    ops: List[List[int]] = []
    if my_seat == 0:
        candidate_offsets = [(6, 1), (7, 1), (4, 2), (6, 2), (8, 2), (4, 3), (5, 3), (6, 4), 
                            (8, 4), (7, 5), (5, 6), (5, 7), (6, 7), (8, 7), (7, 8), (4, 9), 
                            (5, 9), (6, 9), (7, 10), (5, 11), (6, 11), (8, 11), (5, 12), (7, 13), 
                            (6, 14), (8, 14), (4, 15), (5, 15), (4, 16), (6, 16), (8, 16), (6, 17), (7, 17)]
    else:
        candidate_offsets = [(11, 1), (12, 1), (9, 2), (11, 2), (13, 2), (13, 3), (14, 3), (9, 4), 
                             (11, 4), (11, 5), (12, 6), (10, 7), (12, 7), (13, 7), (10, 8), (12, 9), 
                             (13, 9), (14, 9), (10, 10), (10, 11), (12, 11), (13, 11), (12, 12), (11, 13), 
                             (9, 14), (11, 14), (13, 15), (14, 15), (9, 16), (11, 16), (13, 16), (11, 17), (12, 17)]
    pos = random.choice(candidate_offsets)
    x, y = pos[0], pos[1]
    ops.append([TOWER_BUILD, x, y])
    return ops


