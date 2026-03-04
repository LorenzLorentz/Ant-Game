from logic.gamedata import *
import logic.constant as constant
from logic.constant import row, col
from logic.generate_round_replay import get_single_round_replay

# use gamestate.map.is_valid later


def call_generals(state, player: int, position: list[int, int]) -> bool:
    # reject coordinates outside the playable hex or even array bounds
    if position[0] < 0 or position[1] < 0 or position[0] >= row or position[1] >= col:
        return False
    if not getattr(state, "map", None) or not state.map.is_valid(position[0], position[1]):
        return False
    # calculate tower build cost: 15 × 2^i (i = existing towers owned).
    # Only built sub-generals count as towers; main generals must not inflate
    # the first tower's price.
    towers = [
        g for g in state.generals if g.player == player and isinstance(g, SubGenerals)
    ]
    i = len(towers)
    build_cost = 15 * (2 ** i)
    if state.coin[player] < build_cost:
        return False
    # deduct later after verifying other conditions
    elif state.board[position[0]][position[1]].player != player:
        return False
    elif state.board[position[0]][position[1]].generals != None:
        return False
    for sw in state.active_super_weapon:  # 超级武器效果
        if (
            sw.position == position
            and sw.rest
            and sw.type == WeaponType.TRANSMISSION
            and sw.player == player
        ):  # 超时空传送眩晕
            return False
        if (
            abs(sw.position[0] - position[0]) <= 1
            and abs(sw.position[1] - position[1]) <= 1
            and sw.rest
            and sw.type == WeaponType.TIME_STOP
            and sw.player != player
        ):  # 时间暂停效果
            return False
    gen = SubGenerals(state.next_generals_id, player, position)
    state.board[position[0]][position[1]].generals = gen
    state.generals.append(gen)
    state.next_generals_id += 1
    # pay the computed cost
    state.coin[player] -= build_cost
    replay = get_single_round_replay(state, [], player, [7, position[0], position[1]])
    with open(state.replay_file, "a") as f:
        f.write(str(replay).replace("'", '"') + "\n")
    f.close()
    return True


def downgrade_tower(state, player: int, tower_id: int) -> bool:
    """Handle tower downgrade or removal (command 13).

    - If the specified general is not a SubGenerals belonging to *player*,
      the command fails.
    - Grade is derived from :attr:`produce_level` with the mapping
      1→level1 (basic), 2→level2, 4→level3.  Upgrade paths mirror
      :func:`production_up`.
    - Downgrading level3->2 refunds 80% of the level3 upgrade cost (200→160).
      Downgrading level2->1 refunds 80% of level2 upgrade cost (60→48).
    - Removing a level1 tower gives 80% of the build cost that the tower
      would have paid **after** removal (i.e. 15×2^{new_count}).
    """
    # find the tower
    target = None
    for g in state.generals:
        if int(g.id) == int(tower_id):
            target = g
            break
    if target is None or target.player != player or not isinstance(target, SubGenerals):
        return False

    # determine refund
    refund = 0
    lvl = getattr(target, "produce_level", 1)
    if lvl == 4:
        # grade3 -> grade2
        refund = int(constant.lieutenant_production_T2 * 0.8)
        target.produce_level = 2
    elif lvl == 2:
        # grade2 -> grade1
        refund = int(constant.lieutenant_production_T1 * 0.8)
        target.produce_level = 1
    else:
        # remove basic tower
        # compute cost after removal (number of remaining towers)
        remaining = [g for g in state.generals if g.player == player and isinstance(g, SubGenerals) and g.id != tower_id]
        build_cost_after = 15 * (2 ** len(remaining))
        refund = int(build_cost_after * 0.8)
        # actually remove from board and list
        x, y = target.position
        state.board[x][y].generals = None
        state.generals.remove(target)
    state.coin[player] += refund

    replay = get_single_round_replay(state, [], player, [13, tower_id])
    with open(state.replay_file, "a") as f:
        f.write(str(replay).replace("'", '"') + "\n")
    f.close()
    return True
