from logic.constant import *
from logic.gamedata import MainGenerals, SuperWeapon, WeaponType
from logic.generate_round_replay import get_single_round_replay


def handle_bomb(gamestate, location: list[int, int], replay) -> bool:
    combinations = set()
    x, y = location[0], location[1]
    # clamp within board bounds using row for x and col for y
    x1 = x - 1 if x - 1 >= 0 else 0
    x2 = x
    x3 = x + 1 if x + 1 <= row - 1 else row - 1
    y1 = y - 1 if y - 1 >= 0 else 0
    y2 = y
    y3 = y + 1 if y + 1 <= col - 1 else col - 1
    combinations.add((x1, y1))
    combinations.add((x1, y2))
    combinations.add((x1, y3))
    combinations.add((x2, y1))
    combinations.add((x2, y2))
    combinations.add((x2, y3))
    combinations.add((x3, y1))
    combinations.add((x3, y2))
    combinations.add((x3, y3))
    for combination in combinations:
        x, y = combination
        handle_bomb_cell(gamestate, x, y, replay)
    return True


def handle_bomb_cell(gamestate, x: int, y: int, replay) -> bool:
    cell = gamestate.board[x][y]
    if isinstance(cell.generals, MainGenerals):
        cell.army = cell.army // 2
        replay["Cells"].append([[x, y], gamestate.board[x][y].player, cell.army])
    else:
        cell.army = 0
        cell.player = -1
        cell.generals = None
        replay["Cells"].append([[x, y], -1, cell.army])
        for replay_general in replay["Generals"]:
            if replay_general["Position"] == [x, y]:
                replay_general["Alive"] = 0
                break
        for general in gamestate.generals:
            if general.position == [x, y]:
                gamestate.generals.remove(general)
    gamestate.board[x][y] = cell

    return True


def bomb(gamestate, location: list[int, int], player: int) -> bool:
    # 先检查参数
    if player != 0 and player != 1:
        return False
    if location[0] < 0 or location[0] >= row:
        return False
    if location[1] < 0 or location[1] >= col:
        return False
    is_super_weapon_unlocked = gamestate.super_weapon_unlocked[player]
    cd = gamestate.super_weapon_cd[player]
    if is_super_weapon_unlocked == True and cd == 0:
        gamestate.active_super_weapon.append(
            SuperWeapon(WeaponType(0), player, 0, 5, location)
        )
        gamestate.super_weapon_cd[player] = use_cd

        replay = get_single_round_replay(
            gamestate, [], player, [6, 1, location[0], location[1]]
        )

        handle_bomb(gamestate, location, replay)

        with open(gamestate.replay_file, "a") as f:
            f.write(str(replay).replace("'", '"') + "\n")
        f.close()
        return True
    else:
        return False


def strengthen(gamestate, location: list[int, int], player: int) -> bool:
    # 先检查参数
    if player != 0 and player != 1:
        return False
    if location[0] < 0 or location[0] >= row:
        return False
    if location[1] < 0 or location[1] >= col:
        return False
    is_super_weapon_unlocked = gamestate.super_weapon_unlocked[player]
    cd = gamestate.super_weapon_cd[player]
    if is_super_weapon_unlocked == True and cd == 0:
        weapon = SuperWeapon(WeaponType(1), player, 5, 5, location)
        gamestate.active_super_weapon.append(weapon)
        gamestate.super_weapon_cd[player] = use_cd

        replay = get_single_round_replay(
            gamestate, [], player, [6, 2, location[0], location[1]]
        )
        with open(gamestate.replay_file, "a") as f:
            f.write(str(replay).replace("'", '"') + "\n")
        f.close()

        return True
    else:
        return False


def tp(gamestate, start: list[int, int], to: list[int, int], player: int) -> bool:
    # 先检查参数范围
    if player != 0 and player != 1:
        return False
    if start[0] < 0 or start[0] >= row:
        return False
    if start[1] < 0 or start[1] >= col:
        return False
    if to[0] < 0 or to[0] >= row:
        return False
    if to[1] < 0 or to[1] >= col:
        return False
    # 还需判断目标是否合法
    is_super_weapon_unlocked = gamestate.super_weapon_unlocked[player]
    cd = gamestate.super_weapon_cd[player]
    if is_super_weapon_unlocked == True and cd == 0:
        x_st, y_st = start[0], start[1]
        x_to, y_to = to[0], to[1]
        cell_st, cell_to = gamestate.board[x_st][y_st], gamestate.board[x_to][y_to]
        if cell_st.player != player:
            return False
        if cell_to.generals != None:
            return False
        num = 0
        if cell_st.army == 0 or cell_st.army == 1:
            return False
        else:
            num = cell_st.army - 1
            cell_st.army = 1
        cell_to.army = num
        cell_to.player = player
        gamestate.board[x_st][y_st] = cell_st
        gamestate.board[x_to][y_to] = cell_to
        gamestate.super_weapon_cd[player] = use_cd
        weapon = SuperWeapon(WeaponType(2), player, 2, 2, to)
        gamestate.active_super_weapon.append(weapon)
        replay = get_single_round_replay(
            gamestate,
            [[x_st, y_st], [x_to, y_to]],
            player,
            [6, 3, x_to, y_to, x_st, y_st],
        )
        with open(gamestate.replay_file, "a") as f:
            f.write(str(replay).replace("'", '"') + "\n")
        f.close()

        return True
    return False


def timestop(gamestate, location: list[int, int], player: int) -> bool:
    # 首先检查参数范围
    if player != 0 and player != 1:
        return False
    x, y = location[0], location[1]
    if x < 0 or x >= row or y < 0 or y >= col:
        return False
    # 检查参数合理性
    is_super_weapon_unlocked = gamestate.super_weapon_unlocked[player]
    cd = gamestate.super_weapon_cd[player]
    if is_super_weapon_unlocked == True and cd == 0:
        weapon = SuperWeapon(
            type=WeaponType(3), player=player, cd=10, rest=10, position=location
        )
        gamestate.active_super_weapon.append(weapon)
        gamestate.super_weapon_cd[player] = use_cd

        replay = get_single_round_replay(gamestate, [], player, [6, 4, x, y])
        with open(gamestate.replay_file, "a") as f:
            f.write(str(replay).replace("'", '"') + "\n")
        f.close()
        return True
    else:
        return False


"""
location类型为list
这部分需要完成四个超级武器的逻辑,接受gamestate参数
首先需要根据player,科技是否解锁,金钱数判断操作是否合法,不合法返回false
若为合法操作,则需要在gamestate中更新对应的状态
对于bomb,你需要更新location周围的兵力和将军(消灭他们)
对于强化,你需要只需要更新gamestate即可
对于tp,你需要判断操作是否合法,并将军队更新
对于时间停止,只需要在gamestate中更新即可
! ! !建议参考demo的写法
"""
