import math

from logic.constant import *
from logic.gamedata import SkillType
from logic.generate_round_replay import get_single_round_replay
from logic.movement import *

# 本文件定义了将军战法


# 用于处理军队突袭
def army_rush(
    location: list[int, int],
    gamestate,
    player: int,
    destination: list[int, int],
) -> bool:
    x, y = location[0], location[1]
    new_x, new_y = destination[0], destination[1]
    num = gamestate.board[x][y].army - 1
    if gamestate.board[new_x][new_y].player == -1:
        gamestate.board[new_x][new_y].army += num
        gamestate.board[x][y].army -= num
        gamestate.board[new_x][new_y].player = player
    elif gamestate.board[new_x][new_y].player == player:
        gamestate.board[x][y].army -= num
        gamestate.board[new_x][new_y].army += num
    elif gamestate.board[new_x][new_y].player == 1 - player:
        attack = compute_attack(gamestate.board[x][y], gamestate)
        defence = compute_defence(gamestate.board[new_x][new_y], gamestate)
        vs = num * attack - gamestate.board[new_x][new_y].army * defence
        assert vs > 0
        gamestate.board[new_x][new_y].player = player
        # Use proper division before ceil to compute remaining army
        gamestate.board[new_x][new_y].army = math.ceil(vs / attack)
        gamestate.board[x][y].army -= num

    return True


def check_rush_param(
    player: int,
    destination: list[int, int],
    location: list[int, int],
    gamestate,
) -> bool:
    x, y = location[0], location[1]
    x_new, y_new = destination[0], destination[1]
    # 检查参数合理性
    if gamestate.board[x][y].generals == None:
        return False
    if gamestate.board[x_new][y_new].generals != None:
        return False
    if gamestate.board[x][y].army < 2:
        return False
    if gamestate.board[x_new][y_new].player == 1 - player:
        num = gamestate.board[x][y].army - 1
        attack = compute_attack(gamestate.board[x][y], gamestate)
        defence = compute_defence(gamestate.board[x_new][y_new], gamestate)
        vs = num * attack - gamestate.board[x_new][y_new].army * defence
        if vs <= 0:
            return False
    return True


def handle_breakthrough(destination: list[int, int], gamestate) -> bool:
    x, y = destination[0], destination[1]
    if gamestate.board[x][y].army > 20:
        gamestate.board[x][y].army -= 20
    else:
        gamestate.board[x][y].army = 0
        if gamestate.board[x][y].generals == None:
            gamestate.board[x][y].player = -1
    return True


def skill_activate(
    player: int,
    location: list[int, int],
    destination: list[int, int],
    gamestate,
    skillType: SkillType,
) -> bool:
    # 首先检查参数范围
    if player != 0 and player != 1:
        return False
    x, y = location[0], location[1]
    if x < 0 or x >= row or y < 0 or y >= col:
        return False
    if destination == [-1, -1]:
        destination = None
    if destination != None:
        x_new, y_new = destination[0], destination[1]
        if x_new < 0 or x_new >= row or y_new < 0 or y_new >= col:
            return False
        d1 = abs(x_new - x)
        d2 = abs(y_new - y)
        if d1 > 2 or d2 > 2:
            return False
    # 检查参数合理性
    if gamestate.board[x][y].player != player:
        return False
    coin = gamestate.coin[player]
    general = gamestate.board[location[0]][location[1]].generals
    if general == None:
        return False
    for sw in gamestate.active_super_weapon:  # 超级武器效果
        if (
            sw.position == [x, y]
            and sw.rest
            and sw.type == WeaponType.TRANSMISSION
            and sw.player == player
        ):  # 超时空传送眩晕
            return False
        if (
            abs(sw.position[0] - x) <= 1
            and abs(sw.position[1] - y) <= 1
            and sw.rest
            and sw.type == WeaponType.TIME_STOP
            and sw.player != player
        ):  # 时间暂停效果
            return False
    if skillType == SkillType.SURPRISE_ATTACK:
        # 检查参数是否合法
        if not check_rush_param(player, destination, location, gamestate):
            return False

        if coin >= tactical_strike and general.skills_cd[0] == 0:
            army_rush(location, gamestate, player, destination)

            gamestate.board[location[0]][location[1]].generals = None
            gamestate.board[destination[0]][destination[1]].generals = general
            general.position = [destination[0], destination[1]]
            general.skills_cd[0] = 5

            gamestate.coin[player] -= tactical_strike

            replay = get_single_round_replay(
                gamestate,
                [[location[0], location[1]], [destination[0], destination[1]]],
                player,
                [4, general.id, 1, destination[0], destination[1]],
            )
            with open(gamestate.replay_file, "a") as f:
                f.write(str(replay).replace("'", '"') + "\n")
            f.close()
            return True
        else:
            return False
    elif skillType == SkillType.ROUT:
        if coin >= breakthrough and general.skills_cd[1] == 0:
            handle_breakthrough(destination, gamestate)
            general.skills_cd[1] = 10
            gamestate.board[location[0]][location[1]].generals = general
            gamestate.coin[player] -= breakthrough
            replay = get_single_round_replay(
                gamestate,
                [[destination[0], destination[1]]],
                player,
                [4, general.id, 2, destination[0], destination[1]],
            )
            with open(gamestate.replay_file, "a") as f:
                f.write(str(replay).replace("'", '"') + "\n")
            f.close()
            return True
        else:
            return False
    elif skillType == SkillType.COMMAND:
        if coin >= leadership and general.skills_cd[2] == 0:
            general.skills_cd[2] = 10
            general.skill_duration[0] = 10
            gamestate.board[location[0]][location[1]].generals = general
            gamestate.coin[player] -= leadership
            replay = get_single_round_replay(
                gamestate,
                [],
                player,
                [4, general.id, 3],
            )
            with open(gamestate.replay_file, "a") as f:
                f.write(str(replay).replace("'", '"') + "\n")
            f.close()
            return True
        else:
            return False
    elif skillType == SkillType.DEFENCE:
        if coin >= fortification and general.skills_cd[3] == 0:
            general.skills_cd[3] = 10
            general.skill_duration[1] = 10
            gamestate.board[location[0]][location[1]].generals = general
            gamestate.coin[player] -= fortification
            replay = get_single_round_replay(
                gamestate,
                [],
                player,
                [4, general.id, 4],
            )
            with open(gamestate.replay_file, "a") as f:
                f.write(str(replay).replace("'", '"') + "\n")
            f.close()
            return True
        else:
            return False
    else:
        if coin >= weakening and general.skills_cd[4] == 0:
            general.skills_cd[4] = 10
            general.skill_duration[2] = 10
            gamestate.board[location[0]][location[1]].generals = general
            gamestate.coin[player] -= weakening
            replay = get_single_round_replay(
                gamestate,
                [],
                player,
                [4, general.id, 5],
            )
            with open(gamestate.replay_file, "a") as f:
                f.write(str(replay).replace("'", '"') + "\n")
            f.close()
            return True
        else:
            return False


"""
需要根据skilltype这个enum类型(定义在gamedata中)判断要使用那个技能
根据player以及将军的技能冷却,金钱数目来判断操作是否合法
如果必要可以在gamedata中添加属性(不可以删除）
如果不合法返回false,不进行任何操作
对于增益or减益,需要更改skillsactivated属性,而对于瞬移,可以调用movement文件中的general_move(location,gamestate,player,destination)->bool:
对于击破,判断合法性后直接操作gamestate即可
操作成功后需要扣除相应的金币并返回true
"""
