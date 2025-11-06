# 本文件中的函数负责计算攻击敌方格子时的attack和defence参数
from logic.constant import *
from logic.gamedata import *


def compute_attack(cell: Cell, gamestate) -> float:
    attack = 1.0
    cell_x = cell.position[0]
    cell_y = cell.position[1]
    # 遍历cell周围至少5*5的区域，寻找里面是否有将军，他们是否使用了增益或减益技能
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = cell_x + i
            y = cell_y + j
            if 0 <= x < row and 0 <= y < col:
                neighbor_cell = gamestate.board[x][y]
                if neighbor_cell.generals is not None:
                    if (
                        neighbor_cell.player == cell.player
                        and neighbor_cell.generals.skill_duration[0] > 0
                    ):
                        attack = attack * 1.5
                    if (
                        neighbor_cell.player != cell.player
                        and neighbor_cell.generals.skill_duration[2] > 0
                    ):
                        attack = attack * 0.75
    # 考虑gamestate中的超级武器是否被激活，（可以获取到激活的位置）该位置的军队是否会被影响
    for active_weapon in gamestate.active_super_weapon:
        if (
            active_weapon.type == WeaponType.ATTACK_ENHANCE
            and cell_x - 1 <= active_weapon.position[0] <= cell_x + 1
            and cell_y - 1 <= active_weapon.position[1] <= cell_y + 1
            and active_weapon.player == cell.player
        ):
            attack = attack * 3
            break
    return attack


def compute_defence(cell: Cell, gamestate) -> float:
    defence = 1.0
    cell_x = cell.position[0]
    cell_y = cell.position[1]
    # 遍历cell周围至少5*5的区域，寻找里面是否有将军，他们是否使用了增益或减益技能
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = cell_x + i
            y = cell_y + j
            if 0 <= x < row and 0 <= y < col:
                neighbor_cell = gamestate.board[x][y]
                if neighbor_cell.generals is not None:
                    if (
                        neighbor_cell.player == cell.player
                        and neighbor_cell.generals.skill_duration[1] > 0
                    ):
                        defence = defence * 1.5
                    if (
                        neighbor_cell.player != cell.player
                        and neighbor_cell.generals.skill_duration[2] > 0
                    ):
                        defence = defence * 0.75
    # 考虑cell上是否有general，它的防御力是否被升级
    if cell.generals is not None:
        defence = defence * cell.generals.defense_level
    # 考虑gamestate中的超级武器是否被激活，（可以获取到激活的位置）该位置的军队是否会被影响
    for active_weapon in gamestate.active_super_weapon:
        if (
            active_weapon.type == WeaponType.ATTACK_ENHANCE
            and cell_x - 1 <= active_weapon.position[0] <= cell_x + 1
            and cell_y - 1 <= active_weapon.position[1] <= cell_y + 1
            and active_weapon.player == cell.player
        ):
            defence = defence * 3
            break
    return defence


"""
在这一部分，你需要根据格子上的将军以及被激活的效果，计算出cell的攻击力和防御力，并返回一个float类型的值
首先，你需要遍历cell周围至少5*5的区域，寻找里面是否有将军，他们是否使用了增益或减益技能
其次，你需要考虑cell上是否有general，它的防御力是否被升级
最后，你需要考虑gamestate中的超级武器是否被激活，（可以获取到激活的位置）该位置的军队是否会被影响
将这些因素综合起来，返回一个数
"""
