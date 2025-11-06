from logic.call_generals import call_generals
from logic.gamestate import GameState
from logic.general_skills import *
from logic.movement import *
from logic.super_weapons import *
from logic.upgrade import *


def execute_single_command(
    player: int, state: GameState, command: int, params: list = []
) -> bool:
    if command == 1:
        # print("move army", params)
        return army_move(
            [params[0], params[1]], state, player, Direction(params[2] - 1), params[3]
        )
    elif command == 2:
        if (pos := state.find_general_position_by_id(params[0])) != None:
            # print("move generals", params)
            if check_general_movement(pos, state, player, [params[1], params[2]]):
                general_move(pos, state, player, [params[1], params[2]])
                return True
    elif command == 3:
        if (pos := state.find_general_position_by_id(params[0])) != None:
            # print("level up", params)
            if params[1] == 1:
                return production_up(pos, state, player)
            elif params[1] == 2:
                return defence_up(pos, state, player)
            elif params[1] == 3:
                return movement_up(pos, state, player)
    elif command == 4:
        # print("generals skill", params)
        if (pos := state.find_general_position_by_id(params[0])) != None:
            if params[1] == 1 or params[1] == 2:
                return skill_activate(
                    player, pos, params[2:4], state, skillType=(params[1] - 1)
                )
            else:
                return skill_activate(
                    player, pos, [-1, -1], state, skillType=(params[1] - 1)
                )
    elif command == 5:
        # print("update tech", params)
        return tech_update(params[0] - 1, state, player)
    elif command == 6:
        # print("use superweapon", params)
        if params[0] == 1:
            return bomb(state, params[1:3], player)
        elif params[0] == 2:
            return strengthen(state, params[1:3], player)
        elif params[0] == 3:
            return tp(state, params[3:], params[1:3], player)
        elif params[0] == 4:
            return timestop(state, params[1:3], player)
    elif command == 7:
        # print("call generals", params)
        return call_generals(state, player, params)
    return False
