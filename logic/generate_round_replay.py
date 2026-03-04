import copy
from logic.constant import row, col


def replay_enabled(gamestate) -> bool:
    return not getattr(gamestate, "disable_replay", False)


def append_replay_line(gamestate, replay: dict) -> None:
    if not replay_enabled(gamestate):
        return
    with open(gamestate.replay_file, "a") as f:
        f.write(str(replay).replace("'", '"') + "\n")


def get_single_round_replay(gamestate, cells: list[list[int, int]], player: int, action: list):
    replay = {
        "Round": gamestate.round,
        "Player": player,
        "Action": action,
        "Cells": [],
        "Generals": [],
    }

    replay["Weapons"] = [
        {
            "Type": int(weapon.type) + 1,
            "Player": weapon.player,
            "Position": weapon.position,
            "Rest": weapon.rest,
        }
        for weapon in gamestate.active_super_weapon
    ]

    for cell in cells:
        replay["Cells"].append(
            [
                cell,
                gamestate.board[cell[0]][cell[1]].player,
                gamestate.board[cell[0]][cell[1]].army,
            ]
        )

    replay["Weapon_cds"] = gamestate.super_weapon_cd

    replay["Tech_level"] = copy.deepcopy(gamestate.tech_level)
    replay["Tech_level"][0][0] = (gamestate.tech_level[0][0] + 1) // 2
    replay["Tech_level"][1][0] = (gamestate.tech_level[1][0] + 1) // 2

    replay["Coins"] = gamestate.coin

    for general in gamestate.generals:
        replay["Generals"].append(
            {
                "Id": general.id,
                "Player": general.player,
                "Type": 1
                if (type(general).__name__ == "MainGenerals")
                else (2 if type(general).__name__ == "SubGenerals" else 3),
                "Position": general.position,
                "Level": [
                    general.produce_level // 2 + 1,
                    general.defense_level,
                    general.mobility_level // 2 + 1,
                ],
                "Skill_cd": general.skills_cd,
                "Skill_rest": general.skill_duration,
                "Alive": 1,
            }
        )
        if type(general).__name__ == "Farmer":
            if general.defense_level == 1.5:
                replay["Generals"][-1]["Level"][1] = 2
            elif general.defense_level == 2 or general.defense_level == 3:
                replay["Generals"][-1]["Level"][1] = general.defense_level + 1

    # FIX: 生成真实 Cell_type, 不要在循环里反复置空
    cell_type = []
    for i in range(row):
        for j in range(col):
            cell_type.append(str(int(gamestate.board[i][j].type)))
    replay["Cell_type"] = "".join(cell_type)

    return replay
