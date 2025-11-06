from logic.ai2logic import execute_single_command
from logic.gamedata import Direction


def test_execute_single_command_dispatches(plain_state):
    s = plain_state
    from tests.conftest import place_general

    # Prepare basic board
    s.board[1][1].player = 0
    s.board[1][1].army = 5
    # command 1: army move
    assert execute_single_command(0, s, 1, [1, 1, int(Direction.RIGHT) + 1, 2]) is True

    # command 7: call_generals (needs 50 coins and empty general spot on owned cell)
    s.coin[0] = 100
    s.board[1][2].player = 0
    assert execute_single_command(0, s, 7, [1, 2]) is True

    # Prepare a general for movement and upgrades/skills
    gen = place_general(s, "main", 2, 2, 0)
    s.board[2][2].player = 0
    s.board[2][3].player = 0
    gen.rest_move = 2

    # command 2: general move
    assert execute_single_command(0, s, 2, [gen.id, 2, 3]) is True

    # command 3: upgrades
    s.coin[0] = 1000
    assert execute_single_command(0, s, 3, [gen.id, 1]) is True  # production
    assert execute_single_command(0, s, 3, [gen.id, 2]) is True  # defence
    assert execute_single_command(0, s, 3, [gen.id, 3]) is True  # movement

    # command 5: tech update
    assert execute_single_command(0, s, 5, [1]) is True

    # command 4: skills
    s.coin[0] = 1000
    assert execute_single_command(0, s, 4, [gen.id, 3]) is True  # command no target

    # command 6: super weapons (unlock + ready)
    s.super_weapon_unlocked[0] = True
    s.super_weapon_cd[0] = 0
    assert execute_single_command(0, s, 6, [2, 2, 2]) is True  # strengthen

