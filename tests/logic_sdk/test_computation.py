from logic.computation import compute_attack, compute_defence
from logic.gamedata import SuperWeapon, WeaponType


def test_compute_attack_and_defence_with_buffs(plain_state):
    s = plain_state
    # Make target cell owned by player 0
    s.board[5][5].player = 0
    # Friendly general with command buff nearby
    from tests.conftest import place_general

    g_f = place_general(s, "main", 5, 6, 0)
    g_f.skill_duration[0] = 1  # command
    # Enemy general with weaken nearby
    g_e = place_general(s, "sub", 6, 6, 1)
    g_e.skill_duration[2] = 1  # weaken

    # Attack enhance weapon affecting the cell
    s.active_super_weapon.append(
        SuperWeapon(type=WeaponType.ATTACK_ENHANCE, player=0, cd=2, rest=2, position=[5, 5])
    )
    atk = compute_attack(s.board[5][5], s)
    # base 1.0 * 1.5 (command) * 0.75 (weaken) * 3 (weapon)
    assert atk == 1.0 * 1.5 * 0.75 * 3

    # Defence considers friendly defence buff and defence level
    # Move the friendly general onto the cell to apply defence level once
    s.board[5][6].generals = None
    s.board[5][5].generals = g_f
    g_f.position = [5, 5]
    g_f.defense_level = 2
    g_f.skill_duration[1] = 1  # defence
    # swap durations to isolate
    g_f.skill_duration[0] = 0
    g_e.skill_duration[2] = 1
    dfn = compute_defence(s.board[5][5], s)
    # base 1.0 * 1.5 (def buff) * 0.75 (enemy weaken) * 2 (def level) * 3 (weapon)
    assert dfn == 1.0 * 1.5 * 0.75 * 2 * 3
