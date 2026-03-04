from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from AI import ai_greedy, ai_handcraft
from logic.ai2logic import execute_single_command
from logic.computation import compute_attack, compute_defence
from logic.constant import col, row
from logic.gamedata import CellType, Direction, MainGenerals
from logic.game_rules import is_game_over
from logic.gamestate import GameState, update_round
from logic.map import PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP


BASE_CAMPS = (PLAYER_0_BASE_CAMP, PLAYER_1_BASE_CAMP)
MAX_CANDIDATES = 8
STRATEGIC_SCORE_SUM = float(sum(ai_handcraft._SITE_PRIORITY[0].values()))


def _clip_scale(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(np.clip(value / scale, -3.0, 3.0))


def clone_state(state: GameState) -> GameState:
    sim = ai_handcraft._clone_for_planning(state)
    sim.disable_replay = True
    return sim


def apply_ops_in_place(state: GameState, player: int, ops: list[list[int]]) -> None:
    for op in ops:
        if not op:
            continue
        if op[0] == 8:
            break
        execute_single_command(player, state, op[0], op[1:])


def simulate_turn(state: GameState, player: int, ops: list[list[int]]) -> GameState:
    sim = clone_state(state)
    apply_ops_in_place(sim, player, ops)
    if is_game_over(sim) == -1 and player == 1:
        update_round(sim)
    return sim


def _iter_cells(state: GameState, player: int) -> Iterable[tuple[int, int]]:
    for i in range(row):
        for j in range(col):
            if state.board[i][j].player == player:
                yield i, j


def _find_main(state: GameState, player: int) -> MainGenerals | None:
    for g in state.generals:
        if isinstance(g, MainGenerals) and g.player == player:
            return g
    return None


def _strategic_control_score(state: GameState, player: int) -> float:
    score = 0.0
    for pos, weight in ai_handcraft._SITE_PRIORITY[player].items():
        owner = state.board[pos[0]][pos[1]].player
        if owner == player:
            score += weight
        elif owner == 1 - player:
            score -= weight
    return score / STRATEGIC_SCORE_SUM


def _territory_stats(state: GameState, player: int) -> tuple[int, int, int, int]:
    own_cells = enemy_cells = own_army = enemy_army = 0
    for i in range(row):
        for j in range(col):
            cell = state.board[i][j]
            if cell.player == player:
                own_cells += 1
                own_army += int(cell.army)
            elif cell.player == 1 - player:
                enemy_cells += 1
                enemy_army += int(cell.army)
    return own_cells, enemy_cells, own_army, enemy_army


def _frontline_army(context: ai_handcraft.TurnContext) -> tuple[int, int]:
    frontier_cells = 0
    frontier_army = 0
    for pos, frontier in context.frontier.items():
        if frontier <= 0:
            continue
        frontier_cells += 1
        frontier_army += int(context.state.board[pos[0]][pos[1]].army)
    return frontier_cells, frontier_army


def _near_base_army(state: GameState, player: int, radius: int = 4) -> tuple[int, int]:
    base = BASE_CAMPS[player]
    own = enemy = 0
    for i in range(row):
        for j in range(col):
            if ai_handcraft._grid_distance((i, j), base) > radius:
                continue
            cell = state.board[i][j]
            if cell.player == player:
                own += int(cell.army)
            elif cell.player == 1 - player:
                enemy += int(cell.army)
    return own, enemy


def _combat_margin(state: GameState, player: int) -> float:
    margin = 0.0
    for src in _iter_cells(state, player):
        src_cell = state.board[src[0]][src[1]]
        if src_cell.army <= 1:
            continue
        attack = compute_attack(src_cell, state)
        available = src_cell.army - 1
        for _, dst in ai_handcraft._adjacent(src):
            if not ai_handcraft._inside(*dst):
                continue
            dst_cell = state.board[dst[0]][dst[1]]
            if dst_cell.player == player:
                continue
            if dst_cell.type == CellType.MOUNTAIN and state.tech_level[player][1] == 0:
                continue
            defence = compute_defence(dst_cell, state)
            margin += max(0.0, available * attack - dst_cell.army * defence)
    return _clip_scale(margin, 120.0)


def extract_features(state: GameState, player: int) -> np.ndarray:
    enemy = 1 - player
    context = ai_handcraft._build_context(state, player)
    enemy_context = ai_handcraft._build_context(state, enemy)
    own_cells, enemy_cells, own_army, enemy_army = _territory_stats(state, player)
    own_front_cells, own_front_army = _frontline_army(context)
    enemy_front_cells, enemy_front_army = _frontline_army(enemy_context)
    own_near_base, enemy_near_base = _near_base_army(state, player)
    enemy_near_enemy_base, own_near_enemy_base = _near_base_army(state, enemy)
    my_main = _find_main(state, player)
    enemy_main = _find_main(state, enemy)
    my_main_army = 0
    enemy_main_army = 0
    main_distance = 0.0
    main_prod = main_def = main_mob = 0.0
    enemy_prod = enemy_def = enemy_mob = 0.0
    if my_main is not None:
        x, y = my_main.position
        my_main_army = int(state.board[x][y].army)
        main_prod = float(my_main.produce_level)
        main_def = float(my_main.defense_level)
        main_mob = float(my_main.mobility_level)
    if enemy_main is not None:
        x, y = enemy_main.position
        enemy_main_army = int(state.board[x][y].army)
        enemy_prod = float(enemy_main.produce_level)
        enemy_def = float(enemy_main.defense_level)
        enemy_mob = float(enemy_main.mobility_level)
    if my_main is not None and enemy_main is not None:
        main_distance = _clip_scale(
            ai_handcraft._grid_distance(tuple(my_main.position), tuple(enemy_main.position)),
            20.0,
        )

    strategic_diff = _strategic_control_score(state, player)
    ant_pressure = _clip_scale(enemy_context.ant_dist - context.ant_dist, 8.0)
    capturable_margin = _combat_margin(state, player) - _combat_margin(state, enemy)

    features = np.array(
        [
            _clip_scale(state.round, 128.0),
            _clip_scale(state.base_hp[player], 50.0),
            _clip_scale(state.base_hp[enemy], 50.0),
            _clip_scale(state.base_hp[player] - state.base_hp[enemy], 25.0),
            _clip_scale(state.coin[player], 120.0),
            _clip_scale(state.coin[enemy], 120.0),
            _clip_scale(state.coin[player] - state.coin[enemy], 80.0),
            _clip_scale(state.rest_move_step[player], 3.0),
            _clip_scale(state.rest_move_step[enemy], 3.0),
            _clip_scale(state.rest_move_step[player] - state.rest_move_step[enemy], 2.0),
            _clip_scale(own_cells, 80.0),
            _clip_scale(enemy_cells, 80.0),
            _clip_scale(own_cells - enemy_cells, 40.0),
            _clip_scale(own_army, 180.0),
            _clip_scale(enemy_army, 180.0),
            _clip_scale(own_army - enemy_army, 120.0),
            _clip_scale(own_front_cells, 24.0),
            _clip_scale(enemy_front_cells, 24.0),
            _clip_scale(own_front_army - enemy_front_army, 80.0),
            _clip_scale(own_near_base - enemy_near_base, 60.0),
            _clip_scale(own_near_enemy_base - enemy_near_enemy_base, 60.0),
            _clip_scale(len(context.owned_subs), 4.0),
            _clip_scale(len(enemy_context.owned_subs), 4.0),
            _clip_scale(len(context.owned_subs) - len(enemy_context.owned_subs), 3.0),
            _clip_scale(context.bog_count, 12.0),
            _clip_scale(enemy_context.bog_count, 12.0),
            strategic_diff,
            ant_pressure,
            _clip_scale(my_main_army, 50.0),
            _clip_scale(enemy_main_army, 50.0),
            _clip_scale(my_main_army - enemy_main_army, 30.0),
            main_distance,
            _clip_scale(main_prod - enemy_prod, 2.0),
            _clip_scale(main_def - enemy_def, 2.0),
            _clip_scale(main_mob - enemy_mob, 4.0),
            _clip_scale(state.tech_level[player][0] - state.tech_level[enemy][0], 2.0),
            _clip_scale(state.tech_level[player][1] - state.tech_level[enemy][1], 1.0),
            _clip_scale(state.tech_level[player][2] - state.tech_level[enemy][2], 1.0),
            _clip_scale(state.tech_level[player][3] - state.tech_level[enemy][3], 1.0),
            _clip_scale(state.kill_count[player] - state.kill_count[enemy], 10.0),
            _clip_scale(state.superweapon_used[enemy] - state.superweapon_used[player], 4.0),
            capturable_margin,
        ],
        dtype=np.float32,
    )
    return features


def heuristic_value(state: GameState, player: int) -> float:
    features = extract_features(state, player)
    weights = np.array(
        [
            -0.10, 0.35, -0.35, 0.85, 0.18, -0.18, 0.45, 0.10, -0.10, 0.18,
            0.18, -0.18, 0.36, 0.30, -0.30, 0.70, 0.10, -0.10, 0.25, 0.40,
            0.30, 0.12, -0.12, 0.22, -0.05, 0.05, 0.65, 0.45, 0.18, -0.18,
            0.42, -0.08, 0.20, 0.22, 0.12, 0.16, 0.10, 0.08, 0.05, 0.20,
            0.10, 0.55,
        ],
        dtype=np.float32,
    )
    score = float(features @ weights)
    return math.tanh(score / 3.5)


def _top_moves(
    context: ai_handcraft.TurnContext,
    target: tuple[int, int],
    limit: int,
) -> list[list[int]]:
    scored: dict[tuple[int, ...], float] = {}
    state = context.state
    player = context.player
    for src in ai_handcraft._candidate_sources(context, target):
        src_cell = state.board[src[0]][src[1]]
        available = src_cell.army - 1
        if available <= 0:
            continue
        for direction, dst in ai_handcraft._adjacent(src):
            if not ai_handcraft._inside(*dst):
                continue
            required = ai_handcraft._required_force(state, player, src, dst)
            nums = {1, required, available}
            if available > 3:
                nums.add(3)
            for num in nums:
                plan = ai_handcraft._plan_army_move(context, src, direction, num)
                if plan is None:
                    continue
                score = ai_handcraft._score_move(context, target, plan)
                key = tuple(plan.op)
                prev = scored.get(key)
                if prev is None or score > prev:
                    scored[key] = score
    top = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    return [list(op) for op, _ in top[:limit] if _ > -10_000.0]


def _compose_plan(
    state: GameState,
    player: int,
    round_idx: int,
    *,
    target: tuple[int, int] | None = None,
    first_move_rank: int = 0,
    allow_build: bool = True,
    allow_tech: bool = True,
    allow_upgrade: bool = True,
) -> list[list[int]]:
    sim = clone_state(state)
    ops: list[list[int]] = []
    context = ai_handcraft._build_context(sim, player)

    if allow_build and ai_handcraft._maybe_build_defender(sim, context, ops, round_idx):
        context = ai_handcraft._build_context(sim, player)
    if allow_tech and ai_handcraft._maybe_upgrade_tech(sim, context, ops):
        context = ai_handcraft._build_context(sim, player)
    if allow_upgrade and ai_handcraft._maybe_upgrade_generals(sim, context, ops):
        context = ai_handcraft._build_context(sim, player)
    if allow_build and ai_handcraft._maybe_build_defender(sim, context, ops, round_idx):
        context = ai_handcraft._build_context(sim, player)

    pick_rank = first_move_rank
    while sim.rest_move_step[player] > 0:
        chosen_target = target if target is not None else ai_handcraft._strategic_target(context)
        moves = _top_moves(context, chosen_target, max(4, pick_rank + 1))
        if not moves:
            break
        move = moves[pick_rank] if pick_rank < len(moves) else moves[0]
        pick_rank = 0
        if not ai_handcraft._try_add_op(sim, player, ops, move):
            break
        context = ai_handcraft._build_context(sim, player)

    ops.append([8])
    return ops


def candidate_turn_plans(
    state: GameState,
    player: int,
    round_idx: int,
    max_candidates: int = MAX_CANDIDATES,
) -> list[list[list[int]]]:
    context = ai_handcraft._build_context(state, player)
    enemy_main = context.enemy_main_pos or BASE_CAMPS[1 - player]
    defend_target = context.urgent_ant_pos or BASE_CAMPS[player]
    strategic = ai_handcraft._strategic_target(context)

    candidates: list[list[list[int]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()

    def add(ops: list[list[int]]) -> None:
        normalized = [list(op) for op in ops if op]
        if not normalized or normalized[-1] != [8]:
            normalized.append([8])
        key = tuple(tuple(op) for op in normalized)
        if key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    add(ai_handcraft.policy(round_idx, player, state))
    add(_compose_plan(state, player, round_idx, target=strategic))
    add(_compose_plan(state, player, round_idx, target=strategic, first_move_rank=1))
    add(_compose_plan(state, player, round_idx, target=enemy_main, allow_build=False))
    add(_compose_plan(state, player, round_idx, target=enemy_main, first_move_rank=1, allow_build=False))
    add(_compose_plan(state, player, round_idx, target=defend_target, allow_tech=False))
    add(ai_greedy.policy(round_idx, player, state))
    add([[8]])

    return candidates[:max_candidates]
