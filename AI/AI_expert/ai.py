from __future__ import annotations

import itertools
import time
from typing import List, Optional, Sequence, Tuple

try:
    from antwar.core import (
        MAX_ROUND,
        MAP_PROPERTY,
        Ant,
        AntState,
        BuildingType,
        GameInfo,
        Operation,
        OperationType,
        Simulator,
        SuperWeaponType,
        Tower,
        TowerType,
        distance,
        is_valid_pos,
    )
except ModuleNotFoundError as exc:
    if exc.name != "antwar":
        raise
    from AI.AI_expert.antwar.core import (
        MAX_ROUND,
        MAP_PROPERTY,
        Ant,
        AntState,
        BuildingType,
        GameInfo,
        Operation,
        OperationType,
        Simulator,
        SuperWeaponType,
        Tower,
        TowerType,
        distance,
        is_valid_pos,
    )

SEARCH_BUDGET = 0.15
MAX_NODE_COUNT = 20000

SITE_LAYOUT = (
    (
        (2, 9),
        (4, 9),
        (5, 9),
        (5, 7),
        (6, 9),
        (5, 11),
        (5, 6),
        (6, 7),
        (6, 11),
        (5, 12),
        (4, 3),
        (5, 3),
        (7, 8),
        (7, 10),
        (4, 15),
        (5, 15),
        (4, 2),
        (6, 4),
        (7, 5),
        (8, 7),
        (8, 11),
        (7, 13),
        (6, 14),
        (4, 16),
        (6, 1),
        (6, 2),
        (6, 16),
        (6, 17),
        (7, 1),
        (8, 4),
        (8, 14),
        (7, 17),
        (8, 2),
        (8, 16),
        (3, 9),
    ),
    (
        (16, 9),
        (14, 9),
        (13, 9),
        (13, 7),
        (12, 9),
        (13, 11),
        (12, 6),
        (12, 7),
        (12, 11),
        (12, 12),
        (14, 3),
        (13, 3),
        (10, 8),
        (10, 10),
        (14, 15),
        (13, 15),
        (13, 2),
        (11, 4),
        (11, 5),
        (10, 7),
        (10, 11),
        (11, 13),
        (11, 14),
        (13, 16),
        (12, 1),
        (11, 2),
        (11, 16),
        (12, 17),
        (11, 1),
        (9, 4),
        (9, 14),
        (11, 17),
        (9, 2),
        (9, 16),
        (15, 9),
    ),
)

HOME_SLOT = 0
STORM_SLOT = 34
ACTIONABLE_SITES = (
    1,
    2,
    4,
    10,
    16,
    11,
    14,
    23,
    15,
    17,
    18,
    22,
    21,
    3,
    6,
    7,
    5,
    8,
    9,
    19,
    12,
    13,
    20,
    24,
    25,
    28,
    27,
    26,
    31,
)

SITE_FAMILIES = (
    (1, 2, 4),
    (3, 6, 7),
    (5, 8, 9),
    (10, 16, 11),
    (14, 23, 15),
    (19, 12, 13, 20),
    (17, 18),
    (22, 21),
    (24, 25, 28),
    (27, 26, 31),
    (32, 29),
    (30, 33),
)

SITE_TO_FAMILY = {}
for family in SITE_FAMILIES:
    for site in family:
        SITE_TO_FAMILY[site] = family


class ForecastNode:
    def __init__(self, brain: AI, sim: Simulator) -> None:
        self.brain = brain
        self.sim = sim.clone()
        self.node_id = -1
        self.parent = -1
        self.children: List[int] = []
        self.chosen: List[Operation] = []
        self.score = 0.0
        self.best_descendant = 0.0
        self.round_tag = self.sim.info.round
        self.sunk_cost = 0.0
        self.best_depth = 0
        self.expanded_layers = 0
        self.collapse_round = 0
        self.danger = False
        self.solvent = True
        self.distance_trace = [0] * 60

    @property
    def action_count(self) -> int:
        return len(self.chosen)

    def evaluate(self) -> float:
        trial = self.sim.clone()
        info = trial.info
        brain = self.brain

        if info.round - brain.current_round < 60:
            self.distance_trace[info.round - brain.current_round] = brain._nearest_hostile_step(info)

        safe_gap = 0
        if brain.current_round > 60:
            safe_gap = brain._cash_safety_gap(info)
            self.solvent = safe_gap == 0

        ruin_round = brain.current_round + 60
        already_cracked = False
        if info.bases[brain.side].hp <= brain.wall_hp_snapshot - 1:
            already_cracked = True
            ruin_round = self.collapse_round

        if not already_cracked:
            self.collapse_round = brain.current_round + 60
            for turn in range(info.round, brain.current_round + 60):
                if not trial.fast_next_round(brain.side):
                    break
                self.distance_trace[turn - brain.current_round] = brain._nearest_hostile_step(info)
                if info.bases[brain.side].hp <= brain.wall_hp_snapshot - 1:
                    self.collapse_round = info.round
                    if info.bases[brain.side].hp <= brain.wall_hp_snapshot - 2:
                        ruin_round = self.collapse_round
                    break

        if info.bases[brain.side].hp > brain.wall_hp_snapshot - 2:
            for turn in range(info.round, brain.current_round + 60):
                if not trial.fast_next_round(brain.side):
                    break
                self.distance_trace[turn - brain.current_round] = brain._nearest_hostile_step(info)
                if info.bases[brain.side].hp <= brain.wall_hp_snapshot - 2:
                    ruin_round = info.round
                    break

        ant_weight = {0: 3.0, 1: 5.0, 2: 7.0}[info.bases[1 - brain.side].ant_level]
        score = (
            (info.bases[brain.side].hp - brain.wall_hp_snapshot)
            + (self.collapse_round - brain.current_round) * 0.8
            + (ruin_round - self.collapse_round) * 0.1
            - self.sunk_cost * 1.5
            + 20
        )

        if brain.front_state == 0:
            score += -(info.old_count[1 - brain.side] - brain.enemy_old_baseline) * ant_weight * 2
            score += (info.die_count[1 - brain.side] - brain.enemy_die_baseline) * ant_weight * 1.5

        self.danger = False
        if self.collapse_round - brain.current_round <= 16:
            self.danger = True
            score -= 500
            if ruin_round - self.collapse_round <= 8:
                score -= 300

        if not self.solvent and not self.danger and brain.front_state >= 0:
            score += (-40 + safe_gap / 5) * min((brain.current_round - 60) / 30, 1)

        tower_count = info.tower_num_of_player(brain.side)
        score -= (pow(2, tower_count) - 1) * 15 * 0.2 * 0.75

        tower_positions: List[Tuple[int, int]] = []
        distanced = False
        for tower in info.towers:
            if tower.player != brain.side:
                continue
            if int(tower.type) > 0 and int(tower.type) // 10 == 0:
                score -= 60 * 0.2 * 0.75
            elif int(tower.type) // 10 > 0:
                score -= 260 * 0.2 * 0.75
            tower_positions.append((tower.x, tower.y))

        for idx, (x0, y0) in enumerate(tower_positions[:-1]):
            for x1, y1 in tower_positions[idx + 1 :]:
                gap = distance(x0, y0, x1, y1)
                if gap <= 3:
                    score -= 5
                elif gap <= 6:
                    score -= 2
                else:
                    distanced = True

        if tower_count >= 3 and not distanced:
            score -= 20

        base = info.bases[brain.side]
        for x, y in tower_positions:
            score += distance(x, y, base.x, base.y) * 0.4

        if brain.front_state >= 0:
            close_flag = False
            for idx in range(min(60, info.round - brain.current_round - 4)):
                if self.distance_trace[idx] <= 3:
                    close_flag = True
                if self.distance_trace[idx] == 5:
                    score -= 0.2
                elif self.distance_trace[idx] == 4:
                    score -= 2.5
                elif self.distance_trace[idx] in (1, 2, 3):
                    score -= 2
            if close_flag:
                score -= 20

            enemy_count = 0
            pressure = 0.0
            for ant in info.ants:
                if ant.player != 1 - brain.side:
                    continue
                pressure += 32 - ant.age - distance(ant.x, ant.y, base.x, base.y) * 1.5
                enemy_count += 1
            if enemy_count > 0 and brain.current_round >= 20:
                score += pressure / enemy_count * 0.5

        self.score = score
        self.best_descendant = score
        return score

    def expand(self, is_root: bool = False) -> None:
        brain = self.brain
        info = self.sim.info
        if info.round >= MAX_ROUND or info.bases[brain.side].hp <= 0 or info.bases[1 - brain.side].hp <= 0:
            return

        if not is_root:
            if info.round - brain.current_round < 60:
                self.distance_trace[info.round - brain.current_round] = brain._nearest_hostile_step(info)
            if not self.sim.fast_next_round(brain.side):
                return

        emp_blocked = [False] * 34
        for weapon in self.sim.info.super_weapons:
            if weapon.player == 1 - brain.side and weapon.type == SuperWeaponType.EMP_BLASTER:
                for site in range(34):
                    sx, sy = SITE_LAYOUT[brain.side][site]
                    if distance(weapon.x, weapon.y, sx, sy) <= 3:
                        emp_blocked[site] = True
                break

        bundles: List[List[Operation]] = []
        for tactic in range(8):
            if self.action_count > 0 and tactic in (3, 5):
                continue
            if (
                self.action_count == 1
                and self.chosen[0].type == OperationType.BUILD_TOWER
                and self.expanded_layers < 2
                and tactic in (3, 4, 6)
            ):
                continue
            if (
                self.action_count == 1
                and self.chosen[0].type == OperationType.UPGRADE_TOWER
                and self.expanded_layers < 2
                and tactic == 2
            ):
                continue
            if (
                self.action_count == 2
                and self.chosen[1].type == OperationType.BUILD_TOWER
                and self.expanded_layers < 2
                and tactic in (3, 4, 6)
            ):
                continue
            if self.sim.info.tower_num_of_player(brain.side) >= 4 and tactic in (0, 2):
                continue
            bundles.extend(brain._candidate_bundles(tactic, self.sim.info, emp_blocked))

        if is_root:
            idle = ForecastNode(brain, self.sim)
            idle.node_id = len(brain.nodes)
            idle.parent = self.node_id
            idle.evaluate()
            brain.nodes.append(idle)
            self.children.append(idle.node_id)

        for bundle in bundles:
            if len(brain.nodes) >= MAX_NODE_COUNT - 10:
                break
            child = ForecastNode(brain, self.sim)
            child.node_id = len(brain.nodes)
            child.parent = self.node_id
            child.sunk_cost = self.sunk_cost
            child.chosen = list(bundle)
            child.collapse_round = self.collapse_round
            child.score = -1e9
            child.best_descendant = -1e9
            if self.sim.info.round > brain.current_round:
                trace_len = min(60, self.sim.info.round - brain.current_round)
                child.distance_trace[:trace_len] = self.distance_trace[:trace_len]
            child.sim.operations[0].clear()
            child.sim.operations[1].clear()
            mutable = child.sim.info
            for op in bundle:
                if op.type == OperationType.DOWNGRADE_TOWER:
                    tower = brain._tower_by_id(op.arg0, mutable)
                    if tower is not None:
                        if tower.type == TowerType.BASIC:
                            child.sunk_cost += mutable.build_tower_cost(mutable.tower_num_of_player(brain.side)) * 0.2
                        else:
                            child.sunk_cost += mutable.upgrade_tower_cost(int(tower.type)) * 0.2
                child.sim.add_operation_of_player(brain.side, op)
            child.sim.apply_operations_of_player(brain.side)
            value = child.evaluate()
            if value > self.best_descendant:
                self.best_descendant = value
                self.best_depth = self.expanded_layers + 1
            brain.nodes.append(child)
            self.children.append(child.node_id)

        if is_root and not self.sim.fast_next_round(brain.side):
            return
        self.expanded_layers += 1


class AI:
    def __init__(self) -> None:
        self.side = 0
        self.current_round = 0
        self.front_state = 0
        self.wall_hp_snapshot = 0
        self.enemy_old_baseline = 0
        self.enemy_die_baseline = 0
        self.assault_memory = False
        self.last_superweapon_type: Optional[SuperWeaponType] = None
        self.last_superweapon_round = -1
        self.reserve_depth = 0
        self.nodes: List[ForecastNode] = []

    def create_session(self):
        try:
            from runtime import ExpertSession
        except ModuleNotFoundError as exc:
            if exc.name != "runtime":
                raise
            from AI.AI_expert.runtime import ExpertSession
        return ExpertSession(self)

    def _mark_super(self, weapon_type: SuperWeaponType) -> None:
        self.last_superweapon_round = self.current_round
        self.last_superweapon_type = weapon_type

    def _tower_at(self, x: int, y: int, info: GameInfo) -> Optional[Tower]:
        for tower in info.towers:
            if tower.x == x and tower.y == y:
                return tower
        return None

    def _tower_by_id(self, tower_id: int, info: GameInfo) -> Optional[Tower]:
        return info.tower_of_id(tower_id)

    def _nearest_push_distance(self, info: GameInfo) -> int:
        best = 100
        tx, ty = SITE_LAYOUT[1 - self.side][HOME_SLOT]
        for ant in info.ants:
            if ant.player == self.side:
                best = min(best, distance(ant.x, ant.y, tx, ty))
        return best

    def _opponent_emp_buffer(self, info: GameInfo) -> int:
        cd = info.super_weapon_cd[1 - self.side][int(SuperWeaponType.EMP_BLASTER)]
        if cd >= 90:
            return 0
        if cd > 0:
            return max(int(min(info.coins[1 - self.side], 149) - cd * 1.66), 0)
        return min(info.coins[1 - self.side], 149)

    def _cash_safety_gap(self, info: GameInfo) -> int:
        return min(0, info.coins[self.side] - self._opponent_emp_buffer(info))

    def _nearest_hostile_step(self, info: GameInfo) -> int:
        best = 32
        tx, ty = SITE_LAYOUT[self.side][HOME_SLOT]
        for ant in info.ants:
            if ant.player == 1 - self.side:
                best = min(best, distance(ant.x, ant.y, tx, ty))
        return best

    def _site_operation(
        self,
        site: int,
        mode: int,
        info: GameInfo,
        coins: int,
        towers: int,
        upgrade_branch: int = 0,
        exempt_site: int = -1,
    ) -> Tuple[Optional[Operation], int, int]:
        x, y = SITE_LAYOUT[self.side][site]

        if mode == 1:
            cost = info.build_tower_cost(towers)
            if coins < cost:
                return None, coins, towers
            for peer in SITE_TO_FAMILY[site]:
                if peer == exempt_site:
                    continue
                px, py = SITE_LAYOUT[self.side][peer]
                if info.building_tag[px][py] != BuildingType.EMPTY:
                    return None, coins, towers
            return Operation(OperationType.BUILD_TOWER, x, y), coins - cost, towers + 1

        if mode == 2:
            if info.building_tag[x][y] == BuildingType.EMPTY:
                return None, coins, towers
            tower = self._tower_at(x, y, info)
            if tower is None or int(tower.type) // 10 > 0:
                return None, coins, towers

            target: Optional[TowerType] = None
            if tower.type == TowerType.BASIC:
                target = (TowerType.HEAVY, TowerType.MORTAR, TowerType.QUICK)[upgrade_branch]
            elif tower.type == TowerType.HEAVY:
                target = (TowerType.HEAVY_PLUS, TowerType.CANNON, TowerType.ICE)[upgrade_branch]
            elif tower.type == TowerType.MORTAR:
                target = (TowerType.MORTAR_PLUS, TowerType.MISSILE, TowerType.PULSE)[upgrade_branch]
            elif tower.type == TowerType.QUICK:
                target = (TowerType.QUICK_PLUS, TowerType.DOUBLE, TowerType.SNIPER)[upgrade_branch]

            if target is None:
                return None, coins, towers
            cost = info.upgrade_tower_cost(int(target))
            if coins < cost:
                return None, coins, towers
            return Operation(OperationType.UPGRADE_TOWER, tower.id, int(target)), coins - cost, towers

        if mode == 3:
            if info.building_tag[x][y] == BuildingType.EMPTY:
                return None, coins, towers
            tower = self._tower_at(x, y, info)
            if tower is None or tower.type != TowerType.BASIC:
                return None, coins, towers
            refund = info.destroy_tower_income(towers)
            return Operation(OperationType.DOWNGRADE_TOWER, tower.id), coins + refund, towers - 1

        if mode == 4:
            if info.building_tag[x][y] == BuildingType.EMPTY:
                return None, coins, towers
            tower = self._tower_at(x, y, info)
            if tower is None or tower.type == TowerType.BASIC:
                return None, coins, towers
            refund = info.downgrade_tower_income(int(tower.type))
            return Operation(OperationType.DOWNGRADE_TOWER, tower.id), coins + refund, towers

        return None, coins, towers

    def _candidate_bundles(self, tactic: int, info: GameInfo, emp_blocked: Sequence[bool]) -> List[List[Operation]]:
        bundles: List[List[Operation]] = []

        if tactic == 0:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                op, _, _ = self._site_operation(site, 1, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if op is not None:
                    bundles.append([op])
        elif tactic == 1:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                for branch in range(3):
                    op, _, _ = self._site_operation(
                        site,
                        2,
                        info,
                        info.coins[self.side],
                        info.tower_num_of_player(self.side),
                        branch,
                    )
                    if op is not None:
                        bundles.append([op])
        elif tactic == 2:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                head, coins, towers = self._site_operation(
                    site,
                    4,
                    info,
                    info.coins[self.side],
                    info.tower_num_of_player(self.side),
                )
                if head is None:
                    continue
                for site2 in ACTIONABLE_SITES:
                    if emp_blocked[site2] or site2 == site:
                        continue
                    tail, _, _ = self._site_operation(site2, 1, info, coins, towers)
                    if tail is not None:
                        bundles.append([head, tail])
        elif tactic == 3:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                op, _, _ = self._site_operation(site, 3, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if op is not None:
                    bundles.append([op])
        elif tactic == 4:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                head, coins, towers = self._site_operation(
                    site,
                    3,
                    info,
                    info.coins[self.side],
                    info.tower_num_of_player(self.side),
                )
                if head is None:
                    continue
                for site2 in ACTIONABLE_SITES:
                    if emp_blocked[site2] or site2 == site:
                        continue
                    for branch in range(3):
                        tail, _, _ = self._site_operation(site2, 2, info, coins, towers, branch)
                        if tail is not None:
                            bundles.append([head, tail])
        elif tactic == 5:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                op, _, _ = self._site_operation(site, 4, info, info.coins[self.side], info.tower_num_of_player(self.side))
                if op is not None:
                    bundles.append([op])
        elif tactic == 6:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                head, coins, towers = self._site_operation(
                    site,
                    3,
                    info,
                    info.coins[self.side],
                    info.tower_num_of_player(self.side),
                )
                if head is None:
                    continue
                for site2 in ACTIONABLE_SITES:
                    if emp_blocked[site2] or site2 == site:
                        continue
                    tail, _, _ = self._site_operation(site2, 1, info, coins, towers, exempt_site=site)
                    if tail is not None:
                        bundles.append([head, tail])
        elif tactic == 7:
            for site in ACTIONABLE_SITES:
                if emp_blocked[site]:
                    continue
                head, coins, towers = self._site_operation(
                    site,
                    4,
                    info,
                    info.coins[self.side],
                    info.tower_num_of_player(self.side),
                )
                if head is None:
                    continue
                for site2 in ACTIONABLE_SITES:
                    if emp_blocked[site2] or site2 == site:
                        continue
                    for branch in range(3):
                        tail, _, _ = self._site_operation(site2, 2, info, coins, towers, branch)
                        if tail is not None:
                            bundles.append([head, tail])

        return bundles

    def _expand_one(self) -> bool:
        root = self.nodes[0]
        if not root.children:
            return False

        target_id = -1
        best = -1e9
        for child_id in root.children:
            child = self.nodes[child_id]
            value = -child.expanded_layers
            if child_id == 0:
                value += self.reserve_depth
            if not child.children:
                value += 1000
            if child.danger:
                value += 20
            if not child.solvent:
                value -= 20
            if value > best:
                best = value
                target_id = child_id
        if target_id < 0:
            return False
        self.nodes[target_id].expand()
        return True

    def _support_expand(self, bias: int) -> None:
        root = self.nodes[0]
        if not root.children:
            return
        for child_id in root.children:
            child = self.nodes[child_id]
            if child.collapse_round - self.current_round > 24:
                continue
            now_round = child.sim.info.round
            target_round = min(MAX_ROUND - 1, child.collapse_round - bias)
            if now_round >= target_round:
                continue
            for _ in range(now_round, target_round - 1):
                if not child.sim.fast_next_round(self.side):
                    break
            child.expand()

    def _liquidate_all(
        self, coins: int, towers: int, coin_need: int, info: GameInfo
    ) -> Optional[Tuple[List[Operation], int, int]]:
        ops: List[Operation] = []
        max_coins = coins
        valid_tower_num = 0
        start_tower_num = info.tower_num_of_player(self.side)

        for tower in info.towers:
            if tower.player != self.side or info.tower_under_emp(tower):
                continue
            if tower.type == TowerType.BASIC:
                coins += info.destroy_tower_income(towers)
                towers -= 1
            else:
                coins += info.downgrade_tower_income(int(tower.type))
                if int(tower.type) // 10 == 0:
                    max_coins += 48
                else:
                    max_coins += 48 + 160
            ops.append(Operation(OperationType.DOWNGRADE_TOWER, tower.id))
            valid_tower_num += 1
            if coins >= coin_need:
                return ops, coins, towers

        max_coins += (pow(2, start_tower_num) - pow(2, start_tower_num - valid_tower_num)) * 12
        if max_coins >= coin_need:
            return ops, coins, towers
        return None

    def _liquidate_cautious(
        self, coins: int, towers: int, coin_need: int, info: GameInfo
    ) -> Optional[Tuple[List[Operation], int, int]]:
        tower_ids = [tower.id for tower in info.towers if tower.player == self.side and not info.tower_under_emp(tower)]
        if not tower_ids:
            return None

        baseline = Simulator(info)
        fallback_round = 48
        for step in range(1, 49):
            if not baseline.fast_next_round(self.side):
                break
            if baseline.info.bases[self.side].hp < info.bases[self.side].hp:
                fallback_round = step
                break

        max_round = -1
        max_coins = coins
        best_ops: List[Operation] = []
        for order in itertools.permutations(tower_ids):
            plan: List[Operation] = []
            trial = Simulator(info)
            snapshot = trial.info
            wallet = coins
            tower_count = towers
            valid = False

            for tower_id in order:
                tower = self._tower_by_id(tower_id, snapshot)
                if tower is None:
                    continue
                if tower.type == TowerType.BASIC:
                    wallet += snapshot.destroy_tower_income(tower_count)
                    tower_count -= 1
                else:
                    wallet += snapshot.downgrade_tower_income(int(tower.type))
                plan.append(Operation(OperationType.DOWNGRADE_TOWER, tower_id))
                if wallet >= coin_need:
                    valid = True
                    break

            if not valid:
                continue

            for op in plan:
                trial.add_operation_of_player(self.side, op)
            trial.apply_operations_of_player(self.side)
            window = 48
            base_hp = snapshot.bases[self.side].hp
            for step in range(1, 49):
                if not trial.fast_next_round(self.side):
                    break
                if snapshot.bases[self.side].hp < base_hp:
                    window = step
                    break
            if window > max_round:
                max_round = window
                max_coins = wallet
                best_ops = plan

        if max_round < min(24, fallback_round):
            return None
        return best_ops, max_coins, towers

    def _try_use_storm(self, info: GameInfo, all_in: bool) -> List[Operation]:
        if info.super_weapon_cd[self.side][int(SuperWeaponType.LIGHTNING_STORM)] > 0:
            return []

        cost = info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM))
        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        prefix: List[Operation] = []
        can_cast = wallet >= cost

        if not can_cast:
            liquidation = self._liquidate_all(wallet, tower_count, cost, info) if all_in else self._liquidate_cautious(wallet, tower_count, cost, info)
            if liquidation is not None:
                prefix, wallet, tower_count = liquidation
                can_cast = wallet >= cost
        if not can_cast:
            return []

        best_value = -1
        best_point: Optional[Tuple[int, int]] = None
        for x in range(19):
            for y in range(19):
                if not is_valid_pos(x, y):
                    continue
                trial = Simulator(info)
                for op in prefix:
                    trial.add_operation_of_player(self.side, op)
                trial.add_operation_of_player(self.side, Operation(OperationType.USE_LIGHTNING_STORM, x, y))
                trial.apply_operations_of_player(self.side)
                fail_round = 32
                for tick in range(32):
                    if not trial.fast_next_round(self.side):
                        break
                    if trial.info.bases[self.side].hp < info.bases[self.side].hp:
                        fail_round = tick
                        break
                if fail_round < 24:
                    continue
                value = trial.info.die_count[1 - self.side] + fail_round
                if value > best_value:
                    best_value = value
                    best_point = (x, y)

        if best_point is None:
            return []
        return [*prefix, Operation(OperationType.USE_LIGHTNING_STORM, best_point[0], best_point[1])]

    def _try_end_storm(self, info: GameInfo) -> List[Operation]:
        if info.super_weapon_cd[self.side][int(SuperWeaponType.LIGHTNING_STORM)] > 0:
            return []

        cost = info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM))
        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        prefix: List[Operation] = []
        can_cast = wallet >= cost

        if not can_cast:
            liquidation = self._liquidate_all(wallet, tower_count, cost, info)
            if liquidation is not None:
                prefix, wallet, tower_count = liquidation
                can_cast = wallet >= cost
        if not can_cast:
            return []

        x, y = SITE_LAYOUT[self.side][STORM_SLOT]
        return [*prefix, Operation(OperationType.USE_LIGHTNING_STORM, x, y)]

    def _try_use_superweapon(self, info: GameInfo) -> List[Operation]:
        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        can_emp = (
            info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)] == 0
            and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.EMP_BLASTER))
        )
        can_deflect = (
            info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)] == 0
            and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.DEFLECTOR))
        )
        can_eva = (
            info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)] == 0
            and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.EMERGENCY_EVASION))
        )
        enemy_storm = (
            info.super_weapon_cd[1 - self.side][int(SuperWeaponType.LIGHTNING_STORM)] == 0
            and info.coins[1 - self.side] >= info.use_super_weapon_cost(int(SuperWeaponType.LIGHTNING_STORM))
        )

        prefix: List[Operation] = []
        if not can_emp and info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)] == 0:
            sale = self._liquidate_cautious(wallet, tower_count, 150, info)
            if sale is not None:
                prefix, wallet, tower_count = sale

        if (
            not prefix
            and (
                (info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)] == 0 and not can_deflect)
                or (
                    info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)] == 0
                    and not can_eva
                )
            )
        ):
            sale = self._liquidate_cautious(wallet, tower_count, 100, info)
            if sale is not None:
                prefix, wallet, tower_count = sale

        can_emp = (
            info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)] == 0
            and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.EMP_BLASTER))
        )
        can_deflect = (
            info.super_weapon_cd[self.side][int(SuperWeaponType.DEFLECTOR)] == 0
            and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.DEFLECTOR))
        )
        can_eva = (
            info.super_weapon_cd[self.side][int(SuperWeaponType.EMERGENCY_EVASION)] == 0
            and wallet >= info.use_super_weapon_cost(int(SuperWeaponType.EMERGENCY_EVASION))
        )

        preview = Simulator(info)
        for _ in range(24):
            if not preview.fast_next_round(1 - self.side):
                break
        base_enemy_hp = preview.info.bases[1 - self.side].hp
        base_die_count = preview.info.die_count[self.side]

        reserved_emp_targets: List[Tuple[int, int, float]] = []
        if can_emp:
            results: List[Tuple[int, int, float]] = []
            for x in range(19):
                for y in range(19):
                    if not is_valid_pos(x, y):
                        continue
                    value = 0.0
                    for tower in info.towers:
                        if tower.player == 1 - self.side and distance(tower.x, tower.y, x, y) <= 3:
                            if tower.type == TowerType.BASIC:
                                value += 50
                            elif int(tower.type) // 10 < 0:
                                value += 60
                            else:
                                value += 80
                    if value < 100:
                        continue
                    trial = Simulator(info)
                    for op in prefix:
                        trial.add_operation_of_player(self.side, op)
                    trial.add_operation_of_player(self.side, Operation(OperationType.USE_EMP_BLASTER, x, y))
                    trial.apply_operations_of_player(self.side)
                    for _ in range(24):
                        if not trial.fast_next_round(1 - self.side):
                            break
                    if self.current_round > 495:
                        if trial.info.bases[1 - self.side].hp >= base_enemy_hp:
                            continue
                    elif self.current_round > 460:
                        if trial.info.bases[1 - self.side].hp >= base_enemy_hp - 2:
                            continue
                    elif trial.info.bases[1 - self.side].hp >= base_enemy_hp - 4:
                        continue
                    value += 100 * (base_enemy_hp - trial.info.bases[1 - self.side].hp)
                    for site in range(1, 34):
                        sx, sy = SITE_LAYOUT[1 - self.side][site]
                        if distance(sx, sy, x, y) <= 3:
                            bx, by = SITE_LAYOUT[1 - self.side][HOME_SLOT]
                            value += 3 - distance(sx, sy, bx, by) * 0.01
                    results.append((x, y, value))

            if results and not enemy_storm:
                x, y, _ = max(results, key=lambda item: item[2])
                self._mark_super(SuperWeaponType.EMP_BLASTER)
                return [*prefix, Operation(OperationType.USE_EMP_BLASTER, x, y)]
            reserved_emp_targets = results

        if can_deflect or can_eva:
            results: List[Tuple[int, int, float, bool]] = []
            if can_eva:
                for x in range(19):
                    for y in range(19):
                        if not is_valid_pos(x, y):
                            continue
                        value = 0.0
                        count = 0
                        min_dis = 100
                        for ant in info.ants:
                            if ant.player == self.side and distance(ant.x, ant.y, x, y) <= 3 and ant.is_alive():
                                value += ant.level + 1
                                count += 1
                                gap = distance(
                                    ant.x,
                                    ant.y,
                                    SITE_LAYOUT[1 - self.side][HOME_SLOT][0],
                                    SITE_LAYOUT[1 - self.side][HOME_SLOT][1],
                                )
                                min_dis = min(min_dis, gap)
                        if self.current_round <= 506 and min_dis > 5:
                            continue
                        if count < 3 or (self.current_round > 460 and count < 2):
                            continue
                        trial = Simulator(info)
                        for op in prefix:
                            trial.add_operation_of_player(self.side, op)
                        trial.add_operation_of_player(self.side, Operation(OperationType.USE_EMERGENCY_EVASION, x, y))
                        trial.apply_operations_of_player(self.side)
                        for _ in range(24):
                            if not trial.fast_next_round(1 - self.side):
                                break
                        if self.current_round > 506:
                            if (
                                trial.info.bases[1 - self.side].hp >= base_enemy_hp
                                and trial.info.die_count[self.side] >= base_die_count - 2
                            ):
                                continue
                        elif self.current_round > 460:
                            if trial.info.bases[1 - self.side].hp >= base_enemy_hp - 2:
                                continue
                        elif trial.info.bases[1 - self.side].hp >= base_enemy_hp - 3:
                            continue
                        value += 100 * (base_enemy_hp - trial.info.bases[1 - self.side].hp)
                        results.append((x, y, value, True))

            if can_deflect and not results:
                bx, by = SITE_LAYOUT[1 - self.side][HOME_SLOT]
                sx, sy = SITE_LAYOUT[1 - self.side][STORM_SLOT]
                for x in range(19):
                    for y in range(19):
                        if not is_valid_pos(x, y):
                            continue
                        if distance(x, y, bx, by) > 4:
                            continue
                        value = 0.0
                        trial = Simulator(info)
                        for op in prefix:
                            trial.add_operation_of_player(self.side, op)
                        trial.add_operation_of_player(self.side, Operation(OperationType.USE_DEFLECTOR, x, y))
                        trial.apply_operations_of_player(self.side)
                        for _ in range(24):
                            if not trial.fast_next_round(1 - self.side):
                                break
                        if (
                            (self.current_round > 460 and trial.info.bases[1 - self.side].hp >= base_enemy_hp - 2)
                            or trial.info.bases[1 - self.side].hp >= base_enemy_hp - 3
                        ):
                            continue
                        value += 100 * (base_enemy_hp - trial.info.bases[1 - self.side].hp)
                        value -= distance(x, y, sx, sy)
                        results.append((x, y, value, False))

            if results:
                x, y, _, is_eva = max(results, key=lambda item: item[2])
                if is_eva:
                    self._mark_super(SuperWeaponType.EMERGENCY_EVASION)
                    return [*prefix, Operation(OperationType.USE_EMERGENCY_EVASION, x, y)]
                self._mark_super(SuperWeaponType.DEFLECTOR)
                return [*prefix, Operation(OperationType.USE_DEFLECTOR, x, y)]

        if can_emp and reserved_emp_targets:
            x, y, _ = max(reserved_emp_targets, key=lambda item: item[2])
            self._mark_super(SuperWeaponType.EMP_BLASTER)
            return [*prefix, Operation(OperationType.USE_EMP_BLASTER, x, y)]
        return []

    def _try_emp(self, info: GameInfo) -> List[Operation]:
        if self._nearest_push_distance(info) > 5:
            return []
        if info.super_weapon_cd[self.side][int(SuperWeaponType.EMP_BLASTER)] > 0:
            return []

        wallet = info.coins[self.side]
        tower_count = info.tower_num_of_player(self.side)
        enemy_wallet = info.coins[1 - self.side]
        prefix: List[Operation] = []

        if wallet - enemy_wallet < 100 or wallet < 150:
            sale = self._liquidate_cautious(wallet, tower_count, max(enemy_wallet + 100, 150), info)
            if sale is None:
                return []
            prefix, wallet, tower_count = sale

        own_preview = Simulator(info)
        for _ in range(24):
            if not own_preview.fast_next_round(self.side):
                break
            if own_preview.info.bases[self.side].hp < info.bases[self.side].hp:
                return []

        preview = Simulator(info)
        for _ in range(24):
            if not preview.fast_next_round(1 - self.side):
                break
        base_enemy_hp = preview.info.bases[1 - self.side].hp

        results: List[Tuple[int, int, float]] = []
        for x in range(19):
            for y in range(19):
                if MAP_PROPERTY[x][y] < 0:
                    continue
                value = 0.0
                for tower in info.towers:
                    if tower.player == 1 - self.side and distance(tower.x, tower.y, x, y) <= 3:
                        if tower.type == TowerType.BASIC:
                            value += 50
                        elif int(tower.type) // 10 < 0:
                            value += 60
                        else:
                            value += 80
                if value < 100:
                    continue
                trial = Simulator(info)
                for op in prefix:
                    trial.add_operation_of_player(self.side, op)
                trial.add_operation_of_player(self.side, Operation(OperationType.USE_EMP_BLASTER, x, y))
                trial.apply_operations_of_player(self.side)
                for _ in range(24):
                    if not trial.fast_next_round(1 - self.side):
                        break
                if trial.info.bases[1 - self.side].hp >= base_enemy_hp - 4:
                    continue
                value += 100 * (base_enemy_hp - trial.info.bases[1 - self.side].hp)
                for site in range(1, 34):
                    sx, sy = SITE_LAYOUT[1 - self.side][site]
                    if distance(sx, sy, x, y) <= 3:
                        bx, by = SITE_LAYOUT[1 - self.side][HOME_SLOT]
                        value += 3 - distance(sx, sy, bx, by) * 0.01
                results.append((x, y, value))

        if not results:
            return []
        x, y, _ = max(results, key=lambda item: item[2])
        self._mark_super(SuperWeaponType.EMP_BLASTER)
        return [*prefix, Operation(OperationType.USE_EMP_BLASTER, x, y)]

    def _try_attack(self, info: GameInfo) -> List[Operation]:
        if self.front_state == 0:
            return self._try_use_superweapon(info)

        if self.current_round <= 460:
            if info.bases[self.side].ant_level == 0:
                if info.coins[self.side] >= 200:
                    return [Operation(OperationType.UPGRADE_GENERATED_ANT)]
            elif info.bases[self.side].ant_level == 1:
                if info.coins[self.side] >= 250:
                    return [Operation(OperationType.UPGRADE_GENERATED_ANT)]
                sale = self._liquidate_cautious(
                    info.coins[self.side],
                    info.tower_num_of_player(self.side),
                    250,
                    info,
                )
                if sale is not None:
                    ops, _, _ = sale
                    return [*ops, Operation(OperationType.UPGRADE_GENERATED_ANT)]
            elif info.bases[self.side].gen_speed_level == 0:
                if info.coins[self.side] >= 200:
                    return [Operation(OperationType.UPGRADE_GENERATION_SPEED)]
                sale = self._liquidate_all(
                    info.coins[self.side],
                    info.tower_num_of_player(self.side),
                    200,
                    info,
                )
                if sale is not None:
                    ops, _, _ = sale
                    return [*ops, Operation(OperationType.UPGRADE_GENERATION_SPEED)]
            return self._try_use_superweapon(info)

        if self.current_round <= 470 and info.bases[self.side].ant_level == 0:
            if info.coins[self.side] >= 200:
                return [Operation(OperationType.UPGRADE_GENERATED_ANT)]
            return []
        return self._try_use_superweapon(info)

    def __call__(self, player_id: int, game_info: GameInfo) -> List[Operation]:
        self.current_round = game_info.round
        if self.current_round == 0:
            self.side = player_id

        enemy = 1 - self.side
        self.enemy_old_baseline = game_info.old_count[enemy]
        self.enemy_die_baseline = game_info.die_count[enemy]
        self.wall_hp_snapshot = game_info.bases[self.side].hp

        self.front_state = 0
        if game_info.bases[self.side].hp > game_info.bases[enemy].hp:
            self.front_state = 1
            self.assault_memory = False
        elif game_info.bases[self.side].hp < game_info.bases[enemy].hp:
            self.front_state = -1

        attack = self.front_state == -1
        own_pressure = float(game_info.die_count[enemy])
        enemy_pressure = float(game_info.die_count[self.side])
        live_weight = min(1.0, (512 - self.current_round) / 20.0)
        for ant in game_info.ants:
            if ant.player == enemy and ant.is_alive():
                own_pressure += live_weight
            elif ant.player == self.side and ant.is_alive():
                enemy_pressure += live_weight

        if not attack and self.front_state == 0:
            if own_pressure - enemy_pressure >= 4:
                self.assault_memory = False
            elif own_pressure - enemy_pressure <= -3 - max((450 - self.current_round) // 50, 0):
                attack = True
            elif self.assault_memory:
                attack = True
            elif self.current_round >= 450 and own_pressure - enemy_pressure <= 1:
                attack = True

        enemy_emp = -1
        for weapon in game_info.super_weapons:
            if weapon.player == enemy and weapon.type == SuperWeaponType.EMP_BLASTER:
                enemy_emp = weapon.left_time
                break

        if self.front_state <= 0 and not self.reserve_depth:
            ops = self._try_emp(game_info)
            if ops:
                return ops

        if attack and not self.reserve_depth:
            self.assault_memory = True
            ops = self._try_attack(game_info)
            if ops:
                return ops

        if self.front_state == 1 and self.current_round >= 488:
            ops = self._try_end_storm(game_info)
            if ops:
                return ops

        if self.front_state == 0 and self.current_round >= 510:
            ops = self._try_use_storm(game_info, True)
            if ops:
                return ops

        start_cpu = time.process_time()
        staging = game_info.clone()
        staging.bases[enemy].hp = 1000

        self.nodes = []
        root = ForecastNode(self, Simulator(staging))
        root.node_id = 0
        root.parent = -1
        root.evaluate()
        self.nodes.append(root)
        self.nodes[0].expand(is_root=True)

        while True:
            if time.process_time() - start_cpu >= SEARCH_BUDGET or len(self.nodes) >= MAX_NODE_COUNT - 10:
                break
            if not self._expand_one():
                break

        best_id = -1
        best_value = -1e9
        for child_id in root.children:
            child = self.nodes[child_id]
            if child.best_descendant > best_value:
                best_value = child.best_descendant
                best_id = child_id

        if len(self.nodes) > 1 and best_id > 1 and best_value - self.nodes[1].best_descendant < 2:
            best_id = 1
            best_value = self.nodes[1].best_descendant

        emergency_storm = False
        if best_id >= 0:
            imminent = self.nodes[best_id].collapse_round - self.current_round
            emergency_storm = (
                (
                    self.front_state >= 0
                    and (
                        (enemy_emp > 0 and imminent < min(8, enemy_emp) and best_value < -400)
                        or (best_value < -700 and imminent <= 2)
                    )
                )
                or (
                    self.front_state == 0
                    and game_info.die_count[enemy] - game_info.die_count[self.side] >= 8
                    and imminent <= 1
                )
            )

        if emergency_storm:
            ops = self._try_use_storm(game_info, self.current_round >= 480)
            if ops:
                return ops

        if best_id > 0:
            self.reserve_depth = self.nodes[best_id].best_depth
            return list(self.nodes[best_id].chosen)
        return []
