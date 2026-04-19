from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from SDK.utils.constants import (
    ANT_GENERATION_CYCLE,
    ANT_MAX_HP,
    LIGHTNING_STORM_ANT_DAMAGE,
    LIGHTNING_STORM_TOWER_DAMAGE,
    LIGHTNING_STORM_TOWER_INTERVAL,
    MAX_ACTIONS,
    OperationType,
    PLAYER_BASES,
    STRATEGIC_BUILD_ORDER,
    SUPER_WEAPON_STATS,
    SuperWeaponType,
    TOWER_STATS,
    TOWER_UPGRADE_TREE,
    TowerType,
)
from SDK.utils.features import FeatureExtractor
from SDK.utils.geometry import hex_distance
from SDK.backend.state import BackendState
from SDK.backend.model import Operation, Tower
from SDK.utils.turns import DecisionContext


@dataclass(slots=True)
class ActionBundle:
    name: str
    operations: tuple[Operation, ...] = ()
    score: float = 0.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def protocol_lines(self) -> list[list[int]]:
        return [op.to_protocol_tokens() for op in self.operations]


class ActionCatalog:
    def __init__(self, max_actions: int = MAX_ACTIONS, feature_extractor: FeatureExtractor | None = None) -> None:
        self.max_actions = max_actions
        self.feature_extractor = feature_extractor or FeatureExtractor(max_actions=max_actions)

    def build(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext | None = None,
        *,
        rerank: bool = True,
    ) -> list[ActionBundle]:
        if context is None:
            context = DecisionContext.for_player(player)
        bundles: list[ActionBundle] = [ActionBundle(name="hold", score=0.0, tags=("noop",))]
        bundles.extend(self._build_candidates(state, player))
        bundles.extend(self._upgrade_candidates(state, player))
        bundles.extend(self._downgrade_candidates(state, player))
        bundles.extend(self._base_upgrade_candidates(state, player))
        bundles.extend(self._superweapon_candidates(state, player))
        bundles.extend(self._paired_candidates(state, player, bundles[1:]))
        unique: dict[tuple[tuple[int, int, int], ...], ActionBundle] = {}
        for bundle in bundles:
            key = tuple((int(op.op_type), op.arg0, op.arg1) for op in bundle.operations)
            if key not in unique or bundle.score > unique[key].score:
                unique[key] = bundle
        ordered = sorted(unique.values(), key=lambda item: item.score, reverse=True)
        ordered = ordered[: min(len(ordered), self.max_actions * 2)]
        if not rerank:
            return ordered[: self.max_actions]
        reranked = self._rerank_with_one_step_rollout(state, player, context, ordered)
        return reranked[: self.max_actions]

    def action_mask(self, bundles: list[ActionBundle]) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=np.int8)
        mask[: len(bundles)] = 1
        return mask

    def bundle_for_index(self, bundles: list[ActionBundle], action_index: int) -> ActionBundle:
        if 0 <= action_index < len(bundles):
            return bundles[action_index]
        return bundles[0]

    def _build_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        tower_count = state.tower_count(player)
        build_cost = state.build_tower_cost(tower_count)
        if state.coins[player] < build_cost:
            return results
        for x, y in STRATEGIC_BUILD_ORDER[player]:
            op = Operation(OperationType.BUILD_TOWER, x, y)
            if not state.can_apply_operation(player, op):
                continue
            pressure = self._local_enemy_pressure(state, player, x, y)
            lane_bonus = state.slot_priority(player, x, y)
            score = lane_bonus + pressure * 2.5 - build_cost * 0.03
            results.append(ActionBundle(name=f"build@{x},{y}", operations=(op,), score=score, tags=("build",)))
        return results

    def _upgrade_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        enemy_base = PLAYER_BASES[1 - player]
        for tower in state.towers_of(player):
            local_density = self._local_enemy_pressure(state, player, tower.x, tower.y)
            for target in TOWER_UPGRADE_TREE.get(tower.tower_type, ()): 
                op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
                if not state.can_apply_operation(player, op):
                    continue
                fit = self._tower_type_fit(target, local_density, hex_distance(tower.x, tower.y, *enemy_base))
                score = fit + tower.level * 1.5 + state.slot_priority(player, tower.x, tower.y) * 0.15
                results.append(
                    ActionBundle(
                        name=f"upgrade#{tower.tower_id}->{int(target)}",
                        operations=(op,),
                        score=score,
                        tags=("upgrade", f"tower:{int(target)}"),
                    )
                )
        return results

    def _downgrade_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        for tower in state.towers_of(player):
            pressure = self._local_enemy_pressure(state, player, tower.x, tower.y)
            if pressure > 1.5:
                continue
            op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if not state.can_apply_operation(player, op):
                continue
            refund = state.operation_income(player, op)
            score = refund * 0.04 - state.slot_priority(player, tower.x, tower.y) * 0.3 - tower.level * 3.0
            results.append(ActionBundle(name=f"downgrade#{tower.tower_id}", operations=(op,), score=score, tags=("sell",)))
        return results

    def _base_upgrade_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        if state.bases[player].ant_level < 2:
            level = state.bases[player].ant_level
            hp_gain = ANT_MAX_HP[level + 1] - ANT_MAX_HP[level]
            if hp_gain > 0:
                op = Operation(OperationType.UPGRADE_GENERATED_ANT)
                if state.can_apply_operation(player, op):
                    score = 8.0 + hp_gain * 1.4 + state.frontline_distance(player) * 0.22 - state.round_index * 0.01 - level * 1.2
                    results.append(ActionBundle("upgrade-ant", (op,), score, ("base", "offense")))
        if state.bases[player].generation_level < 2:
            level = state.bases[player].generation_level
            current_cycle = ANT_GENERATION_CYCLE[level]
            next_cycle = ANT_GENERATION_CYCLE[level + 1]
            if next_cycle < current_cycle - 1e-6:
                op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
                if state.can_apply_operation(player, op):
                    tempo_gain = current_cycle - next_cycle
                    score = 10.0 + tempo_gain * 14.0 + state.nearest_ant_distance(player) * 0.12 - state.round_index * 0.015
                    results.append(ActionBundle("upgrade-gen", (op,), score, ("base", "tempo")))
        return results

    def _superweapon_candidates(self, state: BackendState, player: int) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        enemy = 1 - player
        enemy_ants = state.ants_of(enemy)
        my_ants = state.ants_of(player)
        enemy_towers = state.towers_of(enemy)

        if (enemy_ants or enemy_towers) and state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost:
            centers = {(ant.x, ant.y) for ant in enemy_ants}
            centers.update((tower.x, tower.y) for tower in enemy_towers)
            best = max(
                ((x, y, self._storm_value(state, player, x, y)) for x, y in centers),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 1.5:
                op = Operation(OperationType.USE_LIGHTNING_STORM, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"storm@{best[0]},{best[1]}", (op,), best[2], ("weapon", "storm")))

        if enemy_towers and state.weapon_cooldowns[player, SuperWeaponType.EMP_BLASTER] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost:
            centers = {(tower.x, tower.y) for tower in enemy_towers}
            scored = [
                (x, y, self._emp_value(state, player, x, y))
                for x, y in centers
            ]
            best = max(scored, key=lambda item: item[2], default=None)
            if best and best[2] > 2.0:
                op = Operation(OperationType.USE_EMP_BLASTER, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"emp@{best[0]},{best[1]}", (op,), best[2], ("weapon", "emp")))

        if my_ants and state.weapon_cooldowns[player, SuperWeaponType.DEFLECTOR] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost:
            best = max(
                ((ant.x, ant.y, self._deflector_value(state, player, ant.x, ant.y)) for ant in my_ants),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 1.5:
                op = Operation(OperationType.USE_DEFLECTOR, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"deflect@{best[0]},{best[1]}", (op,), best[2], ("weapon", "shield")))

        if my_ants and state.weapon_cooldowns[player, SuperWeaponType.EMERGENCY_EVASION] == 0 and state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost:
            best = max(
                ((ant.x, ant.y, self._evasion_value(state, player, ant.x, ant.y)) for ant in my_ants),
                key=lambda item: item[2],
                default=None,
            )
            if best and best[2] > 1.0:
                op = Operation(OperationType.USE_EMERGENCY_EVASION, best[0], best[1])
                if state.can_apply_operation(player, op):
                    results.append(ActionBundle(f"evasion@{best[0]},{best[1]}", (op,), best[2], ("weapon", "panic")))

        return results

    def _paired_candidates(self, state: BackendState, player: int, singles: list[ActionBundle]) -> list[ActionBundle]:
        results: list[ActionBundle] = []
        left = [bundle for bundle in singles if bundle.tags and bundle.tags[0] in {"sell", "build", "upgrade", "base"}]
        left = sorted(left, key=lambda item: item.score, reverse=True)[:8]
        for first in left:
            for second in left:
                if first is second:
                    continue
                operations = first.operations + second.operations
                if len(operations) > 2:
                    continue
                trial = state.clone()
                invalid_ops = trial.apply_operation_list(player, operations)
                if invalid_ops:
                    continue
                score = first.score + second.score * 0.9
                name = f"{first.name}+{second.name}"
                results.append(ActionBundle(name=name, operations=tuple(operations), score=score, tags=("combo",)))
        return results

    def _rerank_with_one_step_rollout(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext,
        bundles: list[ActionBundle],
    ) -> list[ActionBundle]:
        baseline = self.feature_extractor.evaluate(state, player, context=context)
        reranked: list[ActionBundle] = []
        for bundle in bundles:
            trial = state.clone()
            trial.apply_operation_list(player, bundle.operations)
            child_context = context.next_turn()
            if not trial.terminal:
                if context.settles_after_action:
                    trial.advance_round()
                else:
                    enemy = 1 - player
                    enemy_bundles = self.build(trial, enemy, child_context, rerank=False)
                    enemy_bundle = enemy_bundles[0] if enemy_bundles else ActionBundle(name="hold")
                    trial.apply_operation_list(enemy, enemy_bundle.operations)
                    if not trial.terminal:
                        trial.advance_round()
            rollout_value = self.feature_extractor.evaluate(trial, player, context=DecisionContext.for_player(player)) - baseline
            reranked.append(ActionBundle(bundle.name, bundle.operations, bundle.score + rollout_value * 0.2, bundle.tags))
        reranked.sort(key=lambda item: item.score, reverse=True)
        if not reranked:
            return [ActionBundle(name="hold")]
        return reranked

    def _local_enemy_pressure(self, state: BackendState, player: int, x: int, y: int) -> float:
        pressure = 0.0
        for ant in state.ants_of(1 - player):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance <= 6:
                pressure += max(0.0, 6.5 - distance) * (1.0 + ant.level * 0.4)
        return pressure

    def _tower_type_fit(self, tower_type: TowerType, local_density: float, forward_distance: int) -> float:
        if tower_type in (TowerType.HEAVY, TowerType.HEAVY_PLUS, TowerType.BEWITCH):
            return local_density * 1.1 - forward_distance * 0.1
        if tower_type in (TowerType.ICE, TowerType.PULSE):
            return local_density * 1.3
        if tower_type in (TowerType.MORTAR, TowerType.MORTAR_PLUS, TowerType.MISSILE):
            return local_density * 0.85 + max(0.0, 12 - forward_distance)
        if tower_type in (TowerType.QUICK, TowerType.QUICK_PLUS, TowerType.DOUBLE):
            return local_density * 0.9 + 4.0
        if tower_type == TowerType.SNIPER:
            return max(0.0, 18 - forward_distance) + local_density * 0.4
        if tower_type in (TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC):
            stats = TOWER_STATS[tower_type]
            cadence = 12.0 / max(stats.spawn_interval, 1)
            density_bonus = max(0.0, 10 - local_density) * 0.75
            forward_bonus = max(0.0, 16 - forward_distance) * 0.22
            branch_bonus = {
                TowerType.PRODUCER: 0.0,
                TowerType.PRODUCER_FAST: 0.5,
                TowerType.PRODUCER_SIEGE: 1.1,
                TowerType.PRODUCER_MEDIC: 1.0,
            }[tower_type]
            return density_bonus + forward_bonus + cadence + branch_bonus
        return 0.0

    def _storm_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        enemy = 1 - player
        stats = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM]
        tower_strikes = max(stats.duration // LIGHTNING_STORM_TOWER_INTERVAL, 1)
        total = 0.0
        for ant in state.ants_of(enemy):
            distance = hex_distance(x, y, ant.x, ant.y)
            if distance > stats.attack_range:
                continue
            immediate_damage = min(LIGHTNING_STORM_ANT_DAMAGE, ant.hp)
            sustained_damage = min(ant.hp, LIGHTNING_STORM_ANT_DAMAGE * tower_strikes)
            total += immediate_damage / max(ant.max_hp, 1) * (2.5 + ant.level)
            total += sustained_damage / max(ant.max_hp, 1) * (0.8 + ant.level * 0.4)
            if ant.hp <= LIGHTNING_STORM_ANT_DAMAGE:
                total += ant.kill_reward
            total += max(0.0, stats.attack_range + 1 - distance) * 0.2
        for tower in state.towers_of(enemy):
            distance = hex_distance(x, y, tower.x, tower.y)
            if distance > stats.attack_range:
                continue
            projected_damage = min(tower.hp, LIGHTNING_STORM_TOWER_DAMAGE * tower_strikes)
            total += projected_damage / max(tower.max_hp, 1) * (6.0 + tower.level * 2.5)
            total += max(0.0, stats.attack_range + 1 - distance) * 0.15
        return total - stats.cost * 0.03

    def _emp_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        total = 0.0
        for tower in state.towers_of(1 - player):
            distance = hex_distance(x, y, tower.x, tower.y)
            if distance <= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].attack_range:
                total += 3.0 + tower.level * 2.5
        return total - SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost * 0.025

    def _deflector_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        total = 0.0
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].attack_range:
                total += 0.8 + ant.level * 0.8
        total += max(0.0, 7 - state.nearest_ant_distance(player)) * 0.5
        return total - SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost * 0.02

    def _evasion_value(self, state: BackendState, player: int, x: int, y: int) -> float:
        total = 0.0
        for ant in state.ants_of(player):
            if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].attack_range:
                total += 0.6 + ant.level * 0.7
        total += max(0.0, 5 - state.nearest_ant_distance(player))
        return total - SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost * 0.02
